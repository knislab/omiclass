# Native sva::sva training and sva::fsva(method="exact") application.
# No package installation, external labels, or fallback algorithms here.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Expected operation and working directory")
op <- args[[1]]
root <- args[[2]]
library(sva)
library(jsonlite)
options(mc.cores=1L, Ncpus=1L)
BiocParallel::register(BiocParallel::SerialParam(), default=TRUE)
settings <- fromJSON(file.path(root, "settings.json"))
read_expr <- function(path) as.matrix(read.csv(path, row.names=1, check.names=FALSE))
if (op == "fit") {
    dat <- read_expr(file.path(root, "expression.csv"))
    meta <- read.csv(file.path(root, "metadata.csv"), row.names=1, check.names=FALSE)
    meta <- meta[colnames(dat), , drop=FALSE]
    # The first biological design column is the categorical classification target.
    meta[[1]] <- factor(meta[[1]])
    mod <- model.matrix(~ ., data=meta)
    mod0 <- if (ncol(meta) == 1L) matrix(1, ncol(dat), 1L) else model.matrix(~ ., data=meta[, -1, drop=FALSE])
    if (qr(mod)$rank != ncol(mod)) stop("Biological design is rank deficient")
    if (ncol(dat) <= ncol(mod) + settings$n_factors) stop("Insufficient training samples for SVA design")
    set.seed(settings$random_state)
    sv <- sva::sva(dat, mod, mod0, n.sv=settings$n_factors)
    joint <- cbind(mod, sv$sv)
    condition <- kappa(joint, exact=TRUE)
    if (qr(joint)$rank != ncol(joint) || !is.finite(condition) || condition > 1e8)
        stop("Biology and surrogate variables are confounded or ill-conditioned")
    if (any(!is.finite(sv$pprob.b)) || any(!is.finite(sv$pprob.gam)))
        stop("SVA produced nonfinite empirical gene weights")
    # sva 3.50 errors on newdat=NULL because its return references newV.
    # Supply one database column and discard its new-sample result. `$db`
    # remains the native training correction and uses only the training fit.
    adjusted <- sva::fsva(dat, mod, sv, newdat=dat[, 1, drop=FALSE], method="exact")$db
    if (any(!is.finite(adjusted))) stop("Nonfinite native fSVA training correction")
    state <- list(dat=dat, mod=mod, sv=sv, package_version=as.character(packageVersion("sva")))
    saveRDS(state, file.path(root, "state.rds"))
    write.csv(adjusted, file.path(root, "corrected.csv"))
    write_json(list(sva_version=state$package_version, n_factors=sv$n.sv,
                    joint_condition=condition, implementation="sva::fsva(method='exact')",
                    factor_biology_r2=as.numeric(1 - colSums(qr.resid(qr(mod), as.matrix(sv$sv))^2) /
                                                  colSums(scale(as.matrix(sv$sv), scale=FALSE)^2)),
                    R_version=R.version.string), file.path(root, "diagnostics.json"), auto_unbox=TRUE)
} else if (op == "transform") {
    state <- readRDS(file.path(root, "state.rds"))
    if (as.character(packageVersion("sva")) != state$package_version)
        stop("Installed sva version differs from the fitted reference")
    dat <- read_expr(file.path(root, "expression.csv"))
    if (!identical(rownames(dat), rownames(state$dat))) stop("fSVA gene schema mismatch")
    # fsva exact performs its augmented SVD one incoming sample at a time.
    adjusted <- sva::fsva(state$dat, state$mod, state$sv, newdat=dat, method="exact")$new
    if (any(!is.finite(adjusted))) stop("Nonfinite native exact-fSVA external correction")
    dimnames(adjusted) <- dimnames(dat)
    write.csv(adjusted, file.path(root, "corrected.csv"))
} else stop("Unknown operation")
