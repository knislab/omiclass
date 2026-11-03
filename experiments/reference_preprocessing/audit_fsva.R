# Reproduce the native incoming-sample independence audit without real patients.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L || file.exists(args[[1]])) stop("Supply a new output JSON path")
library(sva)
set.seed(11)
n <- 30; g <- 100
x <- matrix(rnorm(g*n), g, n)
mod <- cbind(1, rep(c(0,1), 15))
w <- matrix(rnorm(n), n, 1)
x <- x + matrix(rnorm(g), g, 1) %*% t(w)
sv <- list(sv=w/sqrt(sum(w^2)), n.sv=1,
           pprob.b=rep(0,g), pprob.gam=rep(1,g))
a <- matrix(rnorm(g*3), g, 3)
b <- cbind(a, matrix(rnorm(g*2)+5, g, 2))
f <- function(z, method) sva::fsva(x,mod,sv,z,method=method)$new
result <- list(sva_version=as.character(packageVersion("sva")),
    fast_cohort_dependence=max(abs(f(a,"fast")[,1]-f(b,"fast")[,1])),
    fast_singleton_finite=all(is.finite(f(a[,1,drop=FALSE],"fast"))),
    exact_cohort_dependence=max(abs(f(a,"exact")[,1]-f(b,"exact")[,1])),
    exact_singleton_difference=max(abs(f(a,"exact")[,1]-f(a[,1,drop=FALSE],"exact")[,1])))
jsonlite::write_json(result, args[[1]], auto_unbox=TRUE, pretty=TRUE, digits=15)
print(result)
