if (!requireNamespace("limma", quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager")
    BiocManager::install("limma")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite")
}
if (!requireNamespace("edgeR", quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager")
    BiocManager::install("edgeR")
}

library(limma)
library(jsonlite)
library(edgeR) 

run_limma_voom_de <- function(expr_counts, final_meta_data, config) {
    
    # Extract config parameters
    design_str <- config$design_str
    design_col <- config$design_col
    min_c <- as.numeric(config$min_count)
    min_s_frac <- as.numeric(config$min_samples_frac)
    logfc_thresh <- as.numeric(config$logfc_threshold)
    fdr_thresh <- as.numeric(config$fdr_threshold)

    # 1. Prepare DGEList and Filter Low Counts
    dge <- DGEList(counts=expr_counts, samples=final_meta_data)
    min_samples <- ceiling(ncol(expr_counts) * min_s_frac)
    keep <- filterByExpr(dge,
                        group = final_meta_data[[design_col]],
                        min.count = min_c,
                        min.prop = min_s_frac)
    keep[is.na(keep)] <- FALSE
    dge <- dge[keep,,keep.lib.sizes=FALSE]
    
    # Match metadata to filtered expression columns
    final_meta_data <- final_meta_data[colnames(dge$counts), , drop=FALSE]
    
    # 2. Normalization (TMM)
    dge <- calcNormFactors(dge)
    
    # 3. Create Design Matrix (0 + for contrasts)
    design_formula_str <- paste("~ 0 +", design_str)
    final_meta_data[[design_col]] <- factor(as.character(final_meta_data[[design_col]]))
    design <- model.matrix(as.formula(design_formula_str), data = final_meta_data)

    # Rename ONLY the columns that correspond to the design_col factor
    lvls <- levels(final_meta_data[[design_col]])
    new_cluster_cols <- make.names(paste0(design_col, lvls), unique = TRUE)

    idx <- grep(paste0("^", design_col), colnames(design))
    if (length(idx) != length(lvls)) {
        stop(paste0(
            "Mismatch: expected ", length(lvls), " columns for ", design_col,
            " but found ", length(idx), ". design cols: ",
            paste(colnames(design), collapse = ", ")
        ))
    }
    colnames(design)[idx] <- new_cluster_cols


    
    # 4. Voom Transformation
    v <- voom(dge, design, plot = FALSE)
    
    # 5. Fit Linear Model and Empirical Bayes
    fit <- lmFit(v, design)
    fit <- eBayes(fit, trend = TRUE) 

    # 6. Apply Automated Contrasts

    # 6.1 identify the cluster coefficient columns produced by model.matrix
    cluster_cols <- grep(paste0("^", design_col), colnames(design), value = TRUE)

    if (length(cluster_cols) < 2) {
        stop(paste0("Need >=2 levels in ", design_col, ". Found: ", paste(cluster_cols, collapse=", ")))
    }

    contrast_mode <- if (is.null(config$contrast_mode)) "one_vs_rest" else config$contrast_mode
    lvls_by_col <- setNames(lvls, new_cluster_cols)
    coef_by_level <- setNames(new_cluster_cols, lvls)

    if (identical(contrast_mode, "pairwise")) {
        pairs <- config$pairwise_pairs
        if (is.null(pairs) || length(pairs) == 0) {
            stop("pairwise contrast_mode requires config$pairwise_pairs")
        }
        contrast_strings <- vapply(pairs, function(pair) {
            class_a <- as.character(pair$class_a)
            class_b <- as.character(pair$class_b)
            coef_a <- coef_by_level[[class_a]]
            coef_b <- coef_by_level[[class_b]]
            if (is.null(coef_a) || is.null(coef_b) || !(coef_a %in% cluster_cols) || !(coef_b %in% cluster_cols)) {
                stop(paste0("Pairwise classes not found in design: ", class_a, ", ", class_b))
            }
            paste0(coef_a, " - ", coef_b)
        }, character(1))
        contrast_labels <- vapply(pairs, function(pair) {
            paste0(as.character(pair$class_a), " vs ", as.character(pair$class_b))
        }, character(1))
    } else {
        # 6.2 build ONE-vs-REST contrasts
        contrast_strings <- vapply(cluster_cols, function(target) {
            others <- setdiff(cluster_cols, target)
            others_part <- paste(others, collapse = " + ")
            paste0(target, " - (", others_part, ") / ", length(others))
        }, character(1))
        contrast_labels <- vapply(cluster_cols, function(target) {
            paste0(lvls_by_col[[target]], " vs all")
        }, character(1))
    }

    contrast_matrix <- makeContrasts(contrasts = contrast_strings, levels = design)

    fit2 <- contrasts.fit(fit, contrast_matrix)
    fit2 <- eBayes(fit2, trend = TRUE)

    toptable_list <- list()
    for (i in seq_len(ncol(contrast_matrix))) {
        res <- topTable(fit2, coef = i, number = Inf, sort.by = "P")
        res$gene_id <- rownames(res)
        rownames(res) <- NULL
        res$Contrast <- contrast_labels[[i]]
        toptable_list[[i]] <- res
    }
    toptable_res <- do.call(rbind, toptable_list)

    return(toptable_res)
}

run_limma_de <- function(expr_matrix, final_meta_data, config) {
    design_str <- config$design_str
    design_col <- config$design_col
    logfc_thresh <- as.numeric(config$logfc_threshold)
    fdr_thresh <- as.numeric(config$fdr_threshold)

    design_formula_str <- paste("~ 0 +", design_str)
    final_meta_data[[design_col]] <- factor(as.character(final_meta_data[[design_col]]))
    design <- model.matrix(as.formula(design_formula_str), data = final_meta_data)

    lvls <- levels(final_meta_data[[design_col]])
    new_cluster_cols <- make.names(paste0(design_col, lvls), unique = TRUE)
    idx <- grep(paste0("^", design_col), colnames(design))
    if (length(idx) != length(lvls)) {
        stop(paste0(
            "Mismatch: expected ", length(lvls), " columns for ", design_col,
            " but found ", length(idx), ". design cols: ",
            paste(colnames(design), collapse = ", ")
        ))
    }
    colnames(design)[idx] <- new_cluster_cols

    fit <- lmFit(expr_matrix, design)
    fit <- eBayes(fit, trend = TRUE)

    cluster_cols <- grep(paste0("^", design_col), colnames(design), value = TRUE)
    if (length(cluster_cols) < 2) {
        stop(paste0("Need >=2 levels in ", design_col, ". Found: ", paste(cluster_cols, collapse=", ")))
    }

    contrast_mode <- if (is.null(config$contrast_mode)) "one_vs_rest" else config$contrast_mode
    lvls_by_col <- setNames(lvls, new_cluster_cols)
    coef_by_level <- setNames(new_cluster_cols, lvls)

    if (identical(contrast_mode, "pairwise")) {
        pairs <- config$pairwise_pairs
        if (is.null(pairs) || length(pairs) == 0) {
            stop("pairwise contrast_mode requires config$pairwise_pairs")
        }
        contrast_strings <- vapply(pairs, function(pair) {
            class_a <- as.character(pair$class_a)
            class_b <- as.character(pair$class_b)
            coef_a <- coef_by_level[[class_a]]
            coef_b <- coef_by_level[[class_b]]
            if (is.null(coef_a) || is.null(coef_b) || !(coef_a %in% cluster_cols) || !(coef_b %in% cluster_cols)) {
                stop(paste0("Pairwise classes not found in design: ", class_a, ", ", class_b))
            }
            paste0(coef_a, " - ", coef_b)
        }, character(1))
        contrast_labels <- vapply(pairs, function(pair) {
            paste0(as.character(pair$class_a), " vs ", as.character(pair$class_b))
        }, character(1))
    } else {
        contrast_strings <- vapply(cluster_cols, function(target) {
            others <- setdiff(cluster_cols, target)
            others_part <- paste(others, collapse = " + ")
            paste0(target, " - (", others_part, ") / ", length(others))
        }, character(1))
        contrast_labels <- vapply(cluster_cols, function(target) {
            paste0(lvls_by_col[[target]], " vs all")
        }, character(1))
    }

    contrast_matrix <- makeContrasts(contrasts = contrast_strings, levels = design)
    fit2 <- contrasts.fit(fit, contrast_matrix)
    fit2 <- eBayes(fit2, trend = TRUE)

    toptable_list <- list()
    for (i in seq_len(ncol(contrast_matrix))) {
        res <- topTable(fit2, coef = i, number = Inf, sort.by = "P")
        res$gene_id <- rownames(res)
        rownames(res) <- NULL
        res$Contrast <- contrast_labels[[i]]
        toptable_list[[i]] <- res
    }
    toptable_res <- do.call(rbind, toptable_list)
    return(toptable_res)
}

# --- MAIN FUNCTION CALLED BY PYTHON ---

execute_limma_voom_de <- function(json_filepath) {
    
    failure_output <- list(status = "error", message = "An unknown error occurred.")
    
    tryCatch({
        config <- fromJSON(json_filepath, simplifyVector = FALSE)
        
        expr_counts <- read.csv(config$expr_filepath, row.names = 1, check.names = FALSE)
        expr_counts <- as.matrix(expr_counts)
        final_meta_data <- read.csv(config$meta_filepath, row.names = 1, check.names = FALSE)
        
        de_results_df <- run_limma_voom_de(expr_counts, final_meta_data, config)
        
        write.csv(de_results_df, file = config$output_filepath, row.names = FALSE)
        
        success_output <- list(
            status = "success",
            message = paste("limma-voom DE analysis complete. Result saved to", config$output_filepath),
            filepath = config$output_filepath,
            n_genes = nrow(de_results_df)
        )
        return(toJSON(success_output, pretty = TRUE, auto_unbox = TRUE))
        
    }, error = function(e) {
        failure_output$message <- paste("R Error:", e$message)
        return(toJSON(failure_output, pretty = TRUE, auto_unbox = TRUE))
    })
}

execute_limma_de <- function(json_filepath) {
    failure_output <- list(status = "error", message = "An unknown error occurred.")
    tryCatch({
        config <- fromJSON(json_filepath, simplifyVector = FALSE)
        expr_matrix <- read.csv(config$expr_filepath, row.names = 1, check.names = FALSE)
        expr_matrix <- as.matrix(expr_matrix)
        final_meta_data <- read.csv(config$meta_filepath, row.names = 1, check.names = FALSE)

        de_results_df <- run_limma_de(expr_matrix, final_meta_data, config)
        write.csv(de_results_df, file = config$output_filepath, row.names = FALSE)

        success_output <- list(
            status = "success",
            message = paste("limma DE analysis complete. Result saved to", config$output_filepath),
            filepath = config$output_filepath,
            n_genes = nrow(de_results_df)
        )
        return(toJSON(success_output, pretty = TRUE, auto_unbox = TRUE))
    }, error = function(e) {
        failure_output$message <- paste("R Error:", e$message)
        return(toJSON(failure_output, pretty = TRUE, auto_unbox = TRUE))
    })
}
