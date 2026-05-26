if (!requireNamespace("limma", quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager")
    BiocManager::install("limma")
}
library(limma)
if (!requireNamespace("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite")
}
library(jsonlite)

# --- Core Batch Removal Functions ---

remove_batch_desingn <- function(expr_numeric, final_meta_data, centrum, design) {
    expr_corrected <- removeBatchEffect(
        x = expr_numeric,
        batch = final_meta_data[[centrum]], # Use [[...]] to access column dynamically
        design = design
    )
    return(expr_corrected)
}

remove_batch <- function(expr_numeric, final_meta_data, centrum) {
    expr_corrected <- removeBatchEffect(
        x = expr_numeric,
        batch = final_meta_data[[centrum]] # Use [[...]]
    )
    return(expr_corrected)
}

remove_batch_covariates <- function(expr_numeric, cov) {
    expr_corrected <- removeBatchEffect(
        x = expr_numeric,
        covariates = cov
    )
    return(expr_corrected)
}

remove_batch_covariates_desingn <- function(expr_numeric, cov, design) {
    expr_corrected <- removeBatchEffect(
        x = expr_numeric,
        covariates = cov,
        design = design
    )
    return(expr_corrected)
}

create_pc_for_covariates <- function(expr_numeric, n_pcs) {
    n_pcs <- max(0, as.integer(n_pcs))
    if (n_pcs == 0) return(matrix(numeric(0), nrow = ncol(expr_numeric), ncol = 0,
                                dimnames = list(colnames(expr_numeric), NULL)))
    if (any(!is.finite(expr_numeric))) {
        stop("create_pc_for_covariates: NA/Inf detected in expression matrix; impute or drop before PCA.")
    }
    pca_res <- prcomp(t(expr_numeric), center = TRUE, scale. = TRUE)
    n_pcs <- min(n_pcs, ncol(pca_res$x))
    pcs <- pca_res$x[, seq_len(n_pcs), drop = FALSE]
    return(pcs)
}

validate_covariates <- function(cov, expr_numeric) {
    cov <- as.matrix(cov)
    if (is.null(rownames(cov))) {
        stop("Covariate matrix must have sample IDs as row names.")
    }
    if (!setequal(rownames(cov), colnames(expr_numeric))) {
        stop("Covariate matrix row names must match expression column/sample names.")
    }
    cov <- cov[colnames(expr_numeric), , drop = FALSE]
    if (!all(rownames(cov) == colnames(expr_numeric))) {
        stop("Covariate matrix rows are not aligned to expression columns.")
    }
    storage.mode(cov) <- "numeric"
    if (any(!is.finite(cov))) {
        stop("Covariate matrix contains NA/Inf after numeric conversion.")
    }
    return(cov)
}

# state map 1: batch, 2: batch + design, 3: covariates, 4: covariates + design 5: pcs, 6: pcs + design

run_limma_remove_batch_effect <- function(expr_numeric, final_meta_data, state, extra1 = NULL, extra2 = NULL) {
    # Filter expression data for non-zero variance genes
    gene_var <- apply(expr_numeric, 1, var, na.rm=TRUE)
    expr_filt <- expr_numeric[gene_var > 0, , drop=FALSE]
    
    # Match metadata to filtered expression columns
    final_meta_data <- final_meta_data[colnames(expr_filt), , drop=FALSE]

    # Check for consistency after filtering
    if (!all(colnames(expr_filt) == rownames(final_meta_data))) {
        stop("Sample names in filtered expression data and metadata do not match or are not ordered correctly.")
    }

    # Dispatch logic based on state
    expr_corrected <- NULL
    
    if (state == 1) {
        # State 1: Batch only (extra1 must be the batch column name, e.g., 'centrum')
        expr_corrected <- remove_batch(expr_filt, final_meta_data, extra1)
    } else if (state == 2) {
        # State 2: Batch + Design (extra1=batch col name, extra2=design formula column name, e.g., 'condition')
        design_formula <- formula(paste("~", extra2))
        design <- model.matrix(design_formula, data = final_meta_data)
        expr_corrected <- remove_batch_desingn(expr_filt, final_meta_data, extra1, design)
    } else if (state == 3) {
        # State 3: Covariates only (extra1 must be a covariate matrix/data.frame)
        extra1 <- validate_covariates(extra1, expr_filt)
        expr_corrected <- remove_batch_covariates(expr_filt, extra1)
    } else if (state == 4) {
        # State 4: Covariates + Design (extra1=cov matrix/df, extra2=design formula column name)
        extra1 <- validate_covariates(extra1, expr_filt)
        design_formula <- formula(paste("~", extra2))
        design <- model.matrix(design_formula, data = final_meta_data)
        expr_corrected <- remove_batch_covariates_desingn(expr_filt, extra1, design)
    } else if (state == 5) {
        # State 5: PCs as Covariates (extra1=number of PCs)
        pcs <- create_pc_for_covariates(expr_filt, extra1)
        expr_corrected <- remove_batch_covariates(expr_filt, pcs)
    } else if (state == 6) {
        # State 6: PCs + Design (extra1=number of PCs, extra2=design formula column name)
        pcs <- create_pc_for_covariates(expr_filt, extra1)
        design_formula <- formula(paste("~", extra2))
        design <- model.matrix(design_formula, data = final_meta_data)
        expr_corrected <- remove_batch_covariates_desingn(expr_filt, pcs, design)
    } else {
        stop(paste("Invalid state value:", state))
    }
    
    return(expr_corrected)
}


# --- MAIN FUNCTION CALLED BY PYTHON ---

# This function takes a single argument: the path to the JSON input file.
execute_limma_batch_correction <- function(json_filepath) {
    
    # Define a default failure output
    failure_output <- list(status = "error", message = "An unknown error occurred.")
    
    tryCatch({
        # Read and parse the JSON configuration
        config <- fromJSON(json_filepath, simplifyVector = FALSE)
        
        # Extract configuration parameters
        state <- config$state
        expr_filepath <- config$expr_filepath
        meta_filepath <- config$meta_filepath
        output_filepath <- config$output_filepath
        extra1 <- config$extra1 # Can be column name, matrix filepath, or number (for PCs)
        extra2 <- config$extra2 # Can be design column name or NULL
        
        # Load Expression Matrix (Gene names as row names, Sample names as col names)
        # Assuming the expression file is a CSV where the first column is Gene IDs
        expr_numeric <- read.csv(expr_filepath, row.names = 1, check.names = FALSE)
        expr_numeric <- as.matrix(expr_numeric)
        
        # Load Metadata (Sample names as row names, variables as columns)
        # Assuming metadata file is a CSV where the first column is Sample IDs
        final_meta_data <- read.csv(meta_filepath, row.names = 1, check.names = FALSE)
        
        # Handle Extra1 if it is a filepath (e.g., a covariate matrix)
        if (is.character(extra1) && endsWith(tolower(extra1), ".csv") && state %in% c(3, 4)) {
            cov_matrix <- read.csv(extra1, row.names = 1, check.names = FALSE)
            extra1 <- as.matrix(cov_matrix)
        }
        
        # Execute the main limma logic
        expr_corrected <- run_limma_remove_batch_effect(
            expr_numeric = expr_numeric,
            final_meta_data = final_meta_data,
            state = state,
            extra1 = extra1,
            extra2 = extra2
        )
        
        # Write the output to a file
        write.csv(expr_corrected, file = output_filepath, row.names = TRUE)
        
        # Return success JSON
        success_output <- list(
            status = "success",
            message = paste("Batch effect correction complete. Result saved to", output_filepath),
            filepath = output_filepath,
            corrected_dimensions = dim(expr_corrected)
        )
        return(toJSON(success_output, pretty = TRUE, auto_unbox = TRUE))
        
    }, error = function(e) {
        # Return error JSON
        failure_output$message <- paste("R Error:", e$message)
        return(toJSON(failure_output, pretty = TRUE, auto_unbox = TRUE))
    })
}
