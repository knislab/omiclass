"""
Custom exception hierarchy for the generic-classifier project.

These exceptions provide:
- Clean error boundaries between subsystem components (config → data → CV → tuning → R → reports)
- More readable failure messages at the CLI level
- Easier debugging and future HTML/JSON error reporting
"""

# ----- Base class -----
class GCError(Exception):
    """Base class for all generic-classifier errors."""
    pass


# ----- Config / Schema -----
class ConfigError(GCError):
    """Invalid or missing configuration."""
    pass


class SchemaValidationError(ConfigError):
    """Pydantic or schema validation failed."""
    pass


# ----- Data / Input -----
class DataError(GCError):
    """Errors related to input data, labels, or preprocessing."""
    pass


class DataShapeError(DataError):
    """X/y shape mismatch, wrong class count, invalid label encoding."""
    pass


class ImbalanceError(DataError):
    """Unsupported imbalance strategy or invalid threshold settings."""
    pass


# ----- Models / Parameters -----
class ModelBuildError(GCError):
    """Model builder failed to construct an estimator."""
    pass


class ParamError(ModelBuildError):
    """Invalid hyperparameter values or tuning configuration."""
    pass


# ----- CV / Tuning -----
class CVError(GCError):
    """Cross-validation or fold-level computation failed."""
    pass


class TuningError(GCError):
    """BayesSearchCV, grid search, or tuning pipeline error."""
    pass


# ----- R / External Backend -----
class RBackendError(GCError):
    """R backend failure (rpy2, limma, edgeR, etc.)."""
    pass


# ----- Reporting / Plotting -----
class ReportingError(GCError):
    """HTML/JSON report generation or figure-creation error."""
    pass

class SurvivalPredictionError(GCError):
    """Errors during survival prediction process."""
    pass
