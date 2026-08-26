"""MLOps Framework exceptions."""


class MLopsFrameworkError(Exception):
    """Base exception for MLOps Framework."""

    pass


class DatasetError(MLopsFrameworkError):
    """Exception raised for dataset-related errors."""

    pass


class DatasetNotFoundError(DatasetError):
    """Exception raised when a dataset is not found."""

    pass


class DuplicateDatasetNameError(DatasetError):
    """Exception raised when attempting to create a dataset with a duplicate name."""

    pass


class DatasetVersionError(MLopsFrameworkError):
    """Exception raised for dataset version-related errors."""

    pass


class DatasetVersionNotFoundError(DatasetVersionError):
    """Exception raised when a dataset version is not found."""

    pass


class ImmutableDatasetVersionError(DatasetVersionError):
    """Exception raised when attempting to modify an immutable dataset version."""

    pass


class InvalidVersionNumberError(DatasetVersionError):
    """Exception raised when an invalid version number is provided."""

    pass


class TrainingRunError(MLopsFrameworkError):
    """Exception raised for training run-related errors."""

    pass


class TrainingRunNotFoundError(TrainingRunError):
    """Exception raised when a training run is not found."""

    pass


class InvalidStatusTransitionError(TrainingRunError):
    """Exception raised when an invalid status transition is attempted."""

    pass


class ChecksumError(MLopsFrameworkError):
    """Exception raised for checksum-related errors."""

    pass


class SchemaHashError(MLopsFrameworkError):
    """Exception raised for schema hash-related errors."""

    pass


# --- Orchestration ----------------------------------------------------- #


class OrchestrationError(MLopsFrameworkError):
    """Base exception for orchestrator-related errors."""

    pass


class ExecutionNotFoundError(OrchestrationError):
    """Raised when an orchestrator has no record of the given execution ID."""

    pass


class OrchestratorConfigError(OrchestrationError):
    """Raised when the orchestrator is mis-configured or cannot run."""

    pass


# --- Experiment tracking ----------------------------------------------- #


class ExperimentTrackingError(MLopsFrameworkError):
    """Base exception for experiment tracker errors."""

    pass


# --- Model lifecycle --------------------------------------------------- #


class ModelError(MLopsFrameworkError):
    """Base exception for model lifecycle errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a Model is not found."""

    pass


class ModelVersionNotFoundError(ModelError):
    """Raised when a ModelVersion is not found."""

    pass


class DuplicateModelNameError(ModelError):
    """Raised when attempting to create a Model with a duplicate name."""

    pass


class InvalidModelStateTransitionError(ModelError):
    """Raised when an invalid model lifecycle transition is attempted."""

    pass


class ConcurrentPromotionError(ModelError):
    """Raised when promoting a ModelVersion to PRODUCTION loses a race.

    The transition was valid in isolation (``validate_transition`` passed),
    but another writer promoted a different version of the same model to
    PRODUCTION first — caught via the database's
    ``uq_model_versions_one_production_per_model`` partial unique index
    rather than left as a raw ``IntegrityError``.
    """

    pass


class ApiKeyError(MLopsFrameworkError):
    """Raised when an API key cannot be minted or revoked as asked.

    Not an authentication failure — a caller presenting a bad key gets
    ``None`` from ``ApiKeyManager.resolve`` and a 401 from the
    dependency, deliberately without being told which part was wrong.
    This is the *management* side: an unknown scope, a duplicate name, a
    key that does not exist.
    """

    pass


class RollbackError(ModelError):
    """Raised when a ModelVersion cannot be rolled back to.

    Distinct from :class:`InvalidModelStateTransitionError`: that one is
    the state machine refusing an edge, this one is
    ``ModelManager.rollback_to`` refusing the *operation* — the version
    is already in production, or is CANDIDATE/REJECTED and so has never
    been a known-good production version to return to.
    """

    pass


# --- Readiness / Eligibility / Drift / Promotion ---------------------- #


class ReadinessError(MLopsFrameworkError):
    """Base exception for dataset-readiness errors."""

    pass


class EligibilityError(MLopsFrameworkError):
    """Base exception for training-eligibility errors."""

    pass


class NotEligibleError(EligibilityError):
    """Raised when a training-eligibility policy rejects a retraining request.

    Carries an explainable list of reasons for the rejection.
    """

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__(
            "Training not eligible: " + "; ".join(self.reasons)
        )


class DriftError(MLopsFrameworkError):
    """Base exception for drift-detection errors."""

    pass


class UnrelatedDriftEvidenceError(DriftError):
    """Raised when a caller attaches drift evidence to a decision that the
    evidence does not concern.

    A governance record whose cited evidence is unrelated to the data it
    judged is worse than one citing none: it reads as substantiated and
    is not. The store therefore refuses the link rather than storing a
    claim it cannot support.
    """

    pass


class PromotionPolicyError(MLopsFrameworkError):
    """Base exception for model-promotion policy errors."""

    pass


class ModelNotApprovedError(PromotionPolicyError):
    """Raised when a promotion policy rejects a model.

    Carries an explainable list of reasons.
    """

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__(
            "Model not approved for promotion: " + "; ".join(self.reasons)
        )


# --- Events / Serving ------------------------------------------------ #


class EventPublisherError(MLopsFrameworkError):
    """Base exception for event publishing errors."""

    pass


class ServingError(MLopsFrameworkError):
    """Base exception for serving-bridge errors."""

    pass


# --- Scheduling -------------------------------------------------------- #


class ScheduleError(MLopsFrameworkError):
    """Base exception for schedule lifecycle errors."""

    pass


class ScheduleNotFoundError(ScheduleError):
    """Raised when a Schedule is not found."""

    pass


class InvalidCronExpressionError(ScheduleError):
    """Raised when a cron expression is not valid 5-field cron syntax."""

    pass
