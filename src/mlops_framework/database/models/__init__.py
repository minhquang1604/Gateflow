"""Database models package — and the canonical way to register them all.

A SQLAlchemy table only exists on ``Base.metadata`` once the module
declaring it has been imported. Anything that builds a schema from that
metadata — ``Base.metadata.create_all()`` in a test fixture, Alembic's
``target_metadata`` for autogenerate — therefore depends on every model
module having been imported first, and gets no error when one has not:
it simply builds a schema missing those tables.

This package was empty, so each of those call sites kept its own
hand-maintained list of thirteen imports that looked unused and were
not. They drifted, and the failure mode is nasty in both directions:

* ``tests/sdk/test_project.py``'s fixture never imported
  ``framework_setting``, so its tests only passed when some *other*
  test module had already imported it. Run that file on its own and it
  failed with ``no such table: framework_settings`` — a real,
  order-dependent bug that had been latent for as long as the fixture
  had existed;
* in ``alembic/env.py`` the same omission would have Alembic autogenerate
  a migration that drops every table it cannot see.

Importing this package registers all of them. ``__all__`` is what makes
the re-exports genuine exports rather than "unused imports" a linter
will offer to delete — which is the mistake this file exists to stop
anyone repeating.
"""

from mlops_framework.database.models.api_key import ApiKey
from mlops_framework.database.models.audit_log import AuditLog
from mlops_framework.database.models.dataset import Dataset
from mlops_framework.database.models.dataset_version import DatasetVersion
from mlops_framework.database.models.drift_evaluation import (
    DriftEvaluation,
    DriftOutcome,
)
from mlops_framework.database.models.framework_setting import FrameworkSetting
from mlops_framework.database.models.governance_event import (
    GovernanceEvent,
    GovernanceEventSeverity,
)
from mlops_framework.database.models.model import Model
from mlops_framework.database.models.model_promotion_event import (
    ModelPromotionEvent,
    ModelPromotionStatus,
)
from mlops_framework.database.models.model_version import ModelState, ModelVersion
from mlops_framework.database.models.readiness_evaluation import (
    ReadinessCheckOutcome,
    ReadinessEvaluation,
    ReadinessStatus,
)
from mlops_framework.database.models.retraining_decision import (
    DecisionRecordedBy,
    RetrainingDecision,
    RetrainingOutcomeStatus,
)
from mlops_framework.database.models.schedule import Schedule
from mlops_framework.database.models.serving_instance import ServingInstance
from mlops_framework.database.models.training_run import (
    RunStatus,
    TrainingRun,
    TriggerType,
)

__all__ = [
    "ApiKey",
    "AuditLog",
    "Dataset",
    "DatasetVersion",
    "DecisionRecordedBy",
    "DriftEvaluation",
    "DriftOutcome",
    "FrameworkSetting",
    "GovernanceEvent",
    "GovernanceEventSeverity",
    "Model",
    "ModelPromotionEvent",
    "ModelPromotionStatus",
    "ModelState",
    "ModelVersion",
    "ReadinessCheckOutcome",
    "ReadinessEvaluation",
    "ReadinessStatus",
    "RetrainingDecision",
    "RetrainingOutcomeStatus",
    "RunStatus",
    "Schedule",
    "ServingInstance",
    "TrainingRun",
    "TriggerType",
]
