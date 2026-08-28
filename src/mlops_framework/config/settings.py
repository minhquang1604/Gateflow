"""Application settings using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database configuration
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/mlops_framework",
        description="Database connection URL",
    )

    # Database pool settings
    database_pool_size: int = Field(
        default=5,
        description="Database connection pool size",
    )
    database_max_overflow: int = Field(
        default=10,
        description="Maximum number of connections that can be created beyond pool_size",
    )
    database_pool_timeout: int = Field(
        default=30,
        description="Seconds to wait for a connection from the pool",
    )
    database_echo: bool = Field(
        default=False,
        description="Echo SQL queries to stdout",
    )

    # MLflow tracking server
    mlflow_tracking_uri: str | None = Field(
        default=None,
        description="URI of the MLflow tracking server (e.g. http://mlflow:5000).",
    )
    mlflow_experiment_name: str = Field(
        default="mlops-framework",
        description="MLflow experiment name used by MLflowTracker.",
    )
    mlflow_artifact_root: str | None = Field(
        default=None,
        description="Default MLflow artifact root (informational only).",
    )
    mlflow_s3_endpoint_url: str | None = Field(
        default=None,
        description="S3-compatible endpoint URL used by MLflow artifact uploads.",
    )

    # Airflow orchestrator
    airflow_base_url: str | None = Field(
        default=None,
        description="Base URL of the Airflow webserver (used by AirflowOrchestrator).",
    )
    airflow_username: str = Field(
        default="airflow",
        description="Username for the Airflow REST API.",
    )
    airflow_password: str = Field(
        default="airflow",
        description="Password for the Airflow REST API.",
    )
    airflow_remote_log_base: str | None = Field(
        default=None,
        description=(
            "s3://bucket/prefix that Airflow's own remote task logging "
            "writes under (same value as its AIRFLOW__LOGGING__REMOTE_"
            "BASE_LOG_FOLDER). When set, AirflowOrchestrator.get_task_log "
            "reads the object directly from S3 instead of proxying "
            "through the Airflow webserver's REST endpoint — see "
            "mlops_framework.orchestration.airflow's module docstring."
        ),
    )

    # Write-gate for the small set of Gateflow endpoints that mutate an
    # external system (Airflow retry/clear — see api/security.py). Unset by
    # default: those endpoints stay disabled until an operator opts in, so
    # adding this setting does not by itself open anything up. Not real
    # authentication — see api/security.py's module docstring.
    console_write_token: str | None = Field(
        default=None,
        description=(
            "Shared secret required in the X-Console-Token header for "
            "Gateflow's write endpoints (e.g. Airflow task retry/clear). "
            "Unset disables those endpoints entirely."
        ),
    )

    # FastAPI ServingBridge (used by the HTTP event publisher)
    serving_bridge_url: str | None = Field(
        default=None,
        description="Base URL of the FastAPI ServingBridge.",
    )

    # Telegram admin-approval gate (used by the drift/retrain demo script
    # to block an automated retrain behind a human Approve/Deny before it
    # runs — see mlops_framework/approval/telegram.py).
    telegram_bot_token: str | None = Field(
        default=None,
        description="Telegram bot token used to send the retrain approval request.",
    )
    telegram_admin_chat_id: str | None = Field(
        default=None,
        description="Telegram chat_id of the admin who approves/denies a retrain.",
    )
    telegram_approval_timeout_seconds: float = Field(
        default=3600.0,
        description="How long to wait for the admin's Approve/Deny before treating a retrain as denied.",
    )

    # Slack admin-approval gate (see approval/slack.py). Slack has no
    # equivalent of Telegram's getUpdates for button presses: a click is
    # delivered to *you*, so the gate needs either a public endpoint or a
    # Socket Mode connection. It uses the latter, which is why there are
    # two tokens — a bot token to post the message and an app-level token
    # to open the socket the answer arrives on.
    slack_bot_token: str | None = Field(
        default=None,
        description="Slack bot token (xoxb-...) used to post the approval request.",
    )
    slack_app_token: str | None = Field(
        default=None,
        description="Slack app-level token (xapp-...) with connections:write, "
                    "used to open the Socket Mode connection the answer arrives on.",
    )
    slack_approval_channel: str | None = Field(
        default=None,
        description="Slack channel id the approval request is posted to.",
    )
    slack_approval_timeout_seconds: float = Field(
        default=3600.0,
        description="How long to wait for Approve/Deny before treating a retrain as denied.",
    )

    # Scheduling (cron-triggered automatic retraining — see
    # scheduling/runner.py and api/app.py's _start_scheduler). Off by
    # default: a background loop that can trigger real training runs
    # has no business starting itself in every test that builds an app,
    # only in the actual deployed service (docker-compose sets
    # SCHEDULER_ENABLED=true on the app container).
    scheduler_enabled: bool = Field(
        default=False,
        description="Run the background loop that fires due Schedules automatically.",
    )
    scheduler_poll_seconds: float = Field(
        default=60.0,
        description="How often the scheduler loop checks for due schedules.",
    )

    # Application settings
    app_name: str = Field(
        default="mlops-framework",
        description="Application name",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version",
    )

    # Development settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings instance
    """
    return Settings()
