from __future__ import annotations

from typing import Any


class DPSQLError(Exception):
    """
    Base DPSQL exception. All exceptions provide code / context / hint.
    """

    default_code = "dpsql_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.context = context or {}
        self.hint = hint
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": str(self),
            "hint": self.hint,
            "context": self.context,
            "cause_type": type(self.cause).__name__ if self.cause else None,
        }


# --- Validation / Parsing ---
class ValidationError(DPSQLError):
    default_code = "validation_error"


class QueryParseError(ValidationError):
    default_code = "query_parse_error"


class UnsupportedQueryError(ValidationError):
    default_code = "unsupported_query"


class PrivacyConstraintError(ValidationError):
    default_code = "privacy_constraint_violation"


# --- Privacy / Accounting ---
class PrivacyError(DPSQLError):
    default_code = "privacy_error"


class InsufficientPrivacyBudgetError(PrivacyError):
    default_code = "insufficient_privacy_budget"


class InvalidPrivacyParametersError(PrivacyError):
    default_code = "invalid_privacy_parameters"


# --- Aggregation ---
class AggregationError(DPSQLError):
    default_code = "aggregation_error"


# --- Backend ---
class BackendError(DPSQLError):
    default_code = "backend_error"


class UnsupportedBackendError(BackendError):
    default_code = "unsupported_backend"


class ExecutionBackendError(BackendError):
    default_code = "backend_execution_failed"


# --- Engine / Config / Auditing / Internal ---
class EngineError(DPSQLError):
    default_code = "engine_error"


class InternalError(DPSQLError):
    default_code = "internal_error"
