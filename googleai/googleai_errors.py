"""Errors the Google AI nodes raise beyond what `NodeError` builds itself."""

from __future__ import annotations

from typing import Any, NoReturn

import httpx
import requests
from griptape_nodes.exe_types.core_types import NodeError, NodeErrorLink
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

API_KEY_NAME = "GOOGLE_API_KEY"
API_KEY_URL = "https://aistudio.google.com/apikey"
AI_STUDIO_API = "AI Studio API"
VERTEX_AI = "Vertex AI"
VERTEX_SECRET_NAMES = (
    "GOOGLE_WORKLOAD_IDENTITY_CONFIG_PATH",
    "GOOGLE_SERVICE_ACCOUNT_FILE_PATH",
    "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    "GOOGLE_CLOUD_PROJECT_ID",
)


def has_vertex_credentials() -> bool:
    return any(GriptapeNodes.SecretsManager().get_secret(name) for name in VERTEX_SECRET_NAMES)


def credentials_error(reason: str | None = None, extra_help: str = "") -> NodeError:
    """Google Cloud credentials that are missing (no `reason`) or that Google could not use."""
    if reason is None:
        message = (
            "Google Cloud credentials are not set. Add them in Settings → API Keys & Secrets, then run the node again."
        )
    else:
        message = f"Could not authenticate to Google Cloud: {reason.strip().rstrip('.')}. Check the credentials in Settings → API Keys & Secrets, then run the node again.{extra_help}"
    return NodeError(message, links=[NodeErrorLink.from_secret("GOOGLE", "Add Google Cloud credentials")])


def missing_credential_errors(api_provider: str | None = None) -> list[Exception] | None:
    """The missing-credential error for `validate_before_node_run`. `AI Studio API` checks the key, else Vertex."""
    if api_provider == AI_STUDIO_API:
        if GriptapeNodes.SecretsManager().get_secret(API_KEY_NAME):
            return None
        return [NodeError.from_missing_secret(API_KEY_NAME, key_url=API_KEY_URL)]
    return None if has_vertex_credentials() else [credentials_error()]


def parameter_error(message: str, parameter: str) -> NodeError:
    return NodeError(message, fields={"parameter": parameter})


def node_error_from(error: BaseException, *, api_provider: str | None = None) -> NodeError | None:
    """The NodeError for a Google failure, or None for any other exception.

    A Google failure carries an HTTP response or is a `requests` or `httpx` transport error.
    """
    if isinstance(error, NodeError):
        return error
    if getattr(error, "response", None) is None and not isinstance(
        error, requests.RequestException | httpx.RequestError
    ):
        return None
    studio = api_provider == AI_STUDIO_API
    return NodeError.from_request_exception(
        error,  # type: ignore[arg-type]
        provider="Google",
        secret_name=API_KEY_NAME if studio else None,
        key_refused=None if studio else False,
    )


def raise_node_error(error: BaseException, *, api_provider: str | None = None) -> NoReturn:
    """Re-raise a Google failure as a NodeError and any other exception unchanged. Call from an `except`."""
    converted = node_error_from(error, api_provider=api_provider)
    if converted is None or converted is error:
        raise error
    raise converted from error


def operation_error(error: Any) -> NodeError:
    """The NodeError for the `error` of a finished Veo operation: ``{"code": 13, "message": "..."}``."""
    body = error if isinstance(error, dict) else {"message": str(error)}
    message = body.get("message") or "Google reported an error while generating the video."
    code = body.get("code")
    return NodeError(message, fields={"error_code": code} if code is not None else None, response=body)


def content_filtered_error(count: int, reasons: list[Any]) -> NodeError:
    """Error for a Veo run where Google's content policy filtered every generated video."""
    noun = "video" if count == 1 else "videos"
    return NodeError(
        f"Google's content policy filtered {'the' if count == 1 else f'all {count}'} generated {noun}. "
        "Rephrase the prompt to avoid violent, sexual, or harmful content and run the node again.",
        response={"rai_media_filtered_reasons": [str(r) for r in reasons]} if reasons else None,
    )
