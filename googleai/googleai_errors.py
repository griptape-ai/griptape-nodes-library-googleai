"""Errors the Google AI nodes raise beyond what `NodeError` builds itself.

A missing AI Studio key comes from `NodeError.from_missing_secret`. A failed Google call, from the
google-genai SDK, its interactions API, `requests` or Cloud Storage, goes through
`NodeError.from_request_exception`, because each of those errors carries the HTTP response. This module
covers the Vertex AI credential check, which spans four secrets, and Veo operation results.
"""

from __future__ import annotations

from typing import Any, NoReturn

import httpx
import requests
from griptape_nodes.exe_types.core_types import NodeError, NodeErrorLink
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

API_KEY_NAME = "GOOGLE_API_KEY"
API_KEY_URL = "https://aistudio.google.com/apikey"
PROVIDER = "Google"
AI_STUDIO_API = "AI Studio API"
VERTEX_AI = "Vertex AI"

# The secrets GoogleAuthHelper reads, in the order it tries them.
VERTEX_SECRET_NAMES = (
    "GOOGLE_WORKLOAD_IDENTITY_CONFIG_PATH",
    "GOOGLE_SERVICE_ACCOUNT_FILE_PATH",
    "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    "GOOGLE_CLOUD_PROJECT_ID",
)
CREDENTIALS_HELP = (
    "set GOOGLE_WORKLOAD_IDENTITY_CONFIG_PATH (recommended, path to a workload identity config), "
    "or GOOGLE_SERVICE_ACCOUNT_FILE_PATH (path to a service account JSON), "
    "or GOOGLE_APPLICATION_CREDENTIALS_JSON, "
    "or GOOGLE_CLOUD_PROJECT_ID on a machine with application default credentials."
)


def missing_credential_errors(api_provider: str | None = None) -> list[Exception] | None:
    """The missing-credential error for `validate_before_node_run`, or None when one is set.

    `AI Studio API` checks `GOOGLE_API_KEY`. Anything else, including None for the Vertex-only nodes,
    checks the Google Cloud secrets.
    """
    secrets = GriptapeNodes.SecretsManager()
    if api_provider == AI_STUDIO_API:
        if secrets.get_secret(API_KEY_NAME):
            return None
        return [NodeError.from_missing_secret(API_KEY_NAME, key_url=API_KEY_URL)]
    if has_vertex_credentials():
        return None
    return [missing_credentials_error()]


def has_vertex_credentials() -> bool:
    """True when at least one of the Google Cloud credential secrets is set."""
    secrets = GriptapeNodes.SecretsManager()
    return any(secrets.get_secret(name) for name in VERTEX_SECRET_NAMES)


def missing_credentials_error() -> NodeError:
    """Error for a Vertex AI run with no Google Cloud credentials configured."""
    return NodeError(
        f"Google Cloud credentials are not set. In Settings → API Keys & Secrets, {CREDENTIALS_HELP} "
        "Then run the node again.",
        links=[NodeErrorLink.from_secret("GOOGLE", "Add Google Cloud credentials")],
    )


def refused_credentials_error(reason: str, extra_help: str = "") -> NodeError:
    """Error for Google Cloud credentials that are configured but cannot be used."""
    return NodeError(
        f"Could not authenticate to Google Cloud: {reason.strip().rstrip('.')}. "
        f"Check the credentials in Settings → API Keys & Secrets, then run the node again.{extra_help}",
        links=[NodeErrorLink.from_secret("GOOGLE", "Update Google Cloud credentials")],
    )


def parameter_error(message: str, parameter: str) -> NodeError:
    """Error about one parameter's value, naming it in `fields`."""
    return NodeError(message, fields={"parameter": parameter})


def node_error_from(error: BaseException, *, api_provider: str | None = None) -> NodeError | None:
    """The NodeError for a Google failure, or None when `error` is not one.

    A failure is any error that carries an HTTP response, or a `requests` or `httpx` transport error.
    On AI Studio a refused key links to `GOOGLE_API_KEY`. On Vertex AI there is no single key to link.
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
        provider=PROVIDER,
        secret_name=API_KEY_NAME if studio else None,
        key_refused=None if studio else False,
    )


def raise_node_error(error: BaseException, *, api_provider: str | None = None) -> NoReturn:
    """Re-raise `error` as a NodeError when it is a Google failure, otherwise unchanged.

    Call it from an `except` block.
    """
    converted = node_error_from(error, api_provider=api_provider)
    if converted is None or converted is error:
        raise error
    raise converted from error


def operation_error(error: Any) -> NodeError:
    """The NodeError for the `error` a finished Veo operation carries: ``{"code": 13, "message": "..."}``."""
    body = error if isinstance(error, dict) else {"message": str(error)}
    message = body.get("message") or "Google reported an error while generating the video."
    code = body.get("code")
    if code == 13:  # noqa: PLR2004
        message = (
            f"{message.rstrip('.')}. This is a temporary Google error. Run the node again in a few "
            "minutes, or try a different 'Location'."
        )
    return NodeError(message, fields={"error_code": code} if code is not None else None, response=body)


def content_filtered_error(count: int, reasons: list[Any]) -> NodeError:
    """Error for a Veo run where Google's content policy filtered every generated video."""
    noun = "video" if count == 1 else "videos"
    return NodeError(
        f"Google's content policy filtered {'the' if count == 1 else f'all {count}'} generated {noun}. "
        "Rephrase the prompt to avoid violent, sexual, or harmful content and run the node again.",
        response={"rai_media_filtered_reasons": [str(r) for r in reasons]} if reasons else None,
    )
