"""Shared error construction for the Google AI nodes.

Every node turns Google failures into a NodeError here, so the editor shows Google's own message
with the status code and response body attached, and a link to the setting that fixes a missing or
refused credential.

Google reports an error as ``{"error": {"code": 429, "message": "...", "status": "RESOURCE_EXHAUSTED"}}``.
The google-genai SDK raises it as ``google.genai.errors.APIError`` (``code``, ``message``, ``status``,
``details``). Its interactions API raises a separate ``APIError`` that carries ``status_code``,
``message`` and ``body``. Both are read by shape, because they are different classes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, NoReturn
from urllib.parse import quote

import requests
from griptape_nodes.exe_types.core_types import NodeError, NodeErrorLink
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

try:
    from google.api_core.exceptions import GoogleAPICallError
except ImportError:  # pragma: no cover
    GoogleAPICallError = None  # type: ignore[assignment,misc]

API_KEY_NAME = "GOOGLE_API_KEY"
API_KEY_PAGE_URL = "https://aistudio.google.com/apikey"
CREDENTIALS_DOCS_URL = "https://cloud.google.com/docs/authentication/provide-credentials-adc"

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

# Status codes Google uses when it does not accept the caller's credential.
AUTH_STATUS_CODES = {401, 403}


def secret_link(label: str, secret_filter: str) -> NodeErrorLink:
    """Link that opens the matching secrets in the editor's settings."""
    return NodeErrorLink(label=label, url=f"#settings-secrets?filter={quote(secret_filter)}")


def missing_api_key_error() -> NodeError:
    """Error for an AI Studio run that has no API key."""
    return NodeError(
        f"{API_KEY_NAME} is not set. Add it in Settings → API Keys & Secrets, then run the node again.",
        links=[
            secret_link("Add the API key", API_KEY_NAME),
            NodeErrorLink(label="Get a Google AI Studio API key", url=API_KEY_PAGE_URL),
        ],
    )


def missing_credentials_error() -> NodeError:
    """Error for a Vertex AI run with no Google Cloud credentials configured."""
    return NodeError(
        "Google Cloud credentials are not set. In Settings → API Keys & Secrets, "
        f"{CREDENTIALS_HELP} Then run the node again.",
        links=[
            secret_link("Add Google Cloud credentials", "GOOGLE"),
            NodeErrorLink(label="Google Cloud authentication", url=CREDENTIALS_DOCS_URL),
        ],
    )


def refused_credentials_error(reason: str, extra_help: str = "") -> NodeError:
    """Error for Google Cloud credentials that are configured but cannot be used."""
    return NodeError(
        f"Could not authenticate to Google Cloud: {reason.strip().rstrip('.')}. "
        f"Check the credentials in Settings → API Keys & Secrets, then run the node again.{extra_help}",
        links=[
            secret_link("Update Google Cloud credentials", "GOOGLE"),
            NodeErrorLink(label="Google Cloud authentication", url=CREDENTIALS_DOCS_URL),
        ],
    )


def has_vertex_credentials() -> bool:
    """True when at least one of the Google Cloud credential secrets is set."""
    secrets = GriptapeNodes.SecretsManager()
    return any(secrets.get_secret(name) for name in VERTEX_SECRET_NAMES)


def missing_credential_errors(api_provider: str | None = None) -> list[Exception] | None:
    """The missing-credential error for `validate_before_node_run`, or None when one is set.

    A missing key is reported on its own, before any input problem, since nothing can run without it.

    Args:
        api_provider: The node's `api_provider` value. `AI Studio API` checks `GOOGLE_API_KEY`.
            Anything else, including None for the Vertex-only nodes, checks the Google Cloud secrets.
    """
    if api_provider == AI_STUDIO_API:
        if GriptapeNodes.SecretsManager().get_secret(API_KEY_NAME):
            return None
        return [missing_api_key_error()]
    if has_vertex_credentials():
        return None
    return [missing_credentials_error()]


def parameter_error(message: str, parameter: str) -> NodeError:
    """Error about one parameter's value, naming it in `fields`."""
    return NodeError(message, fields={"parameter": parameter})


def error_from_response(
    status_code: int | None,
    body: Any,
    *,
    api_provider: str | None = None,
) -> NodeError:
    """Build a NodeError from a Google error status and body.

    Args:
        status_code: HTTP status code of the response.
        body: Parsed JSON body, or the raw text when the body is not JSON.
        api_provider: `AI Studio API` or `Vertex AI`, which decides what a refused credential links to.
            None means Vertex AI.
    """
    body = _plain(body)
    message, error_code = _read_error(body)
    if not message:
        message = f"Google returned an error (HTTP {status_code})." if status_code else "Google returned an error."

    fields: dict[str, str | int | float | bool] = {}
    if status_code is not None:
        fields["status_code"] = status_code
    if error_code:
        fields["error_code"] = error_code

    links: list[NodeErrorLink] = []
    if api_provider == AI_STUDIO_API:
        key_invalid = status_code == 401 or "api_key_invalid" in _text(body).lower()
        if key_invalid:
            message = f"Google rejected the API key: {message.rstrip('.')}. Update {API_KEY_NAME} in Settings → API Keys & Secrets, then run the node again."
        if key_invalid or status_code in AUTH_STATUS_CODES:
            links = [
                secret_link("Update the API key", API_KEY_NAME),
                NodeErrorLink(label="Get a Google AI Studio API key", url=API_KEY_PAGE_URL),
            ]
    elif status_code in AUTH_STATUS_CODES:
        links = [
            secret_link("Check Google Cloud credentials", "GOOGLE"),
            NodeErrorLink(label="Google Cloud authentication", url=CREDENTIALS_DOCS_URL),
        ]

    if status_code == 429:
        message = f"{message.rstrip('.')}. Wait a minute and run the node again."

    return NodeError(message, fields=fields, response=_response_dict(body), links=links)


def operation_error(error: Any) -> NodeError:
    """Build a NodeError from the `error` a finished Veo operation carries.

    The error is a Google RPC status: ``{"code": 13, "message": "...", "details": [...]}``. The code
    is the RPC code, not an HTTP status, so it goes in `error_code`.
    """
    error = _plain(error)
    message, _ = _read_error(error if isinstance(error, dict) else str(error))
    fields: dict[str, str | int | float | bool] = {}
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, (int, str)):
        fields["error_code"] = code
    if not message:
        message = "Google reported an error while generating the video."
    if code == 13:  # noqa: PLR2004
        message = (
            f"{message.rstrip('.')}. This is a temporary Google error. Run the node again in a few "
            "minutes, or try a different 'Location'."
        )
    return NodeError(message, fields=fields, response=_response_dict(error))


def content_filtered_error(count: int, reasons: list[Any]) -> NodeError:
    """Error for a Veo run where Google's content policy filtered every generated video."""
    noun = "video" if count == 1 else "videos"
    return NodeError(
        f"Google's content policy filtered {'the' if count == 1 else f'all {count}'} generated {noun}. "
        "Rephrase the prompt to avoid violent, sexual, or harmful content and run the node again.",
        response={"rai_media_filtered_reasons": [str(r) for r in reasons]} if reasons else None,
    )


def is_genai_api_error(error: BaseException) -> bool:
    """True for the google-genai SDK's APIError and the interactions API's APIError."""
    if isinstance(getattr(error, "code", None), int) and hasattr(error, "details"):
        return True
    return isinstance(getattr(error, "status_code", None), int) and hasattr(error, "body")


def error_from_genai_error(error: BaseException, *, api_provider: str | None = None) -> NodeError:
    """Build a NodeError from a google-genai SDK APIError."""
    status_code = getattr(error, "code", None)
    if not isinstance(status_code, int):
        status_code = getattr(error, "status_code", None)
    body = getattr(error, "details", None)
    if body is None:
        body = getattr(error, "body", None)
    if not _read_error(_plain(body))[0]:
        body = {"error": {"message": getattr(error, "message", None), "status": getattr(error, "status", None)}}
    return error_from_response(status_code, body, api_provider=api_provider)


def error_from_http_error(error: requests.exceptions.HTTPError, *, api_provider: str | None = None) -> NodeError:
    """Build a NodeError from a `requests` status error, as raised by the Lyria predict call."""
    response = error.response
    if response is None:
        return NodeError(f"Google returned an error: {error}")
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    return error_from_response(response.status_code, body, api_provider=api_provider)


def connection_error(error: requests.exceptions.RequestException) -> NodeError:
    """Build a NodeError for a request that never got an answer from Google."""
    if isinstance(error, requests.exceptions.Timeout):
        return NodeError("Google did not respond in time. Run the node again.")
    reason = str(error) or type(error).__name__
    return NodeError(f"Could not reach Google: {reason}. Check your connection and run the node again.")


def node_error_from(error: BaseException, *, api_provider: str | None = None) -> NodeError | None:
    """The NodeError for a Google failure, or None when `error` is not one.

    Covers the google-genai SDK's errors, `requests` status and connection errors, and google-api-core
    errors from Cloud Storage. A NodeError comes back unchanged.
    """
    if isinstance(error, NodeError):
        return error
    if isinstance(error, requests.exceptions.HTTPError):
        return error_from_http_error(error, api_provider=api_provider)
    if isinstance(error, requests.exceptions.RequestException):
        return connection_error(error)
    if is_genai_api_error(error):
        return error_from_genai_error(error, api_provider=api_provider)
    if GoogleAPICallError is not None and isinstance(error, GoogleAPICallError):
        code = error.code if isinstance(error.code, int) else None
        return error_from_response(code, {"error": {"message": error.message}}, api_provider=api_provider)
    return None


def raise_node_error(error: BaseException, *, api_provider: str | None = None) -> NoReturn:
    """Re-raise `error` as a NodeError when it is a Google failure, otherwise unchanged.

    Call it from an `except` block.
    """
    converted = node_error_from(error, api_provider=api_provider)
    if converted is None or converted is error:
        raise error
    raise converted from error


@contextmanager
def provider_errors(*, api_provider: str | None = None) -> Iterator[None]:
    """Turn Google failures raised inside the block into NodeError. Anything else propagates unchanged."""
    try:
        yield
    except Exception as e:
        raise_node_error(e, api_provider=api_provider)


def _plain(body: Any) -> Any:
    """Reduce an SDK body object (pydantic model) to plain JSON types."""
    if hasattr(body, "model_dump"):
        try:
            return body.model_dump(mode="json")
        except Exception:
            return str(body)
    return body


def _text(body: Any) -> str:
    """The body as one string, for substring checks."""
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body)
    except (TypeError, ValueError):
        return str(body)


def _read_error(body: Any) -> tuple[str | None, str | None]:
    """Return (message, error_code) read from a Google error body."""
    if isinstance(body, str):
        return (body.strip() or None), None
    if not isinstance(body, dict):
        return None, None
    inner = body.get("error")
    if not isinstance(inner, dict):
        inner = body
    message = inner.get("message")
    status = inner.get("status")
    if not isinstance(message, str) and isinstance(body.get("error"), str):
        message = body["error"]
    return (
        message if isinstance(message, str) and message else None,
        status if isinstance(status, str) and status else None,
    )


def _response_dict(body: Any) -> dict[str, Any] | None:
    """The provider body as the `response` dict a NodeError carries."""
    if isinstance(body, dict):
        return body
    if isinstance(body, str) and body.strip():
        return {"body": body[:2000]}
    return None
