"""NodeError behaviour for the Google AI nodes, reproduced through the real SDK and HTTP paths."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import googleai_errors
import googleai_utils
import httpx
import pytest
import requests
from base_analyze_media import BaseAnalyzeMedia
from describe_media import DescribeMedia
from gemini_3_1_flash_image_generator import NanaBanana2ImageGenerator
from gemini_3_pro_image_generator import NanoBananaProImageGenerator
from gemini_image_generator import GeminiImageGenerator
from gemini_omni_flash_generator import GeminiOmniFlashVideoGenerator
from google import genai
from google.genai import types
from googleai_errors import AI_STUDIO_API, VERTEX_AI
from griptape_nodes.exe_types.core_types import NodeError
from identify_timecodes import IdentifyTimecodes
from imagen_image_generator import VertexAIImageGenerator
from lyria_audio_generator import LyriaAudioGenerator
from veo_image_to_video_generator import VeoImageToVideoGenerator
from veo_text_to_video_with_ref import VeoTextToVideoWithRef
from veo_video_generator import VeoVideoGenerator

REAL_CLIENT = genai.Client
LIBRARY_DIR = Path(googleai_errors.__file__).parent

VERTEX_ONLY_NODES = [
    VeoVideoGenerator,
    VeoImageToVideoGenerator,
    VeoTextToVideoWithRef,
    VertexAIImageGenerator,
    GeminiImageGenerator,
    LyriaAudioGenerator,
    DescribeMedia,
    IdentifyTimecodes,
]
TWO_SURFACE_NODES = [NanoBananaProImageGenerator, NanaBanana2ImageGenerator, GeminiOmniFlashVideoGenerator]


class FakeSecrets:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def get_secret(self, name: str) -> str | None:
        return self.values.get(name)


@pytest.fixture
def secrets(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Secrets the nodes see, so a developer's own settings never leak into a test."""
    values: dict[str, str] = {}
    fake = SimpleNamespace(SecretsManager=lambda: FakeSecrets(values))
    monkeypatch.setattr(googleai_errors, "GriptapeNodes", fake)
    monkeypatch.setattr(googleai_utils, "GriptapeNodes", fake)
    return values


def genai_client(status: int, body: Any) -> genai.Client:
    """A real google-genai client whose HTTP layer answers with `status` and `body`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return REAL_CLIENT(api_key="test", http_options=types.HttpOptions(httpx_client=http_client))


def link_urls(error: NodeError) -> list[str]:
    return [link.url for link in error.links]


class TestMissingCredentials:
    @pytest.mark.parametrize("node_class", VERTEX_ONLY_NODES)
    def test_vertex_node_reports_missing_credentials_alone(self, secrets, node_class) -> None:
        node = node_class(name="node")
        errors = node.validate_before_node_run()
        assert errors is not None
        assert len(errors) == 1
        assert isinstance(errors[0], NodeError)
        assert "Google Cloud credentials are not set" in str(errors[0])
        assert "#settings-secrets?filter=GOOGLE" in link_urls(errors[0])

    @pytest.mark.parametrize("node_class", TWO_SURFACE_NODES)
    def test_ai_studio_node_reports_missing_key_alone(self, secrets, node_class) -> None:
        node = node_class(name="node")
        node.set_parameter_value("api_provider", AI_STUDIO_API)
        secrets["GOOGLE_CLOUD_PROJECT_ID"] = "a-project"
        errors = node.validate_before_node_run()
        assert errors is not None
        assert len(errors) == 1
        assert str(errors[0]).startswith("GOOGLE_API_KEY is not set")
        assert "#settings-secrets?filter=GOOGLE_API_KEY" in link_urls(errors[0])
        assert "https://aistudio.google.com/apikey" in link_urls(errors[0])

    @pytest.mark.parametrize("node_class", TWO_SURFACE_NODES)
    def test_vertex_surface_ignores_the_api_key(self, secrets, node_class) -> None:
        node = node_class(name="node")
        node.set_parameter_value("api_provider", VERTEX_AI)
        secrets["GOOGLE_API_KEY"] = "key"
        errors = node.validate_before_node_run()
        assert errors is not None
        assert "Google Cloud credentials are not set" in str(errors[0])

    @pytest.mark.parametrize("node_class", [*VERTEX_ONLY_NODES, *TWO_SURFACE_NODES])
    def test_input_problems_show_once_credentials_exist(self, secrets, node_class) -> None:
        secrets["GOOGLE_CLOUD_PROJECT_ID"] = "a-project"
        secrets["GOOGLE_API_KEY"] = "key"
        node = node_class(name="node")
        errors = node.validate_before_node_run()
        assert errors
        for error in errors:
            assert isinstance(error, NodeError)
            assert "node:" not in str(error)

    def test_credentials_or_raise_without_any_secret(self, secrets) -> None:
        with pytest.raises(NodeError, match="Google Cloud credentials are not set") as raised:
            googleai_utils.credentials_or_raise()
        assert "#settings-secrets?filter=GOOGLE" in link_urls(raised.value)

    def test_credentials_or_raise_with_unusable_secret(self, secrets) -> None:
        secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = "not json"
        cleared: list[bool] = []
        with pytest.raises(NodeError, match="Could not authenticate to Google Cloud") as raised:
            googleai_utils.credentials_or_raise(on_failure=lambda: cleared.append(True), extra_help=" Extra.")
        assert str(raised.value).endswith("Extra.")
        assert cleared == [True]
        assert "#settings-secrets?filter=GOOGLE" in link_urls(raised.value)

    def test_api_key_or_raise(self, secrets) -> None:
        with pytest.raises(NodeError, match="GOOGLE_API_KEY is not set"):
            googleai_utils.api_key_or_raise()
        secrets["GOOGLE_API_KEY"] = "key"
        assert googleai_utils.api_key_or_raise() == "key"


class TestGenaiErrors:
    """Errors raised by the google-genai SDK itself, through a mocked HTTP layer."""

    def test_invalid_api_key_on_ai_studio(self) -> None:
        body = {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
            }
        }
        with pytest.raises(Exception) as raised:  # noqa: PT011
            genai_client(400, body).models.generate_content(model="m", contents="hi")
        error = googleai_errors.node_error_from(raised.value, api_provider=AI_STUDIO_API)
        assert error is not None
        assert str(error) == "API key not valid. Please pass a valid API key."
        assert error.fields["status_code"] == 400
        assert error.response == body
        assert link_urls(error) == ["#settings-secrets?filter=GOOGLE_API_KEY"]

    def test_permission_denied_on_vertex_has_no_key_link(self) -> None:
        body = {"error": {"code": 403, "message": "Permission denied on resource.", "status": "PERMISSION_DENIED"}}
        with pytest.raises(Exception) as raised:  # noqa: PT011
            genai_client(403, body).models.generate_content(model="m", contents="hi")
        error = googleai_errors.node_error_from(raised.value, api_provider=VERTEX_AI)
        assert error is not None
        assert str(error) == "Permission denied on resource."
        assert error.links == []

    def test_interactions_api_error(self) -> None:
        body = {"error": {"code": 404, "message": "Model not found.", "status": "NOT_FOUND"}}
        with pytest.raises(Exception) as raised:  # noqa: PT011
            genai_client(404, body).interactions.create(model="lyria-3-pro-preview", input="hi")
        error = googleai_errors.node_error_from(raised.value)
        assert error is not None
        assert str(error) == "Model not found."
        assert error.fields["status_code"] == 404
        assert error.response == body

    def test_other_exceptions_are_not_converted(self) -> None:
        assert googleai_errors.node_error_from(ValueError("x")) is None
        with pytest.raises(ValueError, match="x"):
            try:
                raise ValueError("x")
            except ValueError as e:
                googleai_errors.raise_node_error(e)


class TestImageNodes:
    def test_generation_failure_reaches_the_editor_as_a_node_error(self, secrets, monkeypatch) -> None:
        secrets["GOOGLE_API_KEY"] = "key"
        body = {"error": {"code": 429, "message": "Quota exceeded.", "status": "RESOURCE_EXHAUSTED"}}
        node = NanaBanana2ImageGenerator(name="banana")
        node.set_parameter_value("api_provider", AI_STUDIO_API)
        node.set_parameter_value("prompt", "a cat")
        node._output_file.build_file = lambda: pytest.fail("no file should be written")
        monkeypatch.setattr(genai, "Client", lambda **_kwargs: genai_client(429, body))
        with pytest.raises(NodeError, match="Quota exceeded") as raised:
            node._process()
        assert "banana" not in str(raised.value)
        assert raised.value.fields["status_code"] == 429
        assert raised.value.response == body

    def test_missing_key_at_run_time_links_to_settings(self, secrets) -> None:
        node = NanoBananaProImageGenerator(name="banana")
        node.set_parameter_value("api_provider", AI_STUDIO_API)
        with pytest.raises(NodeError, match="GOOGLE_API_KEY is not set") as raised:
            node._process()
        assert "#settings-secrets?filter=GOOGLE_API_KEY" in link_urls(raised.value)


class TestLyria:
    def lyria(self, secrets) -> LyriaAudioGenerator:
        return LyriaAudioGenerator(name="lyria")

    @staticmethod
    def response(status: int, body: Any) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(body).encode()
        return response

    def run_predict(self, node: LyriaAudioGenerator, monkeypatch: pytest.MonkeyPatch, status: int, body: Any) -> None:
        monkeypatch.setattr(requests, "post", lambda *_a, **_k: self.response(status, body))
        credentials = SimpleNamespace(valid=True, token="token")
        node._generate_audio("proj", credentials, "lyria-002", "calm", "", 1, "us-central1")

    def test_http_error_carries_status_and_response(self, secrets, monkeypatch) -> None:
        body = {
            "error": {
                "code": 403,
                "message": "Vertex AI API has not been used in project 1.",
                "status": "PERMISSION_DENIED",
            }
        }
        with pytest.raises(NodeError, match="Vertex AI API has not been used") as raised:
            self.run_predict(self.lyria(secrets), monkeypatch, 403, body)
        assert raised.value.fields["status_code"] == 403
        assert raised.value.response == body

    def test_not_found_adds_the_allowlist_hint(self, secrets, monkeypatch) -> None:
        body = {"error": {"code": 404, "message": "Publisher model not found.", "status": "NOT_FOUND"}}
        with pytest.raises(NodeError) as raised:
            self.run_predict(self.lyria(secrets), monkeypatch, 404, body)
        assert str(raised.value).startswith("Publisher model not found. 'lyria-002' resolves on Vertex AI")

    def test_blocked_prompt_adds_the_copyright_hint(self, secrets, monkeypatch) -> None:
        body = {
            "error": {"code": 400, "message": "Request blocked by recitation checks.", "status": "INVALID_ARGUMENT"}
        }
        with pytest.raises(NodeError, match="blocked for possible copyright similarity"):
            self.run_predict(self.lyria(secrets), monkeypatch, 400, body)

    def test_empty_prediction_is_a_node_error(self, secrets, monkeypatch) -> None:
        with pytest.raises(NodeError, match="returned no audio") as raised:
            self.run_predict(self.lyria(secrets), monkeypatch, 200, {"metadata": {}})
        assert raised.value.response == {"metadata": {}}

    def test_connection_failure(self, secrets, monkeypatch) -> None:
        def fail(*_a: Any, **_k: Any) -> None:
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(requests, "post", fail)
        node = self.lyria(secrets)
        with pytest.raises(NodeError, match="Could not reach Google"):
            node._generate_audio(
                "proj", SimpleNamespace(valid=True, token="t"), "lyria-002", "calm", "", 1, "us-central1"
            )


class TestVeo:
    @staticmethod
    def operation(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**{"done": True, "error": None, "response": None, **kwargs})

    @pytest.mark.parametrize("node_class", [VeoVideoGenerator, VeoImageToVideoGenerator, VeoTextToVideoWithRef])
    def test_operation_error(self, node_class) -> None:
        node = node_class(name="veo")
        operation = self.operation(error={"code": 3, "message": "The prompt could not be processed."})
        with pytest.raises(NodeError, match="The prompt could not be processed.") as raised:
            node._poll_and_process_video_result(None, operation, "proj", None)
        assert raised.value.fields == {"error_code": 3}
        assert raised.value.response == {"code": 3, "message": "The prompt could not be processed."}

    @pytest.mark.parametrize("node_class", [VeoVideoGenerator, VeoImageToVideoGenerator, VeoTextToVideoWithRef])
    def test_internal_error_suggests_trying_again(self, node_class) -> None:
        node = node_class(name="veo")
        operation = self.operation(error={"code": 13, "message": "Internal error."})
        with pytest.raises(NodeError, match="temporary Google error"):
            node._poll_and_process_video_result(None, operation, "proj", None)

    @pytest.mark.parametrize("node_class", [VeoVideoGenerator, VeoImageToVideoGenerator, VeoTextToVideoWithRef])
    def test_content_filter_moves_reasons_into_the_response(self, node_class) -> None:
        node = node_class(name="veo")
        operation = SimpleNamespace(
            done=True,
            error=None,
            response=SimpleNamespace(
                generated_videos=[], rai_media_filtered_count=2, rai_media_filtered_reasons=["Sensitive content"]
            ),
        )
        with pytest.raises(NodeError, match="filtered all 2 generated videos") as raised:
            node._poll_and_process_video_result(None, operation, "proj", None)
        assert raised.value.response == {"rai_media_filtered_reasons": ["Sensitive content"]}

    @pytest.mark.parametrize("node_class", [VeoVideoGenerator, VeoImageToVideoGenerator, VeoTextToVideoWithRef])
    def test_no_response(self, node_class) -> None:
        node = node_class(name="veo")
        with pytest.raises(NodeError, match="returned no response"):
            node._poll_and_process_video_result(None, self.operation(), "proj", None)

    def test_image_to_video_without_an_image_raises(self, secrets) -> None:
        node = VeoImageToVideoGenerator(name="veo")
        with pytest.raises(NodeError, match="'Image' has no input") as raised:
            next(node.process())
        assert raised.value.fields == {"parameter": "image"}

    def test_image_to_video_with_an_unreadable_image_dict_raises(self, secrets) -> None:
        node = VeoImageToVideoGenerator(name="veo")
        node.set_parameter_value("image", {"nothing": "useful"})
        with pytest.raises(NodeError, match="Could not read the image connected to 'Image'"):
            next(node.process())


class TestAnalyze:
    def test_no_candidates(self) -> None:
        with pytest.raises(NodeError, match="returned no response"):
            BaseAnalyzeMedia._read_response_text(SimpleNamespace(text=None, candidates=[]))

    def test_max_tokens(self) -> None:
        response = SimpleNamespace(text=None, candidates=[SimpleNamespace(finish_reason="FinishReason.MAX_TOKENS")])
        with pytest.raises(NodeError, match="Raise 'Max Tokens'") as raised:
            BaseAnalyzeMedia._read_response_text(response)
        assert raised.value.fields == {"parameter": "max_tokens"}

    def test_no_text(self) -> None:
        response = SimpleNamespace(text=None, candidates=[SimpleNamespace(finish_reason="SAFETY")])
        with pytest.raises(NodeError, match="returned no text") as raised:
            BaseAnalyzeMedia._read_response_text(response)
        assert raised.value.fields == {"finish_reason": "SAFETY"}

    def test_invalid_timecode_json_keeps_the_text_in_the_response(self) -> None:
        node = IdentifyTimecodes(name="timecodes")
        response = SimpleNamespace(text="not json at all", candidates=[])
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_k: response))
        with pytest.raises(NodeError, match="did not return valid JSON") as raised:
            node._analyze_multiple_media_with_gemini(client, [], "p", "m", 0.1, 100)
        assert raised.value.response == {"text": "not json at all"}


class TestImageValidation:
    def test_unsupported_mime_type(self) -> None:
        with pytest.raises(NodeError, match="unsupported MIME type image/gif"):
            googleai_utils.validate_and_maybe_shrink_image(b"x", "image/gif", "cat.gif", {"image/png"}, 100)

    def test_too_large_without_auto_resize(self) -> None:
        with pytest.raises(NodeError, match="turn on 'Auto Image Resize'"):
            googleai_utils.validate_and_maybe_shrink_image(
                b"x" * 10, "image/png", "cat.png", {"image/png"}, 5, auto_image_resize=False
            )


NODE_FILES = sorted(
    {
        "veo_video_generator.py",
        "veo_image_to_video_generator.py",
        "veo_text_to_video_with_ref.py",
        "imagen_image_generator.py",
        "gemini_image_generator.py",
        "gemini_3_pro_image_generator.py",
        "gemini_3_1_flash_image_generator.py",
        "gemini_omni_flash_generator.py",
        "lyria_audio_generator.py",
        "describe_media.py",
        "identify_timecodes.py",
    }
)


@pytest.mark.parametrize("file_name", NODE_FILES)
def test_node_file_loads_the_way_the_engine_loads_it(file_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine loads each node file standalone, with the manifest's folder on sys.path.

    A package-name or relative import of googleai_errors would pass in the dev environment, where the
    project is installed, and fail here.
    """
    monkeypatch.setitem(sys.modules, "googleai", None)
    monkeypatch.syspath_prepend(str(LIBRARY_DIR))
    spec = importlib.util.spec_from_file_location(f"_engine_load_{file_name[:-3]}", LIBRARY_DIR / file_name)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
