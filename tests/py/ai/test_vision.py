"""Ollama vision check: 'is there a dog in this photo?'. Model output is untrusted."""

import base64
import json

import httpx
import pytest

from cantrack_api.ai.vision import OllamaVision, PhotoCheck

IMAGE = b"\xff\xd8fake-jpeg-bytes"
B64 = base64.b64encode(IMAGE).decode()


def make(handler, **kw) -> OllamaVision:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaVision(
        host=kw.pop("host", "https://ollama.test"),
        api_key=kw.pop("api_key", "key-123"),
        model=kw.pop("model", "gemma4:31b"),
        client=client,
        **kw,
    )


def reply(content, status=200):
    def handler(request):
        return httpx.Response(status, json={"message": {"content": content}})

    return handler


class TestParsing:
    def test_clean_json(self):
        v = make(reply('{"dog_visible": true, "note": "Calm and clean."}'))
        assert v.check(IMAGE) == PhotoCheck(dog_visible=True, note="Calm and clean.")

    def test_dog_not_visible(self):
        v = make(reply('{"dog_visible": false, "note": "Only a tree."}'))
        assert v.check(IMAGE) == PhotoCheck(dog_visible=False, note="Only a tree.")

    def test_markdown_fenced_json_is_accepted(self):
        content = '```json\n{"dog_visible": true, "note": "Muddy paws."}\n```'
        assert make(reply(content)).check(IMAGE) == PhotoCheck(True, "Muddy paws.")

    def test_is_dog_visible_alias_is_accepted(self):
        # gemma4 really answered with this key name despite the requested schema
        content = '{"is_dog_visible": true, "note": "Wearing a yellow sweater."}'
        assert make(reply(content)).check(IMAGE) == PhotoCheck(True, "Wearing a yellow sweater.")

    def test_json_surrounded_by_prose(self):
        content = 'Sure! Here you go: {"dog_visible": true, "note": "ok"} Hope it helps.'
        assert make(reply(content)).check(IMAGE) == PhotoCheck(True, "ok")

    @pytest.mark.parametrize(
        "content",
        ["I can see a dog.", "", "{not json}", "[]", '"just a string"', "null"],
    )
    def test_unparseable_content_is_unavailable(self, content):
        assert make(reply(content)).check(IMAGE) == PhotoCheck(None, None)

    @pytest.mark.parametrize("value", ['"true"', "1", "null", '"yes"'])
    def test_non_boolean_visibility_is_unknown_not_guessed(self, value):
        content = '{"dog_visible": %s, "note": "x"}' % value
        assert make(reply(content)).check(IMAGE) == PhotoCheck(None, "x")

    def test_missing_note_is_none(self):
        assert make(reply('{"dog_visible": true}')).check(IMAGE) == PhotoCheck(True, None)

    @pytest.mark.parametrize("note", ["5", "null", '["a"]', '{"a": 1}'])
    def test_non_string_note_is_dropped(self, note):
        content = '{"dog_visible": true, "note": %s}' % note
        assert make(reply(content)).check(IMAGE) == PhotoCheck(True, None)

    def test_long_note_is_truncated_to_200_chars(self):
        content = json.dumps({"dog_visible": True, "note": "x" * 500})
        assert len(make(reply(content)).check(IMAGE).note) == 200

    def test_note_is_stripped(self):
        content = json.dumps({"dog_visible": True, "note": "  happy  \n"})
        assert make(reply(content)).check(IMAGE).note == "happy"

    def test_response_without_message_is_unavailable(self):
        v = make(lambda r: httpx.Response(200, json={"unexpected": 1}))
        assert v.check(IMAGE) == PhotoCheck(None, None)

    def test_non_json_http_body_is_unavailable(self):
        v = make(lambda r: httpx.Response(200, content=b"<html>oops</html>"))
        assert v.check(IMAGE) == PhotoCheck(None, None)


class TestFailuresNeverRaise:
    @pytest.mark.parametrize("status", [400, 401, 402, 429, 500, 503])
    def test_http_errors_are_unavailable(self, status):
        assert make(reply("{}", status=status)).check(IMAGE) == PhotoCheck(None, None)

    @pytest.mark.parametrize(
        "exc", [httpx.ConnectTimeout("t"), httpx.ReadTimeout("t"), httpx.ConnectError("c")]
    )
    def test_network_errors_are_unavailable(self, exc):
        def handler(request):
            raise exc

        assert make(handler).check(IMAGE) == PhotoCheck(None, None)

    def test_no_api_key_means_no_request_at_all(self):
        def handler(request):
            raise AssertionError("must not call the network without a key")

        assert make(handler, api_key=None).check(IMAGE) == PhotoCheck(None, None)
        assert make(handler, api_key="").check(IMAGE) == PhotoCheck(None, None)


class TestRequestShape:
    def test_sends_the_image_model_and_auth_to_api_chat(self):
        seen = {}

        def handler(request: httpx.Request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"message": {"content": '{"dog_visible": true}'}})

        make(handler).check(IMAGE)
        assert seen["url"] == "https://ollama.test/api/chat"
        assert seen["auth"] == "Bearer key-123"
        body = seen["body"]
        assert body["model"] == "gemma4:31b"
        assert body["stream"] is False
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["images"] == [B64]
        assert "dog" in body["messages"][0]["content"].lower()
        assert isinstance(body["format"], dict)
        assert "dog_visible" in body["format"]["properties"]

    def test_trailing_slash_in_host_is_tolerated(self):
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, json={"message": {"content": "{}"}})

        make(handler, host="https://ollama.test/").check(IMAGE)
        assert seen == ["https://ollama.test/api/chat"]


class TestFromEnv:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "k")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_VISION_MODEL", raising=False)
        v = OllamaVision.from_env()
        assert (v.host, v.api_key, v.model) == ("https://ollama.com", "k", "gemma4:31b")

    def test_overrides(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "k2")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_VISION_MODEL", "llava")
        v = OllamaVision.from_env()
        assert (v.host, v.api_key, v.model) == ("http://localhost:11434", "k2", "llava")

    def test_missing_key_gives_a_disabled_client(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        assert OllamaVision.from_env().check(IMAGE) == PhotoCheck(None, None)
