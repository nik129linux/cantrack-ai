"""Vision check: "is there actually a dog in this photo?", answered by Ollama.

A walker's check-in photo is a *claim*, and a claim is cheap: a screenshot, a
parked bike, the wrong dog. This module sends the photo to a vision model
hosted on Ollama and reports what it sees.

The model is a third party, so its answer is treated as untrusted input: the
JSON it returns is parsed defensively, and *any* failure degrades to
"no answer" (``PhotoCheck(None, None)``) instead of raising into a request.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_HOST = "https://ollama.com"
DEFAULT_MODEL = "gemma4:31b"
DEFAULT_TIMEOUT = 20.0

#: Notes longer than this are cut: the answer is a caption, not an essay.
NOTE_MAX_CHARS = 200

#: The keys the model is asked for, and the one it actually uses.
VISIBLE_KEY = "dog_visible"
VISIBLE_KEY_ALIAS = "is_dog_visible"

#: JSON schema handed to Ollama as a hint. The model is free to ignore it.
RESPONSE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        VISIBLE_KEY: {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": [VISIBLE_KEY, "note"],
}

PROMPT = (
    "Look at this photo. Is a dog clearly visible in it? "
    f"Answer with JSON only, using the keys \"{VISIBLE_KEY}\" (a boolean) and "
    "\"note\" (one short sentence describing the visible condition of the dog, "
    "such as its coat, cleanliness or mood). If no dog is visible, set "
    f"\"{VISIBLE_KEY}\" to false and say so in the note."
)

_FENCE_RE = re.compile(r"^```[^\n`]*\n?", re.MULTILINE)
_UNPARSED = object()


@dataclass(frozen=True)
class PhotoCheck:
    """What the vision model could tell us about a photo.

    Attributes:
        dog_visible: Whether a dog is visible, or ``None`` when unknown (no API
            key, a failed request, or an answer too broken to trust).
        note: One short sentence about the dog's visible condition, or ``None``.
    """

    dog_visible: bool | None
    note: str | None


class OllamaVision:
    """Asks an Ollama-hosted vision model whether a photo shows a dog."""

    def __init__(
        self,
        host: str,
        api_key: str | None,
        model: str,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Configure the client.

        Args:
            host: Base URL of the Ollama server, with or without a trailing "/".
            api_key: Bearer token for the Ollama cloud. Without one, the client
                is disabled and never makes a request.
            model: Name of the vision model to call.
            client: HTTP client to use. Created on first use when omitted.
            timeout: Per-request timeout in seconds.
        """
        self.host = host
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = client

    @classmethod
    def from_env(cls) -> OllamaVision:
        """Build a client from ``OLLAMA_HOST``, ``OLLAMA_API_KEY`` and
        ``OLLAMA_VISION_MODEL``, falling back to the Ollama cloud defaults.

        The environment is read here rather than at import time, so importing
        this module never depends on the deployment's configuration.

        Returns:
            A client configured from the current environment.
        """
        return cls(
            host=os.environ.get("OLLAMA_HOST") or DEFAULT_HOST,
            api_key=os.environ.get("OLLAMA_API_KEY"),
            model=os.environ.get("OLLAMA_VISION_MODEL") or DEFAULT_MODEL,
        )

    def check(self, image_bytes: bytes) -> PhotoCheck:
        """Ask the model whether ``image_bytes`` shows a dog.

        This never raises. An unreachable service, a rejected key, a non-JSON
        body or a nonsensical answer all mean the same thing to a check-in: we
        could not verify the photo.

        Args:
            image_bytes: The raw contents of an encoded image.

        Returns:
            The model's reading of the photo, or ``PhotoCheck(None, None)`` when
            it could not be obtained or trusted.
        """
        if not self.api_key:
            return PhotoCheck(None, None)

        try:
            payload = {
                "model": self.model,
                "stream": False,
                "format": RESPONSE_FORMAT,
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT,
                        "images": [base64.b64encode(image_bytes).decode("ascii")],
                    }
                ],
            }
            response = self._http().post(
                f"{self.host.removesuffix('/')}/api/chat",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            if response.status_code != httpx.codes.OK:
                return PhotoCheck(None, None)
            body = response.json()
        except Exception:
            return PhotoCheck(None, None)

        return _to_photo_check(body)

    def _http(self) -> httpx.Client:
        """Return the injected client, creating a default one on first use."""
        if self._client is None:
            self._client = httpx.Client()
        return self._client


def _to_photo_check(body: Any) -> PhotoCheck:
    """Read a ``PhotoCheck`` out of an Ollama chat response body."""
    if not isinstance(body, dict):
        return PhotoCheck(None, None)
    message = body.get("message")
    if not isinstance(message, dict):
        return PhotoCheck(None, None)
    content = message.get("content")
    if not isinstance(content, str):
        return PhotoCheck(None, None)

    answer = _extract_object(content)
    if answer is None:
        return PhotoCheck(None, None)

    return PhotoCheck(_visible(answer), _note(answer))


def _extract_object(content: str) -> dict[str, Any] | None:
    """Pull a JSON object out of whatever the model decided to answer with.

    The real model wraps its JSON in markdown fences and has been seen renaming
    keys, so a clean parse is only the first of three attempts.

    Args:
        content: The raw ``message.content`` of the response.

    Returns:
        The decoded object, or ``None`` if no attempt produced one.
    """
    text = content.strip()
    for candidate in (text, _FENCE_RE.sub("", text).strip(), _between_braces(text)):
        if candidate is _UNPARSED:
            continue
        try:
            decoded = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


def _between_braces(text: str) -> str:
    """Return the substring from the first "{" to the last "}", or the text
    itself when it has no braces."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        return text
    return text[start : end + 1]


def _visible(answer: dict[str, Any]) -> bool | None:
    """Read the visibility flag, accepting the alias the model really uses."""
    if VISIBLE_KEY in answer:
        value = answer[VISIBLE_KEY]
    elif VISIBLE_KEY_ALIAS in answer:
        value = answer[VISIBLE_KEY_ALIAS]
    else:
        return None
    # "true", 1 and "yes" are not a boolean answer; guessing would be a lie.
    return value if isinstance(value, bool) else None


def _note(answer: dict[str, Any]) -> str | None:
    """Read the note, truncated to a caption's worth of characters."""
    value = answer.get("note")
    if not isinstance(value, str):
        return None
    trimmed = value.strip()[:NOTE_MAX_CHARS]
    return trimmed or None
