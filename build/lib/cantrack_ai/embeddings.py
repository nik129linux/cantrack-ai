"""Image embeddings: CLIP preprocessing, cosine similarity and the ONNX embedder.

A walker's or owner's photo has to become the *same* kind of vector the browser
already stored in the database, otherwise every comparison is noise. That means
reproducing the CLIP preprocessor exactly (EXIF orientation, shortest-edge
resize, centre crop, channel normalisation) and running the very same quantised
vision model the browser used, unnormalised, so a browser-enrolled dog and a
server-enrolled one live in the same space.
"""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np
from PIL import Image, ImageOps

DEFAULT_MODEL_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
    "/onnx/vision_model_quantized.onnx"
)

#: Edge length the CLIP vision tower expects, in pixels.
IMAGE_SIZE = 224
#: CLIP's per-channel normalisation constants, in RGB order.
CHANNEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
CHANNEL_STD = (0.26862954, 0.26130258, 0.27577711)
#: Refuse anything above this many pixels: 224x224 is all we keep, so decoding
#: a 100-megapixel panorama would only cost memory.
MAX_PIXELS = 50_000_000

_DOWNLOAD_TIMEOUT = 300.0


class ImageError(ValueError):
    """The bytes handed to the AI layer are not a readable image."""


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Turn encoded image bytes into a CLIP input tensor.

    The steps, and their order, are the ones the browser-side CLIP
    preprocessor performs: honour the EXIF orientation flag, convert to RGB,
    scale the *shortest* edge to 224 keeping the aspect ratio, take the centre
    224x224 crop and normalise each channel.

    Args:
        image_bytes: The raw contents of an encoded image (JPEG, PNG, ...).

    Returns:
        A ``float32`` array of shape ``(1, 3, 224, 224)`` ready to be fed to the
        CLIP vision model.

    Raises:
        ImageError: If the bytes cannot be decoded, are empty/truncated, or
            describe an image with more than ``MAX_PIXELS`` pixels.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            # exif_transpose reads the pixels itself, and phones store sideways
            # pixels plus an orientation flag, so this has to come first.
            image = ImageOps.exif_transpose(opened)
            if image.width * image.height > MAX_PIXELS:
                raise ImageError(
                    f"image is too large: {image.width}x{image.height} pixels"
                )
            image.load()
            rgb = image.convert("RGB")
    except ImageError:
        raise
    except Exception as exc:  # Pillow raises a zoo of errors on bad input.
        raise ImageError(f"unreadable image: {exc}") from exc

    resized = _resize_shortest_edge(rgb, IMAGE_SIZE)
    return _to_tensor(_center_crop(resized, IMAGE_SIZE))


def _resize_shortest_edge(image: Image.Image, size: int) -> Image.Image:
    """Scale the shortest edge to ``size``, keeping the aspect ratio. O(1) in n."""
    width, height = image.size
    shortest = min(width, height)
    new_width = round(width * size / shortest)
    new_height = round(height * size / shortest)
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    """Take the centred ``size`` x ``size`` crop. O(1) in n."""
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def _to_tensor(image: Image.Image) -> np.ndarray:
    """Normalise a cropped RGB image into a ``(1, 3, H, W)`` float32 tensor."""
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(CHANNEL_MEAN, dtype=np.float32)
    std = np.array(CHANNEL_STD, dtype=np.float32)
    pixels = (pixels - mean) / std
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two embeddings, as plain Python floats.

    Degenerate input is never an error here: comparing a dog that has no
    embedding yet should simply score zero, not blow up a check-in.

    Args:
        a: The first embedding.
        b: The second embedding.

    Returns:
        The similarity in ``[-1.0, 1.0]``, or ``0.0`` when either vector is
        empty, the two lengths differ, or either has zero magnitude.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    magnitude_a = 0.0
    magnitude_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        magnitude_a += x * x
        magnitude_b += y * y

    if magnitude_a <= 0.0 or magnitude_b <= 0.0:
        return 0.0

    return dot / ((magnitude_a**0.5) * (magnitude_b**0.5))


def parse_embedding(value: Any) -> list[float] | None:
    """Normalise a stored embedding into a list of floats.

    Supabase's REST layer hands a ``pgvector`` column back as its text form
    (``"[0.1,0.2]"``), while an embedding built in this process is already a
    list. Both are accepted; anything else is reported as missing so a caller can
    treat the dog as not enrolled.

    Args:
        value: The raw column value, a JSON-ish string, or a sequence.

    Returns:
        The embedding as a list of floats, or ``None`` when ``value`` is not a
        list of numbers.
    """
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except ValueError:
            return None

    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        return None

    parsed: list[float] = []
    for item in value:
        # bool is a subclass of int, but True is not an embedding coordinate.
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        parsed.append(float(item))
    return parsed


class ClipEmbedder:
    """Turns photo bytes into a 512-float CLIP image embedding.

    The quantised CLIP vision model is downloaded once, on first use, and kept
    on disk. Embeddings are returned *unnormalised*, exactly as the browser
    produced them, so vectors enrolled on either side stay comparable.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        model_url: str = DEFAULT_MODEL_URL,
    ) -> None:
        """Point the embedder at a model on disk, choosing a default location.

        Args:
            model_path: Where the ``.onnx`` file lives. Defaults to
                ``$CLIP_MODEL_PATH`` when that is set, otherwise
                ``~/.cache/cantrack/clip_vision_q.onnx``.
            model_url: Where to download the model from if it is missing.
        """
        self.model_path: Path = (
            Path(model_path)
            if model_path is not None
            else Path(os.environ.get("CLIP_MODEL_PATH") or Path.home() / ".cache" / "cantrack" / "clip_vision_q.onnx")
        )
        self.model_url = model_url
        self._session: Any | None = None
        self._lock = threading.Lock()

    def ensure_model(self, client: httpx.Client | None = None) -> Path:
        """Make sure the model file exists, downloading it once if it does not.

        The download is streamed to a temporary file next to the target and then
        moved into place, so an interrupted run can never leave a half-written
        model behind for the next process to trust.

        Args:
            client: HTTP client to download with. A redirect-following client
                with a long timeout is created when omitted; a client passed in
                is left open for its owner to close.

        Returns:
            The path of the model file.

        Raises:
            RuntimeError: If the download fails or does not answer 200.
        """
        if self.model_path.exists():
            return self.model_path

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.model_path.with_name(f"{self.model_path.name}.{os.getpid()}.part")
        owned = client is None
        http = client or httpx.Client(follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT)

        try:
            with http.stream("GET", self.model_url) as response:
                if response.status_code != httpx.codes.OK:
                    raise RuntimeError(
                        f"CLIP model download failed: HTTP {response.status_code} "
                        f"from {self.model_url}"
                    )
                with open(temporary, "wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            os.replace(temporary, self.model_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            if owned:
                http.close()

        return self.model_path

    def embed(self, image_bytes: bytes) -> list[float]:
        """Embed one photo.

        Args:
            image_bytes: The raw contents of an encoded image.

        Returns:
            The 512 CLIP image embeddings, unnormalised, as Python floats.

        Raises:
            ImageError: If the bytes are not a readable image. Checked before
                the model is loaded, so a bad upload fails fast.
            RuntimeError: If the model is missing or onnxruntime rejects it.
        """
        pixels = preprocess(image_bytes)
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: pixels})
        return outputs[0][0].tolist()

    def _get_session(self) -> Any:
        """Build the onnxruntime session on first use, once per embedder."""
        with self._lock:
            if self._session is None:
                import onnxruntime  # Imported here: loading the runtime is slow.

                self._session = onnxruntime.InferenceSession(
                    str(self.ensure_model()),
                    providers=["CPUExecutionProvider"],
                )
            return self._session
