"""Image -> tensor preprocessing (must match the CLIP preprocessor the browser used) and the embedder."""

import io
import os
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from cantrack_api.ai.embeddings import ClipEmbedder, ImageError, preprocess

MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
STD = np.array([0.26862954, 0.26130258, 0.27577711])


def encode(image: Image.Image, fmt="PNG", **kw) -> bytes:
    buf = io.BytesIO()
    image.save(buf, fmt, **kw)
    return buf.getvalue()


def norm(rgb):
    return (np.array(rgb) / 255.0 - MEAN) / STD


class TestPreprocess:
    def test_shape_and_dtype(self):
        out = preprocess(encode(Image.new("RGB", (300, 200), (10, 20, 30))))
        assert out.shape == (1, 3, 224, 224)
        assert out.dtype == np.float32

    def test_solid_color_is_normalised_with_clip_mean_and_std(self):
        out = preprocess(encode(Image.new("RGB", (64, 64), (255, 0, 0))))
        for channel, expected in enumerate(norm((255, 0, 0))):
            assert out[0, channel].mean() == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
    def test_other_color_modes_are_converted_to_rgb(self, mode):
        img = Image.new("RGB", (50, 50), (200, 100, 50)).convert(mode)
        fmt = "TIFF" if mode == "CMYK" else "PNG"
        assert preprocess(encode(img, fmt)).shape == (1, 3, 224, 224)

    def test_landscape_is_resized_by_shortest_edge_then_center_cropped(self):
        # 400x200: left half black, right half white. Shortest edge 200 -> 224, width 448,
        # centre crop keeps x in [112, 336): black for the first half, white for the second.
        img = Image.new("RGB", (400, 200), (0, 0, 0))
        img.paste((255, 255, 255), (200, 0, 400, 200))
        out = preprocess(encode(img))
        black, white = norm((0, 0, 0)), norm((255, 255, 255))
        for channel in range(3):
            assert out[0, channel, 112, 10] == pytest.approx(black[channel], abs=1e-2)
            assert out[0, channel, 112, 213] == pytest.approx(white[channel], abs=1e-2)

    def test_portrait_is_center_cropped_vertically(self):
        img = Image.new("RGB", (200, 400), (0, 0, 0))
        img.paste((255, 255, 255), (0, 200, 200, 400))
        out = preprocess(encode(img))
        black, white = norm((0, 0, 0)), norm((255, 255, 255))
        for channel in range(3):
            assert out[0, channel, 10, 112] == pytest.approx(black[channel], abs=1e-2)
            assert out[0, channel, 213, 112] == pytest.approx(white[channel], abs=1e-2)

    def test_exif_orientation_is_honoured(self):
        # Phones store sideways pixels plus an orientation flag. Orientation 6 = rotate 90 CW
        # to display, so a left-black/right-white 300x100 file displays as top-black/bottom-white.
        img = Image.new("RGB", (300, 100), (0, 0, 0))
        img.paste((255, 255, 255), (150, 0, 300, 100))
        exif = Image.Exif()
        exif[0x0112] = 6
        out = preprocess(encode(img, "JPEG", exif=exif, quality=100))
        black, white = norm((0, 0, 0)), norm((255, 255, 255))
        for channel in range(3):
            assert out[0, channel, 10, 112] == pytest.approx(black[channel], abs=0.15)
            assert out[0, channel, 213, 112] == pytest.approx(white[channel], abs=0.15)

    @pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\ntruncated"])
    def test_unreadable_bytes_raise_image_error(self, data):
        with pytest.raises(ImageError):
            preprocess(data)

    def test_image_error_is_a_value_error(self):
        assert issubclass(ImageError, ValueError)


class TestEnsureModel:
    def test_downloads_the_model_when_missing(self, tmp_path):
        target = tmp_path / "sub" / "clip.onnx"
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, content=b"fake-onnx-bytes")

        embedder = ClipEmbedder(model_path=target, model_url="https://models.test/clip.onnx")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        assert embedder.ensure_model(client=client) == target
        assert target.read_bytes() == b"fake-onnx-bytes"
        assert calls == ["https://models.test/clip.onnx"]

    def test_does_not_download_when_the_file_exists(self, tmp_path):
        target = tmp_path / "clip.onnx"
        target.write_bytes(b"already here")

        def handler(request):
            raise AssertionError("must not hit the network")

        embedder = ClipEmbedder(model_path=target, model_url="https://models.test/x")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        assert embedder.ensure_model(client=client) == target
        assert target.read_bytes() == b"already here"

    def test_failed_download_raises_and_leaves_no_partial_file(self, tmp_path):
        target = tmp_path / "clip.onnx"
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
        embedder = ClipEmbedder(model_path=target, model_url="https://models.test/x")
        with pytest.raises(RuntimeError):
            embedder.ensure_model(client=client)
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_model_path_defaults_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLIP_MODEL_PATH", str(tmp_path / "from-env.onnx"))
        assert ClipEmbedder().model_path == tmp_path / "from-env.onnx"


REAL_MODEL = os.environ.get("CLIP_MODEL_PATH")


@pytest.mark.skipif(
    not (REAL_MODEL and Path(REAL_MODEL).exists()),
    reason="set CLIP_MODEL_PATH to a downloaded vision_model_quantized.onnx to run",
)
class TestRealModel:
    @pytest.fixture(scope="class")
    def embedder(self):
        return ClipEmbedder(model_path=Path(REAL_MODEL))

    @staticmethod
    def gradient(size):
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        arr[..., 0] = np.linspace(0, 255, size, dtype=np.uint8)[None, :]
        arr[..., 2] = np.linspace(255, 0, size, dtype=np.uint8)[:, None]
        return Image.fromarray(arr)

    def test_returns_512_finite_floats(self, embedder):
        vec = embedder.embed(encode(self.gradient(256)))
        assert len(vec) == 512
        assert all(isinstance(v, float) and np.isfinite(v) for v in vec)
        assert any(v != 0 for v in vec)

    def test_is_deterministic(self, embedder):
        data = encode(self.gradient(256))
        assert embedder.embed(data) == embedder.embed(data)

    def test_same_picture_at_another_size_is_closer_than_noise(self, embedder):
        from cantrack_api.ai.embeddings import cosine_similarity

        big = embedder.embed(encode(self.gradient(512)))
        small = embedder.embed(encode(self.gradient(160)))
        rng = np.random.default_rng(1)
        noise = embedder.embed(
            encode(Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)))
        )
        assert cosine_similarity(big, small) > 0.9
        assert cosine_similarity(big, small) > cosine_similarity(big, noise)

    def test_unreadable_image_raises_image_error(self, embedder):
        with pytest.raises(ImageError):
            embedder.embed(b"junk")
