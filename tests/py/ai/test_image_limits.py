"""Decompression bombs: a tiny file that declares a huge canvas must be refused BEFORE decoding.

A 20000 x 20000 one-colour PNG is a few KB on the wire and ~1.2 GB once decoded.
The API runs in 512 MB, so one upload of such a file (a dog photo, a check-in
photo) would kill the server. ``preprocess`` must read only the header, compare
the declared size against ``MAX_IMAGE_PIXELS`` and raise ``ImageError``.
"""

import io

import pytest
from PIL import Image, ImageFile

from cantrack_ai import embeddings
from cantrack_ai.embeddings import MAX_IMAGE_PIXELS, ImageError, preprocess


def png(width, height, mode="1"):
    buffer = io.BytesIO()
    Image.new(mode, (width, height), 0).save(buffer, format="PNG")
    return buffer.getvalue()


def test_cap_is_25_megapixels():
    assert MAX_IMAGE_PIXELS == 25_000_000


@pytest.mark.parametrize("side", [6000, 20000])
def test_oversized_canvas_is_an_image_error(side):
    data = png(side, side)
    assert len(data) < 2 * 1024 * 1024  # small on the wire: the size check is not enough
    with pytest.raises(ImageError):
        preprocess(data)


def test_the_check_happens_before_any_pixel_is_decoded(monkeypatch):
    decoded = []
    original = ImageFile.ImageFile.load

    def spying_load(self):
        decoded.append(self.size)
        return original(self)

    monkeypatch.setattr(ImageFile.ImageFile, "load", spying_load)
    with pytest.raises(ImageError):
        preprocess(png(6000, 6000))
    assert decoded == []


def test_a_normal_photo_still_works():
    out = preprocess(png(1200, 900, "RGB"))
    assert out.shape == (1, 3, 224, 224)


def test_exactly_at_the_cap_is_allowed():
    assert MAX_IMAGE_PIXELS == 5000 * 5000
    out = preprocess(png(5000, 5000, "L"))
    assert out.shape == (1, 3, 224, 224)


def test_pil_own_bomb_error_is_mapped_too(monkeypatch):
    monkeypatch.setattr(embeddings, "MAX_IMAGE_PIXELS", 10**12)  # let PIL's own guard fire first
    with pytest.raises(ImageError):
        preprocess(png(20000, 20000))
