from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

_QR_PREFIX = "lao-document-ocr:"


def page_id_payload(page_id: str) -> str:
    if not page_id.strip():
        raise ValueError("page_id must not be empty")
    return f"{_QR_PREFIX}{page_id.strip()}"


def render_page_id_qr(
    page_id: str,
    *,
    size_px: int,
) -> Image.Image:
    if size_px < 48:
        raise ValueError("size_px must be at least 48")

    encoder = cv2.QRCodeEncoder_create()
    matrix = encoder.encode(page_id_payload(page_id))
    if matrix.ndim != 2:
        raise RuntimeError("OpenCV returned an invalid QR matrix")

    matrix = matrix.astype(np.uint8)
    if matrix.max() <= 1:
        matrix = matrix * 255

    # OpenCV's encoder output has little/no quiet zone. Four modules is the
    # conventional QR margin and materially improves photo/scanner decoding.
    padded = np.pad(
        matrix,
        pad_width=4,
        mode="constant",
        constant_values=255,
    )
    target = max(size_px, padded.shape[0] * 2)
    return Image.fromarray(padded).resize(
        (target, target),
        Image.Resampling.NEAREST,
    ).convert("RGB")


def _payload_page_id(value: str) -> str | None:
    if not value.startswith(_QR_PREFIX):
        return None
    page_id = value[len(_QR_PREFIX) :].strip()
    return page_id or None


def _decode_array(array: np.ndarray) -> str | None:
    detector = cv2.QRCodeDetector()
    value, points, _ = detector.detectAndDecode(array)
    decoded = _payload_page_id(value)
    if decoded is not None:
        return decoded
    if points is None:
        return None

    coordinates = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    left = int(np.floor(coordinates[:, 0].min()))
    top = int(np.floor(coordinates[:, 1].min()))
    right = int(np.ceil(coordinates[:, 0].max()))
    bottom = int(np.ceil(coordinates[:, 1].max()))
    width = max(1, right - left)
    height = max(1, bottom - top)
    margin_x = max(8, round(width * 0.30))
    margin_y = max(8, round(height * 0.30))

    image_height, image_width = array.shape[:2]
    left = max(0, left - margin_x)
    top = max(0, top - margin_y)
    right = min(image_width, right + margin_x)
    bottom = min(image_height, bottom + margin_y)
    crop = array[top:bottom, left:right]
    if crop.size == 0:
        return None

    longest = max(crop.shape[:2])
    scale = max(2, int(np.ceil(320 / max(1, longest))))
    enlarged = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    retry, _, _ = detector.detectAndDecode(enlarged)
    return _payload_page_id(retry)


def decode_page_id_image(image: Image.Image) -> str | None:
    image = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(image)

    # Try the full image first. A top-region retry helps large phone photos
    # where the page-ID QR occupies a relatively small part of the frame.
    decoded = _decode_array(array)
    if decoded is not None:
        return decoded

    top_height = max(1, round(image.height * 0.40))
    top = array[:top_height, :]
    decoded = _decode_array(top)
    if decoded is not None:
        return decoded

    right_start = max(0, round(image.width * 0.50))
    top_right = array[:top_height, right_start:]
    decoded = _decode_array(top_right)
    if decoded is not None:
        return decoded

    if top_right.size:
        enlarged = cv2.resize(
            top_right,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        decoded = _decode_array(enlarged)
        if decoded is not None:
            return decoded

    # Downscaling very large images can improve QR detector stability while
    # keeping enough pixels per module for the printed marker.
    longest = max(image.width, image.height)
    if longest > 2400:
        scale = 2400 / longest
        resized = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        return _decode_array(np.asarray(resized))

    return None


def decode_page_id(path: str | Path) -> str | None:
    try:
        with Image.open(path) as image:
            return decode_page_id_image(image)
    except Exception as exc:
        raise ValueError(f"Could not decode capture page ID: {path}") from exc
