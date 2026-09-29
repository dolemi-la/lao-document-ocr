from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def _deskew(gray: np.ndarray) -> np.ndarray:
    inverted = cv2.bitwise_not(gray)
    _, threshold = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(threshold)
    if points is None or len(points) < 50:
        return gray

    # Rectangle axes are equivalent modulo 90 degrees. Normalize signed and
    # unsigned angle conventions to the nearest horizontal/vertical axis. In
    # image coordinates this is already the correction for getRotationMatrix2D;
    # negating it doubles the existing tilt instead of cancelling it.
    rectangle_angle = float(cv2.minAreaRect(points)[-1])
    angle = (rectangle_angle + 45.0) % 90.0 - 45.0

    if abs(angle) < 0.15 or abs(angle) > 12:
        return gray

    height, width = gray.shape
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(image: Image.Image) -> Image.Image:
    """Apply conservative cleanup suitable for printed document OCR."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    gray_pil = ImageOps.grayscale(image)
    gray_pil = ImageOps.autocontrast(gray_pil, cutoff=0.5)
    gray_pil = ImageEnhance.Contrast(gray_pil).enhance(1.15)

    gray = np.asarray(gray_pil)
    deskewed = _deskew(gray)
    return Image.fromarray(deskewed)
