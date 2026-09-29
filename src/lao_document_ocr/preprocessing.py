from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def _horizontal_alignment_score(mask: np.ndarray) -> float:
    """Normalized first-difference energy of horizontal foreground projections.

    Sharp row boundaries support horizontal alignment, not OCR accuracy. Use a
    binary mask so contrast changes cannot masquerade as better alignment.
    """
    rows = np.count_nonzero(mask, axis=1).astype(np.float64)
    total = float(rows.sum())
    if total == 0:
        return 0.0
    differences = np.diff(np.pad(rows / total, (1, 1)))
    return float(np.dot(differences, differences))


def _deskew_evidence(
    mask: np.ndarray,
    points: np.ndarray,
    matrix: np.ndarray,
) -> dict[str, bool | str | float | None]:
    """Reject cropping or unsupported rotation; does not guarantee OCR retention.

    Keep the existing canvas and foreground-based proposal. A candidate must
    keep every threshold-foreground pixel center inside that canvas and improve
    horizontal projection alignment strictly, without a fitted gain threshold.
    Only the small convex hull needs projection to check all foreground points.
    """
    height, width = mask.shape
    projected = cv2.transform(cv2.convexHull(points).astype(np.float64), matrix)
    coordinates = projected.reshape(-1, 2)
    inside = bool(
        np.isfinite(coordinates).all()
        and np.all(coordinates >= 0)
        and np.all(coordinates[:, 0] <= width - 1)
        and np.all(coordinates[:, 1] <= height - 1)
    )
    baseline_score = _horizontal_alignment_score(mask)
    evidence: dict[str, bool | str | float | None] = {
        "accepted": False,
        "reason": "foreground-outside-canvas",
        "foreground_within_canvas": inside,
        "baseline_score": baseline_score,
        "candidate_score": None,
    }
    if not inside:
        return evidence

    # Assess geometry with nearest-neighbor binary resampling. Actual grayscale
    # output retains the established cubic interpolation and page dimensions.
    candidate = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    score = _horizontal_alignment_score(candidate)
    accepted = score > baseline_score
    evidence.update(
        accepted=accepted,
        reason="alignment-improved" if accepted else "alignment-not-improved",
        candidate_score=score,
    )
    return evidence


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
    if not math.isfinite(rectangle_angle):
        return gray
    angle = (rectangle_angle + 45.0) % 90.0 - 45.0

    if abs(angle) < 0.15 or abs(angle) > 12:
        return gray

    height, width = gray.shape
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    if not _deskew_evidence(threshold, points, matrix)["accepted"]:
        return gray
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
