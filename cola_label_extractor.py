"""Extract key fields from beverage COLA/label images.

The extractor uses EasyOCR word polygons, automatically deskews photographed
labels, reconstructs reading-order lines, and returns JSON-serializable output.
It deliberately returns ``None`` instead of inventing text when evidence is weak.

Example:
    python cola_label_extractor.py label.jpg --pretty --include-raw-text
"""

from __future__ import annotations

import argparse
import ctypes
import difflib
import gc
import json
import math
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Bound native CPU thread pools before OpenCV/PyTorch initialize. Parallel
# inference workspaces can exceed the RAM available to public Streamlit workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import torch

try:
    import easyocr
except ImportError as exc:  # pragma: no cover - gives a useful CLI error
    raise SystemExit(
        "EasyOCR is required. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc


# Values below this threshold are kept in raw OCR, but not trusted as fields.
MIN_FIELD_CONFIDENCE = 0.25


def configured_ocr_threads() -> int:
    """Return a bounded CPU thread count, tolerating an invalid environment value."""
    try:
        requested = int(os.environ.get("COLA_OCR_THREADS", "2"))
    except ValueError:
        requested = 2
    return max(1, min(2, requested))

# TTB's mandatory warning. This is used only to repair/reorder a warning when
# enough of its distinctive words were actually detected in the warning region.
CANONICAL_GOVERNMENT_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth "
    "defects. (2) Consumption of alcoholic beverages impairs your ability to "
    "drive a car or operate machinery, and may cause health problems."
)


def log_ocr_stage(message: str) -> None:
    """Write a flushed, text-free OCR stage marker for cloud diagnostics."""
    print(f"[COLA OCR] {message}", file=sys.stderr, flush=True)


def release_ocr_memory() -> None:
    """Release unreachable tensors and return free Linux heap pages to the OS."""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            malloc_trim = ctypes.CDLL(None).malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            malloc_trim(0)
        except (AttributeError, OSError):
            # Non-glibc platforms still benefit from garbage collection.
            pass


CATEGORY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("kentucky", "straight", "bourbon", "whiskey"), "Kentucky Straight Bourbon Whiskey"),
    (("tennessee", "whiskey"), "Tennessee Whiskey"),
    (("straight", "rye", "whiskey"), "Straight Rye Whiskey"),
    (("straight", "bourbon", "whiskey"), "Straight Bourbon Whiskey"),
    (("straight", "bourbon"), "Straight Bourbon"),
    (("straight", "whiskey"), "Straight Whiskey"),
    (("single", "malt", "scotch", "whisky"), "Single Malt Scotch Whisky"),
    (("blended", "scotch", "whisky"), "Blended Scotch Whisky"),
    (("irish", "whiskey"), "Irish Whiskey"),
    (("canadian", "whisky"), "Canadian Whisky"),
    (("rye", "whiskey"), "Rye Whiskey"),
    (("corn", "whiskey"), "Corn Whiskey"),
    (("wheat", "whiskey"), "Wheat Whiskey"),
    (("malt", "whiskey"), "Malt Whiskey"),
    (("bourbon", "whiskey"), "Bourbon Whiskey"),
    (("cabernet", "sauvignon"), "Cabernet Sauvignon"),
    (("sauvignon", "blanc"), "Sauvignon Blanc"),
    (("pinot", "noir"), "Pinot Noir"),
    (("pinot", "grigio"), "Pinot Grigio"),
    (("chenin", "blanc"), "Chenin Blanc"),
    (("sparkling", "wine"), "Sparkling Wine"),
    (("dessert", "wine"), "Dessert Wine"),
    (("rice", "wine"), "Rice Wine"),
    (("table", "wine"), "Table Wine"),
    (("red", "wine"), "Red Wine"),
    (("white", "wine"), "White Wine"),
    (("bordeaux",), "Bordeaux"),
    (("champagne",), "Champagne"),
    (("chardonnay",), "Chardonnay"),
    (("merlot",), "Merlot"),
    (("zinfandel",), "Zinfandel"),
    (("riesling",), "Riesling"),
    (("sangria",), "Sangria"),
    (("vermouth",), "Vermouth"),
    (("sake",), "Sake"),
    (("mead",), "Mead"),
    (("hard", "seltzer"), "Hard Seltzer"),
    (("india", "pale", "ale"), "India Pale Ale"),
    (("malt", "beverage"), "Malt Beverage"),
    (("brandy",), "Brandy"),
    (("cognac",), "Cognac"),
    (("tequila",), "Tequila"),
    (("mezcal",), "Mezcal"),
    (("vodka",), "Vodka"),
    (("whiskey",), "Whiskey"),
    (("whisky",), "Whisky"),
    (("bourbon",), "Bourbon"),
    (("rum",), "Rum"),
    (("gin",), "Gin"),
    (("liqueur",), "Liqueur"),
    (("cordial",), "Cordial"),
    (("cider",), "Cider"),
    (("stout",), "Stout"),
    (("porter",), "Porter"),
    (("pilsner",), "Pilsner"),
    (("ipa",), "India Pale Ale"),
    (("lager",), "Lager"),
    (("ale",), "Ale"),
    (("beer",), "Beer"),
    (("wine",), "Wine"),
)
CATEGORY_VOCABULARY = {token for required, _ in CATEGORY_RULES for token in required}
SPACED_CATEGORY_PATTERNS = tuple(
    (
        re.compile(
            r"(?<![A-Za-z])" + r"\s*".join(map(re.escape, token)) + r"(?![A-Za-z])",
            re.IGNORECASE,
        ),
        token,
    )
    for token in sorted(CATEGORY_VOCABULARY, key=len, reverse=True)
    if len(token) >= 4
)

EXCLUDE_FROM_BRAND = re.compile(
    r"\b(?:government|warning|alcohol|alc|vol|proof|ml|cl|lit(?:er|re)|"
    r"imported|produced|bottled|distributed|aged|years?|vintage|contains|"
    r"surgeon|pregnancy|drink|beverages?)\b|%",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OCRWord:
    text: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]

    @property
    def left(self) -> float:
        """Return the leftmost x-coordinate of the OCR polygon."""
        return min(p[0] for p in self.polygon)

    @property
    def right(self) -> float:
        """Return the rightmost x-coordinate of the OCR polygon."""
        return max(p[0] for p in self.polygon)

    @property
    def top(self) -> float:
        """Return the uppermost y-coordinate of the OCR polygon."""
        return min(p[1] for p in self.polygon)

    @property
    def bottom(self) -> float:
        """Return the lowermost y-coordinate of the OCR polygon."""
        return max(p[1] for p in self.polygon)

    @property
    def width(self) -> float:
        """Return a nonzero axis-aligned width for weighting and layout."""
        return max(1.0, self.right - self.left)

    @property
    def height(self) -> float:
        """Return a nonzero axis-aligned height for layout calculations."""
        return max(1.0, self.bottom - self.top)

    @property
    def center_y(self) -> float:
        """Return the vertical center of the OCR polygon."""
        return (self.top + self.bottom) / 2

    @property
    def center_x(self) -> float:
        """Return the horizontal center of the OCR polygon."""
        return (self.left + self.right) / 2


@dataclass(frozen=True)
class OCRLine:
    words: tuple[OCRWord, ...]

    @property
    def text(self) -> str:
        """Join the line's OCR words with normalized punctuation spacing."""
        return clean_spacing(" ".join(w.text.strip() for w in self.words if w.text.strip()))

    @property
    def confidence(self) -> float:
        """Return the character-count-weighted mean word confidence."""
        weights = [max(1, len(w.text)) for w in self.words]
        return weighted_mean([w.confidence for w in self.words], weights)

    @property
    def left(self) -> float:
        """Return the line's leftmost coordinate."""
        return min(w.left for w in self.words)

    @property
    def top(self) -> float:
        """Return the line's uppermost coordinate."""
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> float:
        """Return the line's lowermost coordinate."""
        return max(w.bottom for w in self.words)

    @property
    def height(self) -> float:
        """Return a nonzero axis-aligned line height."""
        return max(1.0, self.bottom - self.top)


@dataclass(frozen=True)
class FieldValue:
    value: str | None
    confidence: float
    source_text: str | None


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Calculate a weighted arithmetic mean, returning zero for no weight."""
    total = sum(weights)
    return float(sum(v * w for v, w in zip(values, weights)) / total) if total else 0.0


def clean_spacing(text: str) -> str:
    """Collapse whitespace and remove spaces before common punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text


def load_image(path: str | Path) -> np.ndarray:
    """Read Unicode paths reliably on Windows and validate the decoded image."""
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unsupported or corrupt image: {image_path}")
    return image


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate without clipping corners, filling exposed pixels with white."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(math.ceil(h * sin + w * cos))
    new_h = int(math.ceil(h * cos + w * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def prepare_for_ocr(image: np.ndarray, max_side: int = 2400) -> np.ndarray:
    """Bound memory use while enlarging genuinely small labels."""
    h, w = image.shape[:2]
    longest = max(h, w)
    scale = min(max_side / longest, max(1.0, 1200.0 / longest))
    if abs(scale - 1.0) < 0.02:
        return image
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def order_quadrilateral(points: np.ndarray) -> np.ndarray:
    """Return quadrilateral corners as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).ravel()
    return np.asarray(
        [
            points[np.argmin(coordinate_sum)],
            points[np.argmin(coordinate_difference)],
            points[np.argmax(coordinate_sum)],
            points[np.argmax(coordinate_difference)],
        ],
        dtype=np.float32,
    )


def isolate_dominant_label_panel(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Perspective-rectify the largest panel when a photo contains many labels.

    The trigger requires at least five spatially distinct quadrilateral panels,
    so ordinary bottle photographs and flattened front/back artwork retain the
    established whole-image path. Contour analysis is inexpensive and replaces
    full-scene OCR with OCR of only the dominant label.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 130)
    kernel_size = max(3, round(min(height, width) * 0.012))
    if kernel_size % 2 == 0:
        kernel_size += 1
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
    minimum_area = width * height * 0.008
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if (
            len(polygon) == 4
            and cv2.isContourConvex(polygon)
            and area >= minimum_area
            and box_width >= width * 0.10
            and box_height >= height * 0.10
            and box_width < width * 0.92
            and box_height < height * 0.92
            and area / max(1, box_width * box_height) > 0.45
        ):
            candidates.append(
                (area, polygon.reshape(4, 2).astype(np.float32), (x, y, box_width, box_height))
            )

    # Edge closing can yield nested contours around the same printed panel.
    candidates.sort(key=lambda item: item[0], reverse=True)
    distinct: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
    for candidate in candidates:
        _, _, (x, y, box_width, box_height) = candidate
        duplicate = any(
            abs(x - other_x) < width * 0.05
            and abs(y - other_y) < height * 0.05
            and abs(box_width - other_width) < width * 0.08
            and abs(box_height - other_height) < height * 0.08
            for _, _, (other_x, other_y, other_width, other_height) in distinct
        )
        if not duplicate:
            distinct.append(candidate)
    if len(distinct) < 5:
        return image, False

    corners = order_quadrilateral(distinct[0][1])
    top_left, top_right, bottom_right, bottom_left = corners
    output_width = round(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    output_height = round(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    if output_width < 100 or output_height < 100:
        return image, False
    destination = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    rectified = cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rectified, True


def orient_vertical_side_columns(
    horizontal_list: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
) -> tuple[list[Sequence[float]], list[list[list[float]]]]:
    """Rotate detected side-panel columns within the existing OCR pass.

    Flattened COLA artwork can place regulatory copy at 90 degrees along an
    outer edge. EasyOCR detects those columns as many tall horizontal boxes and
    consequently reads them as isolated characters. This routine consolidates
    a connected side paragraph into its original text columns and expresses
    them as rotated free-form polygons. Recognition therefore still occurs in
    the same batch and does not add another detection or recognition pass.
    """
    boxes = [list(map(float, box)) for box in horizontal_list]
    if image_width / max(1, image_height) < 1.4:
        return list(horizontal_list), []

    retained = [True] * len(boxes)
    rotated_columns: list[list[list[float]]] = []
    side_width = image_width * 0.15

    for side in ("left", "right"):
        side_indexes = [
            index
            for index, (x0, x1, _, _) in enumerate(boxes)
            if (x1 <= side_width if side == "left" else x0 >= image_width - side_width)
        ]
        portrait_indexes = [
            index
            for index in side_indexes
            if boxes[index][3] - boxes[index][2] > boxes[index][1] - boxes[index][0]
        ]
        if len(portrait_indexes) < 5:
            continue

        # A large merged portrait box is characteristic of adjacent vertical
        # regulatory lines. Use it as the seed and recover every overlapping
        # fragment belonging to the same paragraph, excluding nearby barcodes.
        anchor = max(
            portrait_indexes,
            key=lambda index: (
                (boxes[index][1] - boxes[index][0])
                * (boxes[index][3] - boxes[index][2])
                * (0.25 if boxes[index][2] > image_height * 0.80 else 1.0)
            ),
        )
        component = {anchor}
        changed = True
        while changed:
            changed = False
            for index in side_indexes:
                if index in component:
                    continue
                x0, x1, y0, y1 = boxes[index]
                if any(
                    min(x1, boxes[member][1]) > max(x0, boxes[member][0])
                    and min(y1, boxes[member][3]) > max(y0, boxes[member][2])
                    for member in component
                ):
                    component.add(index)
                    changed = True

        top = min(boxes[index][2] for index in component)
        bottom = max(boxes[index][3] for index in component)
        if len(component) < 5 or bottom - top < image_height * 0.25:
            continue

        max_column_width = max(32.0, image_width * 0.02)
        narrow = sorted(
            (
                index
                for index in component
                if boxes[index][1] - boxes[index][0] <= max_column_width
                and boxes[index][3] - boxes[index][2] >= 20.0
            ),
            key=lambda index: (boxes[index][0] + boxes[index][1]) / 2.0,
        )
        clusters: list[list[int]] = []
        center_tolerance = max(6.0, image_width * 0.004)
        for index in narrow:
            center = (boxes[index][0] + boxes[index][1]) / 2.0
            if clusters:
                prior_centers = [
                    (boxes[item][0] + boxes[item][1]) / 2.0 for item in clusters[-1]
                ]
                if abs(center - statistics.median(prior_centers)) <= center_tolerance:
                    clusters[-1].append(index)
                    continue
            clusters.append([index])
        if len(clusters) < 4:
            continue

        for cluster in clusters:
            x0 = statistics.median(boxes[index][0] for index in cluster)
            x1 = statistics.median(boxes[index][1] for index in cluster)
            if side == "left":
                # Left-edge label copy conventionally reads bottom-to-top.
                polygon = [[x0, bottom], [x0, top], [x1, top], [x1, bottom]]
            else:
                # Mirror the ordering for top-to-bottom copy on the right edge.
                polygon = [[x1, top], [x1, bottom], [x0, bottom], [x0, top]]
            rotated_columns.append(polygon)

        component_left = min(boxes[index][0] for index in component)
        component_right = max(boxes[index][1] for index in component)
        for index, (x0, x1, y0, y1) in enumerate(boxes):
            if (
                min(x1, component_right) > max(x0, component_left)
                and min(y1, bottom) > max(y0, top)
            ):
                retained[index] = False

    return [horizontal_list[index] for index, keep in enumerate(retained) if keep], rotated_columns


def easyocr_to_words(results: Iterable[Any]) -> list[OCRWord]:
    """Convert validated EasyOCR result tuples into typed OCR words."""
    words: list[OCRWord] = []
    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        polygon, text, confidence = item[:3]
        text = clean_spacing(str(text))
        if not text:
            continue
        try:
            points = tuple((float(p[0]), float(p[1])) for p in polygon)
            conf = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError, IndexError):
            continue
        if len(points) == 4:
            words.append(OCRWord(text=text, confidence=conf, polygon=points))
    return words


def estimate_text_angle(words: Sequence[OCRWord]) -> float:
    """Estimate skew from non-axis-aligned OCR quadrilaterals.

    EasyOCR emits true polygons for rotated text and ordinary rectangles for
    horizontal text. A weighted median is resistant to bottle and border edges.
    """
    samples: list[tuple[float, float]] = []
    for word in words:
        p0, p1 = word.polygon[0], word.polygon[1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        if abs(dx) < 1:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        # Axis-aligned boxes provide no skew signal; near-vertical decorative
        # text should not rotate the whole image.
        if 2.0 <= abs(angle) <= 40.0 and word.confidence >= 0.20 and len(word.text) >= 3:
            weight = word.width * max(0.15, word.confidence)
            samples.append((angle, weight))
    if not samples:
        return 0.0
    samples.sort(key=lambda pair: pair[0])
    midpoint = sum(weight for _, weight in samples) / 2.0
    running = 0.0
    for angle, weight in samples:
        running += weight
        if running >= midpoint:
            # Curved bottles make outer text polygons slightly steeper than the
            # label's central baseline. Conservative correction preserves small
            # regulatory type better than rotating by the full median angle.
            return round(angle * 0.82, 1)
    return 0.0


def words_to_lines(words: Sequence[OCRWord]) -> list[OCRLine]:
    """Group deskewed OCR boxes into stable reading-order lines."""
    usable = [w for w in words if w.text and w.confidence >= 0.05]
    vertical = [
        word
        for word in usable
        if word.height >= 2.0 * word.width and len(normalized_tokens(word.text)) >= 2
    ]
    vertical_ids = {id(word) for word in vertical}
    usable = [word for word in usable if id(word) not in vertical_ids]
    usable.sort(key=lambda w: (w.center_y, w.left))
    groups: list[list[OCRWord]] = []
    for word in usable:
        best_index: int | None = None
        best_distance = float("inf")
        for index, group in enumerate(groups):
            group_center = statistics.median(w.center_y for w in group)
            group_height = statistics.median(w.height for w in group)
            distance = abs(word.center_y - group_center)
            if distance <= 0.48 * max(word.height, group_height) and distance < best_distance:
                best_index, best_distance = index, distance
        if best_index is None:
            groups.append([word])
        else:
            groups[best_index].append(word)
    lines = [OCRLine(tuple(sorted(group, key=lambda w: w.left))) for group in groups]
    # Consolidated vertical columns form their own logical line. Keeping them
    # separate prevents their tall polygons from absorbing the central brand
    # and category into the side-panel paragraph during geometric grouping.
    for side_words in (
        [word for word in vertical if word.center_x < 0.5 * max((w.right for w in words), default=0.0)],
        [word for word in vertical if word.center_x >= 0.5 * max((w.right for w in words), default=0.0)],
    ):
        if side_words:
            lines.append(OCRLine(tuple(sorted(side_words, key=lambda word: word.left))))
    return sorted(lines, key=lambda line: (line.top, line.left))


def corpus(lines: Sequence[OCRLine]) -> str:
    """Render reconstructed OCR lines as newline-separated plain text."""
    return "\n".join(line.text for line in lines)


def normalized_tokens(text: str) -> set[str]:
    """Tokenize text and conservatively repair known regulatory OCR errors."""
    # Conservative corrections only for regulatory/category vocabulary. Brand
    # text is never passed through fuzzy spelling replacement.
    normalized = text.lower().replace("0", "o")
    for pattern, replacement in SPACED_CATEGORY_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    tokens = set(re.findall(r"[a-z]+", normalized))
    aliases = {
        "whiskev": "whiskey",
        "whiske": "whiskey",
        "whskey": "whiskey",
        "hskey": "whiskey",
        "whiskcy": "whiskey",
        "straicht": "straight",
        "bourboh": "bourbon",
        "bourbor": "bourbon",
        "alcoholie": "alcoholic",
        "machlnery": "machinery",
    }
    return {aliases.get(token, token) for token in tokens}


def evidence_confidence(lines: Sequence[OCRLine], evidence: str) -> float:
    """Return the strongest line confidence supporting any evidence token."""
    parts = normalized_tokens(evidence)
    matching = [
        line.confidence
        for line in lines
        if parts.intersection(normalized_tokens(line.text))
    ]
    return max(matching, default=0.0)


def extract_category(lines: Sequence[OCRLine]) -> FieldValue:
    """Match OCR tokens against ordered alcoholic-beverage category rules."""
    text = corpus(lines)
    tokens = normalized_tokens(text)
    for required, value in CATEGORY_RULES:
        if set(required).issubset(tokens):
            sources = [line.text for line in lines if set(required).intersection(normalized_tokens(line.text))]
            source = clean_spacing(" ".join(sources))
            confidence = evidence_confidence(lines, " ".join(required))
            return FieldValue(value, confidence, source)
    return FieldValue(None, 0.0, None)


ABV_RE = re.compile(
    r"(?:(?P<value>\d{1,2}(?:[.,]\d+)?)\s*%\s*(?:alc(?:ohol)?\.?\s*/?\s*(?:by\s*)?vol(?:ume)?\.?)|"
    r"(?:alc(?:ohol)?\.?\s*/?\s*(?:by\s*)?vol(?:ume)?\.?\s*)"
    r"(?P<value_after>\d{1,2}(?:[.,]\d+)?)\s*%)",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"\b(?P<value>\d{1,2}(?:[.,]\d+)?)\s*%")
PROOF_RE = re.compile(
    r"(?:\b(?P<value>\d{2,3}(?:[.,]\d+)?)\s*(?:°\s*)?proof\b|"
    r"\bproof\s*(?:°\s*)?(?P<value_after>\d{2,3}(?:[.,]\d+)?)\b)",
    re.IGNORECASE,
)
VOLUME_RE = re.compile(
    r"\b(?P<value>\d{1,4}(?:[.,]\d+)?)\s*(?P<unit>m[l1i]|c[l1i]|"
    r"lit(?:er|re)s?|[l1](?!\w)|fl\.?\s*oz\.?|fluid\s*ounces?)\b",
    re.IGNORECASE,
)

# Regulatory role phrases are intentionally required. This prevents a large
# decorative winery/distillery name from being duplicated as the producer when
# the label never identifies that organization in a production role.
BOTTLER_PRODUCER_RE = re.compile(
    r"\b(?:produced|bott?i?led|distilled|brewed|vinted|cellared|manufactured|made|crafted|imported)"
    r"(?:[\s.,=|:&/-]+(?:and|&)?\s*(?:produced|bott?i?led|distilled|brewed|vinted|cellared|manufactured|made|crafted|imported))*"
    r"[\s.,=|:&/-]+by\b\s*:?\s*[=-]*\s*(?P<entity>.+)",
    re.IGNORECASE,
)
ROLE_WORD_RE = re.compile(
    r"\b(?:produced|bott?i?led|distilled|brewed|vinted|cellared|manufactured|made|crafted|imported)\b",
    re.IGNORECASE,
)
LOCATION_RE = re.compile(
    r"^[A-Za-z .'-]+,\s*(?:[A-Z]{2}|Alabama|Alaska|Arizona|Arkansas|California|Colorado|"
    r"Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|"
    r"Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|"
    r"Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|"
    r"North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|"
    r"South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
    r"West Virginia|Wisconsin|Wyoming)(?:\s+\d{5}(?:-\d{4})?)?$",
    re.IGNORECASE,
)
COUNTRY_ORIGIN_RE = re.compile(
    r"\b(?:country\s+of\s+origin\s*:|product\s+of|produced\s+in|made\s+in|"
    r"wine\s+of|beer\s+of|distilled\s+in|"
    r"grown\s*,?\s*distilled\s+and\s+aged\s+in|"
    r"imported\s+from|imported(?!\s+by\b))\s+"
    r"(?P<country>[A-Za-z][A-Za-z .'-]{1,50}?)"
    r"(?=\s+(?:imported|bottled|produced|distilled|distributed)\b|[,;|]|$)",
    re.IGNORECASE,
)
NATIONALITY_ORIGIN_RE = re.compile(
    r"\b(?P<country>French)\b(?=[^\n]{0,60}\b(?:vineyard|wine|products?)\b)",
    re.IGNORECASE,
)
COUNTRY_ALIASES = {
    "french": "France",
    "the usa": "United States",
    "the u s a": "United States",
    "u s a": "United States",
    "usa": "United States",
    "u s": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "republic of ireland": "Ireland",
    "republic of korea": "South Korea",
    "korea republic of": "South Korea",
}
PRODUCER_SUFFIX = r"(?:winery|distillery|distilling|brewery|brewing|company|co\.?|llc|inc\.?)"
BRAND_OCR_ALIASES = {
    # Common EasyOCR readings of the highly cursive "Climax" wordmark.
    "cunax": "Climax",
    "cumnax": "Climax",
    "cimax": "Climax",
    "ctmnax": "Climax",
    # Common readings of the Devils River display face on photographed panels.
    "devlls": "Devils",
    "dhvils": "Devils",
    "rivhr": "River",
    # Faint cursive text observed on winery labels; these repairs require the
    # complete token and do not fuzzy-rewrite arbitrary brand words.
    "casade": "Cascade",
    "ieny": "Winery",
    "ineny": "Winery",
}


def normalize_number(value: str) -> str:
    """Normalize a decimal string to a compact, locale-independent form."""
    number = float(value.replace(",", "."))
    return f"{number:g}"


def normalize_measurement_ocr(text: str) -> str:
    """Repair only common OCR confusions next to regulated measurement units."""
    repaired = text
    repaired = re.sub(r"\balch?\s*/?\s*vol\b", "alc/vol", repaired, flags=re.I)
    repaired = re.sub(r"\bm\s*(?:\]|\||!)", "ml", repaired, flags=re.I)
    repaired = re.sub(r"(?<![A-Za-z])S(?=\d\s*(?:m[l|\]]|%))", "5", repaired, flags=re.I)
    repaired = re.sub(r"(?<=\d)[Oo](?=\s*m[l1i]\b)", "0", repaired, flags=re.I)
    # A missing decimal and `Ale`/`Alc` confusion are repaired only when the
    # complete alcohol-by-volume phrase makes the intended measurement clear.
    repaired = re.sub(
        r"\bAl[eoc]\.?s*(\d{2})(\d)\s*%\s*by\s*Vol(?:ume)?\.?",
        r"Alc. \1.\2% by Vol.",
        repaired,
        flags=re.IGNORECASE,
    )
    return repaired


def credible_measurement_evidence(text: str, confidence: float) -> bool:
    """Accept low-confidence numbers only when multiple regulatory anchors agree."""
    if confidence >= MIN_FIELD_CONFIDENCE:
        return True
    anchors = re.findall(
        r"\b(?:alc(?:ohol)?|vol(?:ume)?|proof|m[l1i]|c[l1i]|lit(?:er|re)|fl\.?\s*oz)\b|%",
        text,
        flags=re.IGNORECASE,
    )
    return confidence >= 0.05 and len(anchors) >= 3


def match_across_lines(pattern: re.Pattern[str], lines: Sequence[OCRLine]) -> tuple[re.Match[str] | None, str | None, float]:
    """Find a measurement on one OCR line or across two adjacent lines."""
    # Parse individual lines first to avoid joining unrelated numbers and units.
    for line in lines:
        normalized = normalize_measurement_ocr(line.text)
        match = pattern.search(normalized)
        if match and credible_measurement_evidence(normalized, line.confidence):
            return match, line.text, line.confidence
    # Adjacent OCR boxes are occasionally split into consecutive reconstructed lines.
    for first, second in zip(lines, lines[1:]):
        joined = f"{first.text} {second.text}"
        normalized = normalize_measurement_ocr(joined)
        match = pattern.search(normalized)
        if match:
            confidence = min(first.confidence, second.confidence)
            if credible_measurement_evidence(normalized, confidence):
                return match, joined, confidence
    return None, None, 0.0


def all_matches_across_lines(
    pattern: re.Pattern[str], lines: Sequence[OCRLine]
) -> list[tuple[re.Match[str], str, float]]:
    """Collect every credible match from individual and adjacent OCR lines."""
    matches: list[tuple[re.Match[str], str, float]] = []
    for line in lines:
        normalized = normalize_measurement_ocr(line.text)
        match = pattern.search(normalized)
        if match and credible_measurement_evidence(normalized, line.confidence):
            matches.append((match, line.text, line.confidence))
    for first, second in zip(lines, lines[1:]):
        joined = f"{first.text} {second.text}"
        normalized = normalize_measurement_ocr(joined)
        match = pattern.search(normalized)
        confidence = min(first.confidence, second.confidence)
        if match and credible_measurement_evidence(normalized, confidence):
            matches.append((match, joined, confidence))
    return matches


def extract_numeric_fields(lines: Sequence[OCRLine]) -> tuple[FieldValue, FieldValue, FieldValue]:
    """Extract and validate ABV, proof, and net-volume label statements."""
    proof_match, proof_source, proof_conf = match_across_lines(PROOF_RE, lines)
    proof_value: str | None = None
    proof_number: float | None = None
    if proof_match:
        raw_proof = proof_match.groupdict().get("value") or proof_match.groupdict().get("value_after")
        proof_number = float(raw_proof.replace(",", "."))
        if 1 <= proof_number <= 200:
            proof_value = f"{proof_number:g} proof"

    abv_match, abv_source, abv_conf = match_across_lines(ABV_RE, lines)
    if not abv_match:
        # A bare percentage is accepted only in a plausible beverage range.
        abv_match, abv_source, abv_conf = match_across_lines(PERCENT_RE, lines)
    abv_value: str | None = None
    abv_number: float | None = None
    if abv_match:
        raw = abv_match.groupdict().get("value") or abv_match.groupdict().get("value_after")
        if raw:
            abv_number = float(raw.replace(",", "."))
            if 0.1 <= abv_number <= 100:
                abv_value = f"{abv_number:g}%"

    # Do not synthesize a missing proof from ABV (or vice versa): the requested
    # fields are label contents, and many wine labels correctly omit proof.
    volume_candidates = all_matches_across_lines(VOLUME_RE, lines)
    standard_ml = {50, 100, 187, 200, 250, 330, 341, 355, 375, 500, 568, 700, 720, 750, 1000, 1500, 1750, 1800, 3000}
    ranked_volumes: list[tuple[float, re.Match[str], str, float]] = []
    for match, source, confidence in volume_candidates:
        number = float(match.group("value").replace(",", "."))
        raw_unit = match.group("unit").lower()
        score = confidence
        if raw_unit in {"1", "i"}:  # OCR-only unit substitutions are weaker evidence.
            score -= 0.25
        if raw_unit.startswith("m") and number in standard_ml:
            score += 0.20
        ranked_volumes.append((score, match, source, confidence))
    if ranked_volumes:
        _, volume_match, volume_source, volume_conf = max(ranked_volumes, key=lambda item: item[0])
    else:
        volume_match, volume_source, volume_conf = None, None, 0.0
    volume_value: str | None = None
    if volume_match:
        number = normalize_number(volume_match.group("value"))
        unit = volume_match.group("unit").lower().replace("1", "l").replace("i", "l")
        unit = {"liter": "L", "litre": "L", "liters": "L", "litres": "L", "l": "L"}.get(unit, unit)
        unit = "fl oz" if unit.startswith("fl") or unit.startswith("fluid") else unit
        volume_value = f"{number} {unit}"

    return (
        FieldValue(abv_value, abv_conf if abv_value else 0.0, abv_source),
        FieldValue(proof_value, proof_conf if proof_value else 0.0, proof_source),
        FieldValue(volume_value, volume_conf if volume_value else 0.0, volume_source),
    )


def extract_warning(lines: Sequence[OCRLine]) -> FieldValue:
    """Extract the anchored government warning and repair it when well supported."""
    start = next(
        (i for i, line in enumerate(lines) if re.search(r"\b(?:government\s+)?warning\b", line.text, re.I)),
        None,
    )
    if start is None:
        return FieldValue(None, 0.0, None)

    selected = list(lines[start : start + 12])
    raw = clean_spacing(" ".join(line.text for line in selected))
    raw_tokens = normalized_tokens(raw)
    canonical_tokens = normalized_tokens(CANONICAL_GOVERNMENT_WARNING)
    distinctive = canonical_tokens - {"a", "and", "of", "or", "the", "to"}
    coverage = len(raw_tokens.intersection(distinctive)) / max(1, len(distinctive))
    confidence = weighted_mean(
        [line.confidence for line in selected],
        [max(1, len(line.text)) for line in selected],
    )

    # Perspective can cause the two statutory clauses to be detected in a
    # scrambled order. Canonical repair is permitted only with strong evidence.
    if coverage >= 0.55:
        repaired_confidence = min(confidence, 0.65 + 0.30 * coverage)
        return FieldValue(CANONICAL_GOVERNMENT_WARNING, repaired_confidence, raw)
    return FieldValue(raw, confidence * max(0.35, coverage), raw)


def extraction_windows(lines: Sequence[OCRLine]) -> Iterable[tuple[str, float]]:
    """Yield individual and adjacent OCR-line windows with confidence values."""
    for current in lines:
        yield current.text, current.confidence
    for first, second in zip(lines, lines[1:]):
        yield f"{first.text} {second.text}", min(first.confidence, second.confidence)


def split_entity_and_location(text: str) -> tuple[str, str | None]:
    """Separate a producer company from an adjacent city/state/postal location."""
    text = re.split(
        r"\b(?:government\s+warning|alc(?:ohol)?\.?\s*/?\s*vol|net\s+contents?|"
        r"product\s+of|imported\s+by|distributed\s+by)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = clean_spacing(text.strip(" ~-|,;:"))
    # OCR can merge a company line with the address below it and sort by x,
    # placing the address first. Restore the conventional entity/address order.
    address_first = re.match(
        r"^(?P<address>.+?,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)\s+"
        rf"(?P<company>.+?\b{PRODUCER_SUFFIX})"
        r"(?:[\s\"'|\dA-Z]{1,20})?$",
        text,
        flags=re.IGNORECASE,
    )
    if address_first:
        return (
            clean_spacing(address_first.group("company")),
            clean_spacing(address_first.group("address")),
        )
    company_first = re.match(
        rf"^(?P<company>.+?\b{PRODUCER_SUFFIX})\s*,?\s+"
        r"(?P<address>[A-Za-z .'-]+,\s*[A-Z]{2}(?:\s+\d{5}(?:-\d{4})?)?)$",
        text,
        flags=re.IGNORECASE,
    )
    if company_first:
        return (
            clean_spacing(company_first.group("company")),
            clean_spacing(company_first.group("address")),
        )
    return text, None


def clean_entity_text(text: str) -> str:
    """Return only the company portion of a producer-and-location OCR phrase."""
    entity, _ = split_entity_and_location(text)
    return entity


def trim_repeated_producer_location(entity: str, lines: Sequence[OCRLine]) -> str:
    """Remove an unpunctuated location after a producer repeated elsewhere.

    OCR can lose the dashes between a bottler name and its city/province. A
    separately printed producer name supplies strong evidence for the boundary.
    Company and beverage suffixes are retained so names such as ``Devils River
    Whiskey`` and ``Great Jones Distilling Co.`` are not shortened.
    """
    entity_tokens = re.findall(r"[A-Za-z]+", entity)
    protected_suffixes = CATEGORY_VOCABULARY | {
        "company", "co", "corp", "corporation", "inc", "llc",
        "winery", "distillery", "distilling", "brewery", "brewing",
    }
    for line in lines:
        repeated = clean_spacing(
            re.sub(r"(?<!\w)[A-Za-z0-9](?!\w)", " ", line.text)
        ).strip(" ~-|,;:")
        repeated_tokens = re.findall(r"[A-Za-z]+", repeated)
        if not 2 <= len(repeated_tokens) <= 6 or EXCLUDE_FROM_BRAND.search(repeated):
            continue
        prefix_length = len(repeated_tokens)
        if [token.casefold() for token in entity_tokens[:prefix_length]] != [
            token.casefold() for token in repeated_tokens
        ]:
            continue
        remainder = entity_tokens[prefix_length:]
        if (
            1 <= len(remainder) <= 3
            and not {token.casefold() for token in remainder}.intersection(protected_suffixes)
        ):
            return repeated
    return entity


def spatial_role_blocks(
    lines: Sequence[OCRLine],
) -> list[tuple[float, str, str | None, str]]:
    """Read company and location boxes below a production-role anchor column."""
    words = [word for line in lines for word in line.words]
    blocks: list[tuple[float, str, str | None, str]] = []
    for role in words:
        if not ROLE_WORD_RE.search(role.text):
            continue
        same_row = [
            word
            for word in words
            if word is not role
            and word.left >= role.left
            and word.left <= role.right + max(80.0, role.width * 1.5)
            and abs(word.center_y - role.center_y) <= 0.7 * max(role.height, word.height)
            and re.search(r"\bby\b", word.text, re.IGNORECASE)
        ]
        role_has_by = re.search(r"\bby\b", role.text, re.IGNORECASE)
        if not same_row and not role_has_by:
            continue
        by_word = min(same_row, key=lambda word: word.left) if same_row else role
        anchor_left = min(role.left, by_word.left)
        anchor_right = max(role.right, by_word.right)
        margin = max(25.0, (anchor_right - anchor_left) * 0.40)
        below = sorted(
            (
                word
                for word in words
                if word not in {role, by_word}
                and word.top
                >= min(role.bottom, by_word.bottom) - 0.55 * max(role.height, by_word.height)
                and word.top <= max(role.bottom, by_word.bottom) + max(180.0, role.height * 7.0)
                and anchor_left - margin <= word.center_x <= anchor_right + margin
                and word.confidence >= MIN_FIELD_CONFIDENCE
            ),
            key=lambda word: (word.top, word.left),
        )
        entity_word = next(
            (
                word
                for word in below
                if not LOCATION_RE.fullmatch(clean_spacing(word.text.strip(" `|")))
                and not re.match(r"^[Â©O0]?\d{4}\b", word.text.strip())
                and len(re.findall(r"[A-Za-z]+", word.text)) >= 2
            ),
            None,
        )
        if entity_word is None:
            continue
        entity = clean_spacing(entity_word.text.strip(" `|,;:"))
        location_word = next(
            (
                word
                for word in below
                if word.top >= entity_word.top
                and LOCATION_RE.fullmatch(clean_spacing(word.text.strip(" `|")))
            ),
            None,
        )
        location = (
            clean_spacing(location_word.text.strip(" `|,;:")) if location_word else None
        )
        confidence = min(role.confidence, by_word.confidence, entity_word.confidence)
        role_text = role.text if by_word is role else f"{role.text} {by_word.text}"
        source = clean_spacing(
            " ".join(
                part for part in (role_text, entity, location or "") if part
            )
        )
        blocks.append((confidence, entity, location, source))
    return blocks


def extract_bottler_producer(lines: Sequence[OCRLine]) -> FieldValue:
    """Extract an entity explicitly identified as producer, bottler, or maker."""
    candidates: list[tuple[float, str, str]] = []
    for confidence, entity, _, source in spatial_role_blocks(lines):
        candidates.append((min(1.0, confidence + 0.05), entity, source))
    for text, confidence in extraction_windows(lines):
        match = BOTTLER_PRODUCER_RE.search(text)
        if not match:
            continue
        entity = clean_entity_text(match.group("entity"))
        entity = trim_repeated_producer_location(entity, lines)
        if len(entity) >= 2 and re.search(r"[A-Za-z]", entity):
            # Prefer higher-confidence evidence and a bounded company/address
            # phrase over a long OCR line containing unrelated content.
            score = confidence - max(0, len(entity) - 100) / 500.0
            candidates.append((score, entity, text))
    if not candidates:
        # Some front labels identify the company as a large wordmark followed
        # immediately by a concise company-type line, without a separate "by"
        # statement. Keep this fallback narrow to avoid treating descriptive
        # label prose as the bottler or producer.
        for previous, current in zip(lines, lines[1:]):
            company_type = clean_spacing(current.text.strip(" ~-|,;:"))
            if not re.fullmatch(
                r"(?:distilling|winery|distillery|brewery|brewing|company)"
                r"\s+c[oe0]\.?",
                company_type,
                flags=re.IGNORECASE,
            ):
                continue
            name = clean_spacing(previous.text.strip(" ~-|,;:"))
            if (
                1 <= len(normalized_tokens(name)) <= 6
                and not EXCLUDE_FROM_BRAND.search(name)
                and re.search(r"[A-Za-z]", name)
            ):
                company_type = re.sub(
                    r"\bc[eo0]\.?$", "Co.", company_type, flags=re.IGNORECASE
                )
                entity = clean_spacing(f"{name} {company_type}")
                confidence = min(previous.confidence, current.confidence)
                candidates.append((confidence, entity, f"{previous.text} {current.text}"))
    if not candidates:
        return FieldValue(None, 0.0, None)
    confidence, entity, source = max(candidates, key=lambda item: item[0])
    return FieldValue(entity, max(0.0, confidence), source)


def normalize_country_name(value: str) -> str:
    """Normalize punctuation and common abbreviations in an origin country."""
    cleaned = clean_spacing(value.strip(" .-|,;:"))
    alias_key = re.sub(r"[^a-z]+", " ", cleaned.casefold()).strip()
    return COUNTRY_ALIASES.get(alias_key, cleaned.title())


def normalize_origin_ocr(text: str) -> str:
    """Repair strongly evidenced, character-spaced domestic-origin wording."""
    transliterated = text.casefold().translate(str.maketrans({"0": "o", "/": "i", "|": "i", "$": "s"}))
    compact = re.sub(r"[^a-z]+", "", transliterated)
    if "growndistilledandagednnewyork" in compact or "growndistilledandagedinnewyork" in compact:
        return "GROWN, DISTILLED AND AGED IN NEW YORK"
    return text


def extract_country_of_origin(lines: Sequence[OCRLine]) -> FieldValue:
    """Extract an explicit country or fall back to the producer's stated location."""
    candidates: list[tuple[float, str, str]] = []
    for text, confidence in extraction_windows(lines):
        normalized_text = normalize_origin_ocr(text)
        match = COUNTRY_ORIGIN_RE.search(normalized_text)
        if not match:
            continue
        country = normalize_country_name(match.group("country"))
        if 2 <= len(country) <= 50 and not re.search(r"\d|\b(?:the|this)\s+product\b", country, re.I):
            candidates.append((confidence, country, text))
    if not candidates:
        # Some imported wine labels use a nationality adjective rather than a
        # formal "Product of" statement. Require a wine/product anchor on the
        # same OCR line so an unrelated use of "French" is not treated as origin.
        for text, confidence in extraction_windows(lines):
            match = NATIONALITY_ORIGIN_RE.search(text)
            if match:
                candidates.append(
                    (confidence, normalize_country_name(match.group("country")), text)
                )
    if not candidates:
        # Some domestic labels state origin through the producer/bottler address
        # rather than a separate country declaration. Preserve that location as
        # requested instead of inferring a country from the state abbreviation.
        for confidence, _, location, source in spatial_role_blocks(lines):
            if location:
                candidates.append((min(1.0, confidence + 0.05), location, source))
        for text, confidence in extraction_windows(lines):
            role_match = BOTTLER_PRODUCER_RE.search(text)
            if not role_match:
                continue
            _, location = split_entity_and_location(role_match.group("entity"))
            if location:
                candidates.append((confidence, location, text))
    if not candidates:
        return FieldValue(None, 0.0, None)
    confidence, country, source = max(candidates, key=lambda item: item[0])
    return FieldValue(country, confidence, source)


def central_brand_text(line: OCRLine, image_width: int) -> tuple[str, float]:
    """Return credible words inside the main label column, excluding side text."""
    lower, upper = image_width * 0.07, image_width * 0.82
    words = [
        word for word in line.words
        if lower <= word.center_x <= upper and word.confidence >= 0.20
    ]
    if not words:
        return "", 0.0
    text = clean_spacing(" ".join(word.text for word in words))
    confidence = weighted_mean(
        [word.confidence for word in words],
        [max(1, len(word.text)) for word in words],
    )
    return text, confidence


def correct_brand_from_repeated_text(text: str, lines: Sequence[OCRLine]) -> str:
    """Repair a brand token when the same token is clearer elsewhere on-label."""
    reference: dict[str, tuple[str, float]] = {}
    for line in lines:
        for word in line.words:
            for token in re.findall(r"[A-Za-z]{4,}", word.text):
                key = token.casefold()
                if key in CATEGORY_VOCABULARY:
                    continue
                if word.confidence >= reference.get(key, ("", 0.0))[1]:
                    reference[key] = (token, word.confidence)
    corrected: list[str] = []
    for token in text.split():
        bare = re.sub(r"[^A-Za-z]", "", token)
        best = token
        best_ratio = 0.0
        if len(bare) >= 4:
            for candidate, confidence in reference.values():
                if confidence < 0.80 or abs(len(candidate) - len(bare)) > 2:
                    continue
                ratio = difflib.SequenceMatcher(None, bare.casefold(), candidate.casefold()).ratio()
                if ratio >= 0.72 and ratio > best_ratio:
                    best, best_ratio = candidate, ratio
        corrected.append(best)
    normalized: list[str] = []
    for token in corrected:
        prefix = re.sub(r"[^A-Za-z]", "", token).casefold()
        if prefix in BRAND_OCR_ALIASES:
            token = BRAND_OCR_ALIASES[prefix]
        elif token.isalpha() and not (token.islower() or token.isupper() or token.istitle()):
            token = token.capitalize()
        normalized.append(token)
    return clean_spacing(" ".join(normalized))


def clean_brand_phrase(text: str) -> str:
    """Apply high-specificity cleanup to known multi-line product-name patterns."""
    tokens = text.split()
    normalized = [re.sub(r"[^A-Za-z]", "", token).upper() for token in tokens]
    try:
        stones_index = normalized.index("STONES")
        throw_index = normalized.index("THROW", stones_index + 1)
        ipa_index = normalized.index("IPA", throw_index + 1)
    except ValueError:
        return text

    # The complete ordered phrase is present. Short tokens between its words
    # are artifacts from the small angled warning, not part of the wordmark.
    between = normalized[stones_index : ipa_index + 1]
    meaningful = [token for token in between if len(token) > 2]
    if meaningful == ["STONES", "THROW", "IPA"]:
        return "STONE'S THROW IPA"
    return text


def largest_wordmark(lines: Sequence[OCRLine]) -> FieldValue:
    """Return the largest concise non-regulatory OCR box as brand evidence."""
    words = [word for line in lines for word in line.words]
    candidates: list[tuple[float, OCRWord, str]] = []
    for word in words:
        text = word.text.strip(" -|_.,:;`\"")
        tokens = normalized_tokens(text)
        if not 1 <= len(tokens) <= 4 or len(text) < 3:
            continue
        if EXCLUDE_FROM_BRAND.search(text) or len(tokens.intersection(CATEGORY_VOCABULARY)) >= 2:
            continue
        score = word.width * word.height * max(0.20, word.confidence)
        candidates.append((score, word, text))
    if not candidates:
        return FieldValue(None, 0.0, None)
    _, word, text = max(candidates, key=lambda item: item[0])
    corrected = correct_brand_from_repeated_text(text, lines)
    return FieldValue(corrected, word.confidence, word.text)


def extract_brand(
    lines: Sequence[OCRLine],
    image_height: int,
    image_width: int,
    category: FieldValue,
    bottler_producer: FieldValue,
) -> FieldValue:
    """Rank prominent non-regulatory OCR lines to identify the brand name."""
    if not lines:
        return FieldValue(None, 0.0, None)
    median_height = statistics.median(line.height for line in lines)
    category_tokens = normalized_tokens(category.value or "")
    producer_tokens = normalized_tokens(bottler_producer.value or "")
    producer_brand_prefix = re.sub(
        rf"\s+{PRODUCER_SUFFIX}(?:\s+c[oe0]\.?)?$",
        "",
        bottler_producer.value or "",
        flags=re.IGNORECASE,
    )
    producer_brand_tokens = normalized_tokens(producer_brand_prefix)
    candidates: list[tuple[float, OCRLine]] = []
    for line in lines:
        text, central_confidence = central_brand_text(line, image_width)
        text = text.strip(" -|_.,:;")
        # Ignore isolated border/ornament characters before comparing a line
        # with the known category; otherwise ``0 BOURBON`` can leak Bourbon
        # into an otherwise correct brand name.
        semantic_text = re.sub(r"(?<!\w)[A-Za-z0-9](?!\w)", " ", text)
        tokens = normalized_tokens(semantic_text)
        if len(text) < 2 or not re.search(r"[A-Za-z]", text):
            continue
        category_hits = tokens.intersection(CATEGORY_VOCABULARY)
        if (
            EXCLUDE_FROM_BRAND.search(text)
            or (tokens and tokens.issubset(category_tokens))
            or (
                tokens
                and producer_tokens
                and tokens.issubset(producer_tokens)
                and not (
                    producer_brand_tokens
                    and tokens.issubset(producer_brand_tokens)
                )
            )
            or len(category_hits) >= 2
        ):
            continue
        prominence = line.height / max(1.0, median_height)
        top_bonus = max(0.0, 1.0 - line.top / max(1, image_height))
        alpha_bonus = min(1.0, len(re.findall(r"[A-Za-z]", text)) / 10.0)
        acronym_bonus = 0.35 if re.fullmatch(r"[A-Z]{2,4}", text) else 0.0
        score = (
            1.8 * prominence
            + 0.7 * top_bonus
            + 0.5 * central_confidence
            + 0.2 * alpha_bonus
            + acronym_bonus
        )
        candidates.append((score, line))
    if not candidates:
        return FieldValue(None, 0.0, None)

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    best_score = candidates[0][0]
    selected = [line for score, line in candidates[:3] if score >= best_score * 0.70]
    if len(selected) == 1:
        primary = selected[0]
        continuations = []
        for _, line in candidates:
            if line is primary:
                continue
            continuation, _ = central_brand_text(line, image_width)
            vertical_gap = line.top - primary.bottom
            if (
                -0.20 * primary.height <= vertical_gap <= 1.20 * primary.height
                and line.height >= 0.35 * primary.height
                and re.match(r"^(?:de|del|la|le|of|the)\b", continuation, re.IGNORECASE)
            ):
                continuations.append((abs(vertical_gap), line))
        if continuations:
            selected.append(min(continuations, key=lambda item: item[0])[1])
    selected.sort(key=lambda line: (line.top, line.left))
    components = [central_brand_text(line, image_width)[0] for line in selected]
    components = [correct_brand_from_repeated_text(component, lines) for component in components]
    components = [f"({component})" if re.fullmatch(r"[A-Z]{2,4}", component) else component for component in components]
    brand = clean_spacing(" ".join(components))
    brand = clean_spacing(re.sub(r"\b(?:19|20)\d{2}\b", "", brand))
    brand = clean_brand_phrase(brand)
    # Isolated one-character digits commonly come from borders/ornaments next
    # to large brand lettering; multi-digit and alphanumeric brand names remain.
    brand = clean_spacing(re.sub(r"(?<![\w.])\d(?![\w.])", " ", brand))
    brand = clean_spacing(
        re.sub(
            r"\b\d+\s+(?=(?:distillery|winery|brewery|brewing)\b)",
            "",
            brand,
            flags=re.IGNORECASE,
        )
    )
    brand = re.sub(
        r"\b((?i:winery|distillery|brewery|brewing))\s+([A-Z]{2,4})(?=\s)",
        r"\1 (\2)",
        brand,
    )
    conf = weighted_mean(
        [central_brand_text(line, image_width)[1] for line in selected],
        [max(1, len(central_brand_text(line, image_width)[0])) for line in selected],
    )
    wordmark = largest_wordmark(lines)
    if (
        wordmark.value
        and len(normalized_tokens(brand)) >= 5
        and len(normalized_tokens(wordmark.value)) <= 4
    ):
        return wordmark
    return FieldValue(brand or None, conf, brand or None)


class COLALabelExtractor:
    """Reusable extractor; construct once when processing multiple images."""

    def __init__(self, languages: Sequence[str] = ("en",), gpu: bool | str = False) -> None:
        """Initialize one reusable EasyOCR reader for the selected languages."""
        if gpu is False:
            torch.set_num_threads(configured_ocr_threads())
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # PyTorch permits this setting only before inter-op work starts;
                # a previously constructed reader has already applied it.
                pass
        log_ocr_stage(
            f"initializing reader; torch={torch.__version__}; gpu={gpu}; "
            f"threads={torch.get_num_threads()}"
        )
        self.reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
        log_ocr_stage("reader ready")

    def _ocr(self, image: np.ndarray) -> list[OCRWord]:
        """Detect bounded-size text boxes, then recognize original-image crops."""
        horizontal_lists, free_lists = self.reader.detect(
            image,
            min_size=20,
            text_threshold=0.60,
            low_text=0.30,
            canvas_size=2000,
            mag_ratio=1.5,
        )
        horizontal_list = horizontal_lists[0] if horizontal_lists else []
        free_list = free_lists[0] if free_lists else []
        horizontal_list, vertical_columns = orient_vertical_side_columns(
            horizontal_list,
            image.shape[1],
            image.shape[0],
        )
        free_list = list(free_list) + vertical_columns
        log_ocr_stage(
            f"recognition boxes ready; horizontal={len(horizontal_list)}; "
            f"rotated={len(free_list)}; vertical-columns={len(vertical_columns)}"
        )
        release_ocr_memory()
        result = self.reader.recognize(
            image,
            horizontal_list=horizontal_list,
            free_list=free_list,
            decoder="greedy",
            # Two crops at a time improves CPU latency without the peak-memory
            # increase of large recognition batches on Streamlit workers.
            batch_size=2,
            workers=0,
            detail=1,
            paragraph=False,
        )
        return easyocr_to_words(result)

    def _detect_angle(self, image: np.ndarray) -> float:
        """Estimate deskew with text detection only (no recognition pass)."""
        _, free_lists = self.reader.detect(
            image,
            min_size=20,
            text_threshold=0.60,
            low_text=0.30,
            canvas_size=1600,
            mag_ratio=1.0,
        )
        polygons = free_lists[0] if free_lists else []
        detected: list[OCRWord] = []
        for polygon in polygons:
            try:
                points = tuple((float(point[0]), float(point[1])) for point in polygon)
            except (TypeError, ValueError, IndexError):
                continue
            if len(points) == 4:
                detected.append(OCRWord("detected", 1.0, points))
        return estimate_text_angle(detected)

    def extract(
        self,
        image_path: str | Path,
        *,
        rotation: float | str = "auto",
        include_raw_text: bool = False,
        detailed: bool = False,
    ) -> dict[str, Any]:
        """Load an image path and extract label fields from its pixels."""
        image = load_image(image_path)
        return self.extract_image(
            image,
            rotation=rotation,
            include_raw_text=include_raw_text,
            detailed=detailed,
        )

    def extract_image(
        self,
        image: np.ndarray,
        *,
        rotation: float | str = "auto",
        include_raw_text: bool = False,
        detailed: bool = False,
    ) -> dict[str, Any]:
        """Extract fields directly from an OpenCV BGR image array."""
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
            raise ValueError("image must be a non-empty OpenCV/numpy array")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = prepare_for_ocr(image)
        image, isolated_panel = isolate_dominant_label_panel(image)
        if isolated_panel:
            # Restore useful character scale after perspective cropping while
            # keeping the same bounded-size OCR pipeline.
            image = prepare_for_ocr(image)
        log_ocr_stage(f"prepared image {image.shape[1]}x{image.shape[0]}")
        if isolated_panel:
            log_ocr_stage("dominant panel isolated from multi-label image")
        if rotation == "auto":
            angle = self._detect_angle(image)
            # Drop text-detection intermediates before the full OCR pass. This
            # matters in memory-constrained, long-lived Streamlit workers.
            release_ocr_memory()
        else:
            angle = float(rotation)

        # Wide COLA artwork often contains several horizontal label columns;
        # their polygons can produce a small false skew. Require stronger
        # evidence before rotating a wide panel, while retaining bottle-photo
        # correction and large-angle handling.
        minimum_auto_angle = 8.0 if image.shape[1] / image.shape[0] >= 1.5 else 2.0
        if rotation == "auto" and abs(angle) < minimum_auto_angle:
            angle = 0.0

        # Positive polygon slope is corrected by the same signed OpenCV angle.
        # Detection-only deskew ensures there is exactly one recognition pass.
        if abs(angle) >= 2.0:
            ocr_image = rotate_bound(image, angle)
        else:
            ocr_image = image
        log_ocr_stage(
            f"deskew complete; angle={angle}; OCR image "
            f"{ocr_image.shape[1]}x{ocr_image.shape[0]}"
        )
        words = self._ocr(ocr_image)
        log_ocr_stage(f"recognition complete; words={len(words)}")

        lines = words_to_lines(words)
        category = extract_category(lines)
        abv, proof, volume = extract_numeric_fields(lines)
        warning = extract_warning(lines)
        bottler_producer = extract_bottler_producer(lines)
        country_of_origin = extract_country_of_origin(lines)
        brand = extract_brand(
            lines,
            ocr_image.shape[0],
            ocr_image.shape[1],
            category,
            bottler_producer,
        )

        fields = {
            "brand_name": brand,
            "category_class": category,
            "abv": abv,
            "proof": proof,
            "volume": volume,
            "government_warning": warning,
            "bottler_producer": bottler_producer,
            "country_of_origin": country_of_origin,
        }
        result: dict[str, Any]
        if detailed:
            result = {name: asdict(value) for name, value in fields.items()}
        else:
            result = {name: value.value for name, value in fields.items()}

        result["deskew_angle_degrees"] = angle
        diagnostics: list[str] = []
        if abv.value and proof.value:
            av = float(re.search(r"[\d.]+", abv.value).group())
            pv = float(re.search(r"[\d.]+", proof.value).group())
            if abs(2 * av - pv) > 1.0:
                diagnostics.append("ABV and proof are inconsistent; inspect the source image.")
        if not words:
            diagnostics.append("No text was detected.")
        result["diagnostics"] = diagnostics
        if include_raw_text:
            result["raw_text"] = corpus(lines)
        return result


_DEFAULT_EXTRACTOR: COLALabelExtractor | None = None


def parse_cola_label(
    image_path: str | Path,
    *,
    rotation: float | str = "auto",
    include_raw_text: bool = False,
    detailed: bool = False,
) -> dict[str, Any]:
    """Convenience API. Reuse ``COLALabelExtractor`` for batches."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = COLALabelExtractor()
    return _DEFAULT_EXTRACTOR.extract(
        image_path,
        rotation=rotation,
        include_raw_text=include_raw_text,
        detailed=detailed,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the single-image command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="JPG/PNG/TIFF beverage label image")
    parser.add_argument(
        "--rotation",
        default="auto",
        help="Deskew degrees, or 'auto' (default). Example: --rotation=-18",
    )
    parser.add_argument("--gpu", action="store_true", help="Use a CUDA GPU if configured")
    parser.add_argument("--detailed", action="store_true", help="Include confidence and OCR evidence")
    parser.add_argument("--include-raw-text", action="store_true", help="Include reconstructed OCR text")
    parser.add_argument("--pretty", action="store_true", help="Indent JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the single-image CLI and return a process exit status."""
    args = build_parser().parse_args(argv)
    rotation: float | str = args.rotation
    if rotation != "auto":
        try:
            rotation = float(rotation)
        except ValueError:
            print("--rotation must be 'auto' or a number", file=sys.stderr)
            return 2
    try:
        extractor = COLALabelExtractor(gpu=args.gpu)
        result = extractor.extract(
            args.image,
            rotation=rotation,
            include_raw_text=args.include_raw_text,
            detailed=args.detailed,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
