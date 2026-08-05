"""Comparison helpers for validating entered COLA fields against OCR output."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Mapping


FIELD_LABELS = {
    "brand_name": "Brand Name",
    "category_class": "Category / Class / Type",
    "alcohol_content": "Alcohol Content",
    "abv": "ABV",
    "volume": "Volume",
    "bottler_producer": "Producer / Bottler",
    "country_of_origin": "Country of Origin",
    "government_warning": "Government Warning",
}
PASS_THRESHOLD = 0.85


def normalize_for_comparison(value: str | None) -> str:
    """Normalize case, accents, punctuation, spacing, and measurement units."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.casefold().replace("’", "'")
    text = re.sub(r"\bmillilit(?:er|re)s?\b", "ml", text)
    text = re.sub(r"\blit(?:er|re)s?\b", "l", text)
    text = re.sub(r"\bfluid\s*ounces?\b|\bfl\.?\s*oz\.?​?\b", "fl oz", text)
    text = re.sub(r"\balc(?:ohol)?\.?\s*/?\s*(?:by\s*)?vol(?:ume)?\.?​?\b", "abv", text)
    text = re.sub(r"[^a-z0-9.%']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(entered: str, extracted: str) -> float:
    """Return a zero-to-one similarity score for two normalized values."""
    left = normalize_for_comparison(entered)
    right = normalize_for_comparison(extracted)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def select_alcohol_content(entered: str, extracted: Mapping[str, Any]) -> str | None:
    """Choose proof or ABV evidence according to the entered alcohol statement."""
    if re.search(r"\bproof\b", entered, re.IGNORECASE):
        return extracted.get("proof")
    return extracted.get("abv") or extracted.get("proof")


def validation_status(entered: str, extracted: str | None) -> tuple[str, float]:
    """Classify a comparison as match, close match, mismatch, or unavailable."""
    if not entered.strip():
        return "Not entered", 0.0
    if extracted is None or not str(extracted).strip():
        return "Not extracted", 0.0
    score = similarity(entered, str(extracted))
    if score >= 0.98:
        return "Match", score
    if score >= 0.80:
        return "Close match", score
    return "Mismatch", score


def validate_label_fields(
    entered_values: Mapping[str, str], extracted_values: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Compare all sidebar fields and return rows suitable for a UI table."""
    rows: list[dict[str, Any]] = []
    icons = {
        "Match": "✅",
        "Close match": "🟨",
        "Mismatch": "❌",
        "Not entered": "➖",
        "Not extracted": "⚠️",
    }
    for key, label in FIELD_LABELS.items():
        entered = entered_values.get(key, "")
        if key == "alcohol_content":
            extracted = select_alcohol_content(entered, extracted_values)
        else:
            extracted = extracted_values.get(key)
        status, score = validation_status(entered, extracted)
        rows.append(
            {
                "Label": label,
                "Entered value": entered or "—",
                "Extracted value": extracted or "—",
                "Result": f"{icons[status]} {status}",
                "Similarity": round(score, 3),
            }
        )
    return rows


def validation_verdict(
    rows: list[dict[str, Any]], threshold: float = PASS_THRESHOLD
) -> dict[str, Any]:
    """Return PASS only when every comparable label meets the threshold.

    A label is comparable when both its entered and extracted values exist.
    Missing inputs and missing OCR values remain in the table but do not enter
    the similarity set. With no comparable labels, the verdict is FAIL.
    """
    comparable = [
        row
        for row in rows
        if row.get("Entered value") not in (None, "", "—")
        and row.get("Extracted value") not in (None, "", "—")
        and float(row.get("Similarity", -1.0)) >= 0.0
    ]
    failed_labels = [
        str(row["Label"])
        for row in comparable
        if float(row.get("Similarity", 0.0)) < threshold
    ]
    passed = bool(comparable) and not failed_labels
    return {
        "verdict": "PASS" if passed else "FAIL",
        "threshold": threshold,
        "compared_labels": len(comparable),
        "failed_labels": failed_labels,
        "minimum_similarity": (
            min(float(row["Similarity"]) for row in comparable) if comparable else None
        ),
    }
