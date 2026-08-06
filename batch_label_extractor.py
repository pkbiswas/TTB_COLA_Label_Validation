"""Document-level OCR for TTB COLA alcoholic-beverage label files.

Supported containers include ordinary raster images, multi-frame TIFF files,
and multi-page PDFs. Every page/panel is OCR'd independently, then the best
evidence for each requested field is merged at document level.

Examples:
    python batch_label_extractor.py label.jpg --pretty
    python batch_label_extractor.py cola.pdf --include-pages --pretty
    python batch_label_extractor.py ./downloads --recursive --output results.json

OCR is probabilistic. Use ``review_required``, per-field confidence, source text,
and page numbers when deciding whether a result may enter a production system.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

FIELD_NAMES = (
    "brand_name",
    "category_class",
    "abv",
    "proof",
    "volume",
    "government_warning",
    "bottler_producer",
    "country_of_origin",
)

RASTER_EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".png", ".tif", ".tiff", ".bmp",
    ".webp", ".gif", ".ppm", ".pgm", ".pbm", ".jp2",
}
DOCUMENT_EXTENSIONS = RASTER_EXTENSIONS | {".pdf"}


def pipeline_signature() -> str:
    """Fingerprint both code files so stale OCR cache entries cannot be reused."""
    digest = hashlib.sha256()
    for source in (Path(__file__), Path(__file__).with_name("cola_label_extractor.py")):
        digest.update(source.read_bytes())
    return digest.hexdigest()


class UnsupportedDocumentError(ValueError):
    """Raised when an input cannot be decoded as a supported document."""


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Apply camera orientation and convert a Pillow frame to OpenCV BGR."""
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_pdf(path: Path, dpi: int, max_pages: int) -> list[np.ndarray]:
    """Render PDF pages with PDFium; no system Poppler install is required."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise UnsupportedDocumentError(
            "PDF input requires pypdfium2; run: python -m pip install -r requirements.txt"
        ) from exc

    try:
        document = pdfium.PdfDocument(str(path))
    except Exception as exc:
        raise UnsupportedDocumentError(f"Cannot open PDF {path}: {exc}") from exc

    try:
        page_count = len(document)
        if page_count == 0:
            raise UnsupportedDocumentError(f"PDF has no pages: {path}")
        if page_count > max_pages:
            raise UnsupportedDocumentError(
                f"PDF has {page_count} pages; --max-pages is {max_pages}"
            )
        scale = dpi / 72.0
        pages: list[np.ndarray] = []
        for page_number in range(page_count):
            page = document[page_number]
            bitmap = page.render(scale=scale, rotation=0)
            pil_image = bitmap.to_pil()
            pages.append(pil_to_bgr(pil_image))
            bitmap.close()
            page.close()
        return pages
    finally:
        document.close()


def load_raster_frames(path: Path, max_pages: int) -> list[np.ndarray]:
    """Load a raster file, retaining all TIFF/GIF frames."""
    try:
        with Image.open(path) as image:
            frame_count = getattr(image, "n_frames", 1)
            if frame_count > max_pages:
                raise UnsupportedDocumentError(
                    f"Image has {frame_count} frames; --max-pages is {max_pages}"
                )
            return [pil_to_bgr(frame.copy()) for frame in ImageSequence.Iterator(image)]
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedDocumentError(f"Cannot decode image {path}: {exc}") from exc


def load_document_pages(path: str | Path, *, dpi: int = 240, max_pages: int = 20) -> list[np.ndarray]:
    """Return every document page/frame as a BGR numpy array."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return render_pdf(source, dpi=dpi, max_pages=max_pages)
    if suffix in RASTER_EXTENSIONS:
        return load_raster_frames(source, max_pages=max_pages)
    raise UnsupportedDocumentError(
        f"Unsupported extension '{suffix}'. Supported: {', '.join(sorted(DOCUMENT_EXTENSIONS))}"
    )


def normalize_candidate(value: str) -> str:
    """Normalize a field value for case-insensitive conflict comparison."""
    return re.sub(r"[^a-z0-9%]+", " ", value.casefold()).strip()


def candidate_score(field: str, detail: dict[str, Any]) -> float:
    """Score one page-level field candidate using confidence and plausibility."""
    value = detail.get("value")
    if not value:
        return -1.0
    confidence = float(detail.get("confidence") or 0.0)
    score = confidence
    if field == "government_warning":
        warning_words = {
            "government", "warning", "surgeon", "pregnancy", "defects",
            "consumption", "drive", "machinery", "health", "problems",
        }
        tokens = set(re.findall(r"[a-z]+", value.casefold()))
        score += 0.30 * len(tokens.intersection(warning_words)) / len(warning_words)
    elif field == "brand_name":
        word_count = len(value.split())
        if 1 <= word_count <= 8 and len(value) <= 80:
            score += 0.08
        if re.search(r"government|warning|imported|bottled|alcohol", value, re.I):
            score -= 0.40
    elif field in {"bottler_producer", "country_of_origin"}:
        if 2 <= len(value) <= 120:
            score += 0.05
    elif field in {"abv", "proof", "volume"}:
        if re.search(r"\d", value):
            score += 0.05
    return score


def merge_page_results(page_results: Sequence[dict[str, Any]], review_threshold: float) -> tuple[dict[str, Any], list[str]]:
    """Select fields independently; front and back labels need not share a page."""
    merged: dict[str, Any] = {}
    review_reasons: list[str] = []

    for field in FIELD_NAMES:
        candidates: list[dict[str, Any]] = []
        for page in page_results:
            detail = page.get(field, {})
            if detail.get("value"):
                candidate = dict(detail)
                candidate["page_number"] = page["page_number"]
                candidate["score"] = candidate_score(field, detail)
                candidates.append(candidate)

        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        if not candidates:
            merged[field] = {
                "value": None,
                "confidence": 0.0,
                "source_text": None,
                "page_number": None,
            }
            continue

        winner = candidates[0]
        winner.pop("score", None)
        merged[field] = winner
        if float(winner.get("confidence") or 0.0) < review_threshold:
            review_reasons.append(
                f"{field} confidence {winner.get('confidence', 0.0):.2f} is below {review_threshold:.2f}"
            )

        credible_values = {
            normalize_candidate(str(candidate["value"]))
            for candidate in candidates
            if float(candidate.get("confidence") or 0.0) >= review_threshold
        }
        if len(credible_values) > 1:
            review_reasons.append(f"conflicting {field} values were found on different pages")

    # Proof is optional and often absent from wine/malt labels, so it is not a
    # completeness error. The other missing fields are reported, not invented.
    for required_field in ("brand_name", "category_class", "abv", "volume", "bottler_producer"):
        if merged[required_field]["value"] is None:
            review_reasons.append(f"{required_field} was not found")

    warning = merged["government_warning"]
    abv = merged["abv"]["value"]
    if warning["value"] is None and abv:
        match = re.search(r"\d+(?:\.\d+)?", abv)
        if match and float(match.group()) >= 0.5:
            review_reasons.append("government_warning was not found for a beverage at or above 0.5% ABV")

    abv_value = merged["abv"]["value"]
    proof_value = merged["proof"]["value"]
    if abv_value and proof_value:
        av = float(re.search(r"\d+(?:\.\d+)?", abv_value).group())
        pv = float(re.search(r"\d+(?:\.\d+)?", proof_value).group())
        if abs(2.0 * av - pv) > 1.0:
            review_reasons.append("ABV and US proof are inconsistent")

    return merged, list(dict.fromkeys(review_reasons))


class TTBCOLADocumentExtractor:
    """Extract and merge label fields from complete COLA image documents."""

    def __init__(
        self,
        *,
        languages: Sequence[str] = ("en",),
        gpu: bool | str = False,
        dpi: int = 240,
        max_pages: int = 20,
        review_threshold: float = 0.55,
        cache_dir: str | Path | None = None,
        ocr_extractor: Any = None,
    ) -> None:
        """Configure document rendering, review policy, OCR, and exact caching."""
        if not 120 <= dpi <= 600:
            raise ValueError("dpi must be between 120 and 600")
        if not 1 <= max_pages <= 100:
            raise ValueError("max_pages must be between 1 and 100")
        if not 0.0 <= review_threshold <= 1.0:
            raise ValueError("review_threshold must be between 0 and 1")
        self.languages = tuple(languages)
        self.gpu = gpu
        self._ocr: Any = ocr_extractor
        self.dpi = dpi
        self.max_pages = max_pages
        self.review_threshold = review_threshold
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self._pipeline_signature = pipeline_signature()

    @property
    def ocr(self) -> Any:
        """Load EasyOCR only after an exact-input cache miss."""
        if callable(self._ocr):
            self._ocr = self._ocr()
        elif self._ocr is None:
            from cola_label_extractor import COLALabelExtractor

            self._ocr = COLALabelExtractor(languages=self.languages, gpu=self.gpu)
        return self._ocr

    def _cache_path(self, source: Path, *, memory_safe: bool) -> Path | None:
        """Build a content-addressed cache path for an input and all settings."""
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(self._pipeline_signature.encode("ascii"))
        settings = (
            f"dpi={self.dpi};max_pages={self.max_pages};review={self.review_threshold};"
            f"languages={','.join(self.languages)};gpu={self.gpu};"
            f"memory_safe={memory_safe}"
        )
        digest.update(settings.encode("utf-8"))
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return self.cache_dir / f"{digest.hexdigest()}.json"

    def extract(
        self,
        path: str | Path,
        *,
        include_pages: bool = False,
        memory_safe: bool = True,
    ) -> dict[str, Any]:
        """Extract every page, merge its fields, and return document-level JSON."""
        source = Path(path).expanduser().resolve()
        print(
            f"[COLA batch] source={source.name}; memory-safe={memory_safe}",
            file=sys.stderr,
            flush=True,
        )
        cache_path = self._cache_path(source, memory_safe=memory_safe)
        if cache_path and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                # Content-identical files may share an OCR cache entry, but the
                # result must always identify the path requested by this run.
                cached["source_file"] = str(source)
                cached["cache_hit"] = True
                if not include_pages:
                    cached.pop("pages", None)
                return cached
            except (OSError, json.JSONDecodeError, TypeError):
                # A truncated or edited cache is harmless; recompute it.
                pass

        images = load_document_pages(source, dpi=self.dpi, max_pages=self.max_pages)
        page_count = len(images)
        page_results: list[dict[str, Any]] = []
        for page_index in range(page_count):
            image = images[page_index]
            try:
                page = self.ocr.extract_image(
                    image,
                    rotation="auto",
                    detailed=True,
                    include_raw_text=True,
                    memory_safe=memory_safe,
                )
                page["page_number"] = page_index + 1
                page_results.append(page)
            finally:
                # A multi-page document must not retain every decoded raster or
                # native inference workspace until the last page completes.
                images[page_index] = None
                del image
                from cola_label_extractor import release_ocr_memory

                release_ocr_memory()

        merged, reasons = merge_page_results(page_results, self.review_threshold)
        result: dict[str, Any] = {
            "source_file": str(source),
            "page_count": page_count,
            **{field: merged[field]["value"] for field in FIELD_NAMES},
            "field_details": merged,
            "review_required": bool(reasons),
            "review_reasons": reasons,
            "cache_hit": False,
            "pages": page_results,
        }
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            temporary.replace(cache_path)
        returned = copy.deepcopy(result)
        if not include_pages:
            returned.pop("pages", None)
        return returned


def discover_inputs(inputs: Sequence[Path], recursive: bool) -> list[Path]:
    """Expand files and directories into a sorted, duplicate-free input list."""
    discovered: list[Path] = []
    for item in inputs:
        path = item.expanduser().resolve()
        if path.is_file():
            discovered.append(path)
        elif path.is_dir():
            iterator: Iterable[Path] = path.rglob("*") if recursive else path.glob("*")
            discovered.extend(p for p in iterator if p.is_file() and p.suffix.lower() in DOCUMENT_EXTENSIONS)
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")
    # Resolve duplicates while retaining deterministic ordering.
    return sorted(dict.fromkeys(discovered), key=lambda path: str(path).casefold())


def build_parser() -> argparse.ArgumentParser:
    """Construct the batch/document command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Image/PDF files or directories")
    parser.add_argument("--recursive", action="store_true", help="Search input directories recursively")
    parser.add_argument("--languages", default="en", help="EasyOCR languages, comma-separated")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA when correctly installed")
    parser.add_argument("--dpi", type=int, default=240, help="PDF render DPI (default: 240)")
    parser.add_argument("--max-pages", type=int, default=20, help="Safety limit per document")
    parser.add_argument("--review-threshold", type=float, default=0.55)
    parser.add_argument("--include-pages", action="store_true", help="Include raw page OCR and candidates")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cola_ocr_cache"),
        help="Exact-input OCR cache (default: .cola_ocr_cache)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable the OCR cache")
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON to this file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run batch extraction, write JSON output, and return an exit status."""
    args = build_parser().parse_args(argv)
    try:
        paths = discover_inputs(args.inputs, args.recursive)
        if not paths:
            raise FileNotFoundError("No supported documents were found")
        extractor = TTBCOLADocumentExtractor(
            languages=[part.strip() for part in args.languages.split(",") if part.strip()],
            gpu=args.gpu,
            dpi=args.dpi,
            max_pages=args.max_pages,
            review_threshold=args.review_threshold,
            cache_dir=None if args.no_cache else args.cache_dir,
        )
        results: list[dict[str, Any]] = []
        for path in paths:
            try:
                results.append(extractor.extract(path, include_pages=args.include_pages))
            except Exception as exc:  # keep a large batch moving
                results.append({
                    "source_file": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "review_required": True,
                })
        payload: Any = results[0] if len(results) == 1 else results
        text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
        if args.output:
            destination = args.output.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text + "\n", encoding="utf-8")
            print(f"Wrote {len(results)} result(s) to {destination}", file=sys.stderr)
        else:
            print(text)
        return 1 if any("error" in result for result in results) else 0
    except (FileNotFoundError, UnsupportedDocumentError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
