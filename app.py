"""Streamlit interface for COLA label extraction, validation, and batching."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import streamlit as st

from label_validation import (
    FIELD_LABELS,
    PASS_THRESHOLD,
    validate_label_fields,
    validation_verdict,
)


SINGLE_IMAGE_TYPES = ["jpg", "jpeg", "jpe", "png", "tif", "tiff", "bmp", "webp"]
BATCH_TYPES = SINGLE_IMAGE_TYPES + ["pdf", "gif", "ppm", "pgm", "pbm", "jp2"]
APPLICATION_DIRECTORY = Path(__file__).resolve().parent
EXAMPLE_IMAGES = (
    ("Old-Tom-Distillery-Bourbon-Warning.png", "Old Tom Distillery Bourbon"),
    ("cascade-winery.jpg", "Cascade Winery"),
    ("imported-wine.png", "Imported Wine"),
)
RESULT_FIELDS = (
    "brand_name",
    "category_class",
    "abv",
    "proof",
    "volume",
    "bottler_producer",
    "country_of_origin",
    "government_warning",
)


def _build_single_extractor() -> Any:
    """Construct the shared OCR reader without invoking Streamlit from a worker."""
    from cola_label_extractor import COLALabelExtractor

    return COLALabelExtractor(gpu=False)


def single_extractor_signature() -> str:
    """Fingerprint OCR source so Streamlit cannot retain an outdated reader."""
    source = APPLICATION_DIRECTORY / "cola_label_extractor.py"
    return hashlib.sha256(source.read_bytes()).hexdigest()


@st.cache_resource(show_spinner=False)
def get_ocr_warmup(extractor_signature: str) -> Future[Any]:
    """Start OCR initialization in one background thread and cache its future."""
    del extractor_signature  # Its value participates in Streamlit's resource key.
    future: Future[Any] = Future()

    def initialize() -> None:
        """Publish either the initialized reader or its startup exception."""
        try:
            future.set_result(_build_single_extractor())
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(target=initialize, name="cola-ocr-warmup", daemon=True).start()
    return future


def get_single_extractor() -> Any:
    """Return the warmed shared reader, waiting only if initialization is unfinished."""
    return get_ocr_warmup(single_extractor_signature()).result()


@st.cache_resource(show_spinner="Loading batch extractor...")
def get_batch_extractor(extractor_signature: str) -> Any:
    """Create one document extractor per pipeline version to prevent stale OCR."""
    del extractor_signature  # Its value participates in Streamlit's resource key.
    from batch_label_extractor import TTBCOLADocumentExtractor

    return TTBCOLADocumentExtractor(
        cache_dir=Path(".cola_ocr_cache"),
        ocr_extractor=get_single_extractor,
    )


def flatten_detailed_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert detailed single-image fields into the batch-style value schema."""
    flattened: dict[str, Any] = {}
    for field in RESULT_FIELDS:
        detail = result.get(field)
        flattened[field] = detail.get("value") if isinstance(detail, dict) else detail
    return flattened


@st.cache_data(show_spinner=False, max_entries=32)
def extract_single_image(
    image_bytes: bytes, filename: str, extractor_signature: str
) -> dict[str, Any]:
    """Use the shared document cache and return the first page's OCR details."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {f".{item}" for item in SINGLE_IMAGE_TYPES}:
        suffix = ".png"
    with tempfile.TemporaryDirectory(prefix="cola_single_") as temporary_directory:
        input_path = Path(temporary_directory) / f"uploaded{suffix}"
        input_path.write_bytes(image_bytes)
        document = get_batch_extractor(extractor_signature).extract(
            input_path, include_pages=True, memory_safe=True
        )
    pages = document.get("pages", [])
    if not pages:
        raise ValueError("No readable label page was found in the uploaded image.")
    return pages[0]


def process_batch_uploads(
    uploaded_files: list[Any], entered_values: dict[str, str]
) -> list[dict[str, Any]]:
    """Extract and validate every uploaded document while retaining failures."""
    from batch_label_extractor import pipeline_signature
    from cola_label_extractor import release_ocr_memory

    extractor = get_batch_extractor(pipeline_signature())
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cola_streamlit_") as temporary_directory:
        temporary_path = Path(temporary_directory)
        for index, uploaded_file in enumerate(uploaded_files):
            safe_name = Path(uploaded_file.name).name
            input_path = temporary_path / f"{index:04d}_{safe_name}"
            input_path.write_bytes(uploaded_file.getvalue())
            try:
                result = extractor.extract(
                    input_path,
                    include_pages=False,
                    memory_safe=True,
                )
                result["source_file"] = safe_name
                validation_rows = validate_label_fields(entered_values, result)
                result["validation"] = {
                    **validation_verdict(validation_rows),
                    "results": validation_rows,
                }
            except Exception as exc:  # Keep other uploads moving after one failure.
                result = {
                    "source_file": safe_name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "review_required": True,
                    "validation": {
                        **validation_verdict([]),
                        "failed_labels": ["Extraction error"],
                        "results": [],
                    },
                }
            results.append(result)
            # Streamlit keeps this worker alive across every uploaded file;
            # return native inference workspaces before starting the next one.
            release_ocr_memory()
    return results


def batch_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build compact rows for displaying a batch without nested OCR evidence."""
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {"File": Path(str(result.get("source_file", ""))).name}
        for field in RESULT_FIELDS:
            row[FIELD_LABELS.get(field, field.replace("_", " ").title())] = result.get(field)
        validation = result.get("validation", {})
        row["Validation Verdict"] = validation.get("verdict", "FAIL")
        row["Compared Labels"] = validation.get("compared_labels", 0)
        row["Review Required"] = result.get("review_required")
        row["Error"] = result.get("error")
        rows.append(row)
    return rows


def clear_application_state() -> None:
    """Clear form widgets, uploads, and the latest single and batch results."""
    keys = {
        "expected_brand_name",
        "expected_category_class",
        "expected_alcohol_content",
        "expected_abv",
        "expected_volume",
        "expected_bottler_producer",
        "expected_country_of_origin",
        "expected_government_warning",
        "single_upload",
        "batch_uploads",
        "selected_example",
        "single_validation",
        "batch_results",
    }
    for key in keys:
        st.session_state.pop(key, None)


def select_example_image(filename: str) -> None:
    """Select a bundled example and discard the previous single-image state."""
    st.session_state["selected_example"] = filename
    st.session_state.pop("single_upload", None)
    st.session_state.pop("single_validation", None)


def render_example_gallery() -> None:
    """Display selectable bundled labels in a three-column gallery."""
    st.subheader("Example COLA beverage labels")
    st.caption("Select an example below, or upload your own image.")
    columns = st.columns(len(EXAMPLE_IMAGES))
    selected = st.session_state.get("selected_example")
    for column, (filename, display_name) in zip(columns, EXAMPLE_IMAGES):
        image_path = APPLICATION_DIRECTORY / "images" / filename
        with column:
            if image_path.is_file():
                st.image(str(image_path), caption=filename, width="stretch")
                button_label = "Selected" if selected == filename else "Select example"
                st.button(
                    button_label,
                    key=f"select_example_{filename}",
                    type="primary" if selected == filename else "secondary",
                    width="stretch",
                    on_click=select_example_image,
                    args=(filename,),
                )
                st.caption(display_name)
            else:
                st.error(f"Example image is unavailable: {filename}")


def render_sidebar_form() -> tuple[dict[str, str], bool, bool]:
    """Render validation inputs and return values plus both submit actions."""
    with st.sidebar:
        st.header("Label validation")
        st.caption("Enter the expected text printed on the uploaded label.")
        with st.form("cola_validation_form"):
            entered = {
                "brand_name": st.text_input("Brand Name", key="expected_brand_name"),
                "category_class": st.text_input(
                    "Category / Class / Type", key="expected_category_class"
                ),
                "alcohol_content": st.text_input(
                    "Alcohol Content",
                    help="Enter a printed percentage or proof statement, such as 45% or 90 proof.",
                    key="expected_alcohol_content",
                ),
                "abv": st.text_input("ABV", help="Example: 13.5%", key="expected_abv"),
                "volume": st.text_input(
                    "Net contents",
                    help="Example: 750 ml or 12 fl oz",
                    key="expected_volume",
                ),
                "bottler_producer": st.text_input(
                    "Producer / Bottler", key="expected_bottler_producer"
                ),
                "country_of_origin": st.text_input(
                    "Country of Origin", key="expected_country_of_origin"
                ),
                "government_warning": st.text_area(
                    "Government Warning", height=160, key="expected_government_warning"
                ),
            }
            validate_clicked = st.form_submit_button(
                "Label validation", type="primary", width="stretch"
            )
            batch_clicked = st.form_submit_button(
                "Batch processing", type="primary", width="stretch"
            )
            st.form_submit_button(
                "Clear",
                width="stretch",
                on_click=clear_application_state,
                help="Clear form fields, uploads, and the latest validation results.",
            )
    return entered, validate_clicked, batch_clicked


def render_verdict(verdict: dict[str, Any], subject: str = "Validation") -> None:
    """Display a prominent PASS or FAIL banner and its threshold details."""
    compared = int(verdict.get("compared_labels", 0))
    threshold = float(verdict.get("threshold", PASS_THRESHOLD))
    failed = verdict.get("failed_labels", [])
    message = (
        f"{subject}: {verdict.get('verdict', 'FAIL')} - {compared} comparable label(s); "
        f"every similarity must be at least {threshold:.0%}."
    )
    if verdict.get("verdict") == "PASS":
        st.success(message)
    else:
        st.error(message)
        if failed:
            st.caption("Below threshold: " + ", ".join(str(label) for label in failed))


def main() -> None:
    """Render the application and dispatch single or batch extraction actions."""
    st.set_page_config(page_title="TTB COLA Label Validator", page_icon="🏷️", layout="wide")
    # Warm the expensive OCR engine while the user reviews and fills in the form.
    # Single and batch validation both reuse this same reader.
    ocr_warmup = get_ocr_warmup(single_extractor_signature())
    st.title("TTB COLA Beverage Label Validator")
    st.write(
        "Upload one label for field-by-field validation, or upload several files "
        "for batch extraction. OCR may take tens of seconds for uncached images."
    )
    st.caption(
        "PASS requires every comparable entered/extracted label to reach at least "
        f"{PASS_THRESHOLD:.0%} similarity."
    )
    if ocr_warmup.done() and ocr_warmup.exception() is None:
        st.caption("OCR engine ready.")
    else:
        st.caption("OCR engine is warming up while you prepare the form.")
    render_example_gallery()

    entered, validate_clicked, batch_clicked = render_sidebar_form()

    single_upload = st.file_uploader(
        "Upload a COLA beverage label image",
        type=SINGLE_IMAGE_TYPES,
        accept_multiple_files=False,
        key="single_upload",
    )
    source_name: str | None = None
    source_bytes: bytes | None = None
    if single_upload is not None:
        source_name = single_upload.name
        source_bytes = single_upload.getvalue()
    elif st.session_state.get("selected_example"):
        example_name = Path(str(st.session_state["selected_example"])).name
        example_path = APPLICATION_DIRECTORY / "images" / example_name
        if example_path.is_file():
            source_name = example_name
            source_bytes = example_path.read_bytes()

    if source_bytes is not None:
        st.image(source_bytes, caption=source_name, width="stretch")
        if single_upload is None:
            st.success(f"Selected example: {source_name}")
    else:
        st.info("Select an example above or upload a label image to preview and validate it.")

    st.divider()
    batch_uploads = st.file_uploader(
        "Batch files",
        type=BATCH_TYPES,
        accept_multiple_files=True,
        key="batch_uploads",
        help="Select multiple images, PDFs, or multi-frame TIFF files.",
    )

    if validate_clicked:
        if source_bytes is None:
            st.error("Select an example or upload an image before clicking Label validation.")
        else:
            try:
                with st.spinner("Extracting and validating label text..."):
                    from batch_label_extractor import pipeline_signature

                    detailed = extract_single_image(
                        source_bytes, source_name or "uploaded.png", pipeline_signature()
                    )
                    extracted = flatten_detailed_result(detailed)
                    validation_rows = validate_label_fields(entered, extracted)
                    st.session_state["single_validation"] = {
                        "source_file": source_name,
                        "rows": validation_rows,
                        "verdict": validation_verdict(validation_rows),
                        "details": detailed,
                    }
            except Exception as exc:
                st.exception(exc)

    if st.session_state.get("single_validation"):
        single_result = st.session_state["single_validation"]
        st.subheader("Validation results")
        st.caption(f"Image: {single_result['source_file']}")
        render_verdict(single_result["verdict"])
        st.dataframe(single_result["rows"], width="stretch", hide_index=True)
        with st.expander("Extracted OCR details"):
            st.json(single_result["details"])

    if batch_clicked:
        if not batch_uploads:
            st.error("Select one or more batch files before clicking Batch processing.")
        else:
            with st.spinner(f"Processing {len(batch_uploads)} file(s)..."):
                results = process_batch_uploads(list(batch_uploads), entered)
            st.session_state["batch_results"] = results

    if st.session_state.get("batch_results"):
        results = st.session_state["batch_results"]
        st.subheader("Batch extraction results")
        passed = sum(
            result.get("validation", {}).get("verdict") == "PASS" for result in results
        )
        failed = len(results) - passed
        st.info(f"Batch verdicts: {passed} PASS, {failed} FAIL")
        st.dataframe(batch_summary(results), width="stretch", hide_index=True)
        batch_json = json.dumps(results, ensure_ascii=False, indent=2)
        st.download_button(
            "Download batch JSON",
            data=batch_json,
            file_name="cola_batch_results.json",
            mime="application/json",
        )
        with st.expander("Full batch JSON"):
            st.json(results)


if __name__ == "__main__":
    main()
