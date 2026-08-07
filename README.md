# COLA beverage-label OCR

**Live application:** [Test the TTB COLA Label Validator](https://ttb-cola-label-validation.streamlit.app/)

## Public deployment

This folder is a self-contained Git repository payload for the Streamlit label
validation application. Push the **contents of this folder** to the root of a
public Git repository. In Streamlit Community Cloud, create an app from that
repository, select `app.py` as the entry point, and deploy. The application does
not require API keys or committed secrets. Dependencies are declared in
`requirements.txt`; local startup instructions appear below.

This project extracts these fields from a photographed beverage label:

- brand name
- category/class
- alcohol content
- alcohol by volume (ABV)
- net contents
- government warning
- bottler/producer
- country of origin

The output is JSON.

## Extraction and validation approach

The extractor prepares each TTB COLA image with OpenCV and then uses EasyOCR to
recover text together with confidence scores and page positions. Field-specific
rules combine wording, layout, and regulatory context to identify the brand,
beverage class, alcohol content, ABV, net contents, producer/bottler, country of
origin, and government warning. Numeric candidates are checked for credible
units and ranges, and each selected value retains confidence and supporting
source text for audit.

### Handling difficult images

- **Size and color format:** Grayscale and transparent images are converted to
  three-channel images. Small labels are enlarged with cubic interpolation to
  improve small-print recognition; very large images are reduced with area
  interpolation to bound memory and processing time.
- **Angles and rotation:** A fast text-detection pass estimates the dominant
  label angle from OCR polygons using a confidence- and width-weighted median.
  The estimate is deliberately conservative for curved bottles. Supported skew
  is corrected on an expanded white canvas so text is not cropped, followed by
  one full recognition pass. Portrait and bottle images use
  a two-degree correction threshold; wide multi-column panels require eight
  degrees to avoid false rotation from unrelated columns. A manual `--rotation`
  value is available when automatic deskew is insufficient.
- **Low contrast and small text:** OCR runs at increased magnification and canvas
  resolution, with a lower secondary text-detection threshold for faint text.
  EasyOCR also retries low-contrast text regions with contrast adjustment. OCR
  confidence is carried into field scoring.
- **Glare, shadows, and curved labels:** There is no destructive glare-removal or
  inpainting step, global binarization, blanket sharpening, or denoising because
  these operations can alter regulatory characters and numbers. The enlarged OCR
  pass, faint-text threshold, spatial grouping, contextual field rules, and
  confidence checks can recover text that remains visible around a reflection.
  Text fully obscured by glare, severe shadow, curvature, blur, or cropping may
  remain unreadable and require review or re-photographing.

For multi-page COLA documents and batches, every page is processed independently
and the strongest supported value for each field is selected with its page and
source-text provenance. Validation compares extracted and user-entered values
and returns per-field similarity scores. The resulting verdict follows the
policy defined under **Assumptions and limitations**.

## Assumptions and limitations

The implementation relies on the following assumptions:

- **Document scope:** Inputs are images or documents of alcoholic-beverage labels
  associated with the TTB COLA process. The parser is tuned to common U.S.
  beverage-label vocabulary and regulatory layout, not arbitrary product images.
- **Language and characters:** The application uses English OCR by default and
  assumes the target fields are printed with Latin characters. Other EasyOCR
  languages can be configured through the Python or command-line APIs, but the
  Streamlit form currently uses English.
- **Image evidence:** Each requested value must be visibly present in the supplied
  label material. The difficult-image behavior and corrective processing are
  described under **Handling difficult images** above.
- **OCR confidence:** EasyOCR confidence is treated as evidence quality, not a
  probability that a field is legally correct. Field candidates below 0.25 are
  generally excluded; the default document-level review threshold is 0.55.
- **Brand and class:** Brand text is assumed to be visually prominent and usually
  located in the central label region. Category/class extraction uses a finite,
  ordered vocabulary plus conservative corrections for observed OCR errors;
  an unlisted or unusually worded class may remain unclassified.
- **Producer/bottler:** A company is extracted only when an explicit role phrase
  such as “produced by,” “bottled by,” or “distilled by” supports it. An
  importer-only statement is not assumed to identify the producer or bottler.
- **Origin semantics:** An explicit “product of,” “made in,” or similar origin
  statement is preferred. If none exists, `country_of_origin` may contain the
  city/state/postal location attached to the producer/bottler statement, as
  requested by this project; the code does not infer a country from a U.S. state.
- **Alcohol measurements:** ABV must be in the plausible range 0.1–100%.
  Alcohol Content may be printed as ABV or in proof format; printed proof must
  be in the range 1–200. Missing measurements are not calculated. When both
  formats are printed, a difference greater than one proof unit from twice the
  ABV is considered inconsistent.
- **Net Contents:** A volume must include a recognizable unit such as ml, cl, L, or
  fluid ounces. Common standard ml sizes receive a ranking bonus when OCR
  produces several candidates, but nonstandard printed sizes are still allowed.
  Units are normalized for display but quantities are not converted.
- **Government warning:** A warning anchor must be detected before warning text is
  returned. The canonical U.S. warning is reconstructed only when at least 55%
  of its distinctive vocabulary is present; otherwise the supported OCR text is
  returned rather than inventing the missing language.
- **Multi-page documents:** Front-label identity and back-label regulatory fields
  may come from different pages. Credible conflicting values are not
  automatically reconciled.
- **Expected form values:** User-entered values are assumed to be the validation
  reference. Comparison normalizes case, accents, punctuation, whitespace, and
  common unit wording, then uses character-sequence similarity; it is not a
  semantic or legal equivalence test.
- **Validation verdict:** An image passes only if at least one field has both an
  entered and extracted value and every comparable field scores at least 85%.
  Missing entered or extracted values remain visible but do not enter the score;
  no comparable fields produces a failure. The table labels scores of at least
  98% as “Match” and at least 80% as “Close match.”
- **Alcohol Content field:** If the entered Alcohol Content contains the word
  `proof`, proof-formatted Alcohol Content is preferred and ABV is shown when
  that format is absent. Otherwise ABV is preferred, with proof-formatted
  Alcohol Content as the fallback. This keeps available alcohol evidence
  visible; different measurement representations may still produce a mismatch.
- **Batch validation:** The same sidebar reference values are applied to every
  file in a submitted batch. A heterogeneous batch that requires different
  expected values should be split into separate runs.
- **Runtime:** Public deployment assumes CPU inference, sufficient memory, and
  network access for EasyOCR's initial model download.
- **Human review:** OCR and heuristic parsing are assistive. A PASS verdict is not
  TTB approval, a legal-compliance determination, or a substitute for reviewing
  the original label and the extractor's source text/confidence evidence.

## Install

Python 3.10+ is recommended. Create a virtual environment, activate it, then:

```powershell
python -m pip install -r requirements.txt
```

The dependency file selects CPU-only PyTorch builds. Text boxes are detected on
a bounded canvas, recognized individually from the original pixels, and native
CPU thread pools are limited to control public-cloud peak memory.

## Run

```powershell
python cola_label_extractor.py label.png --pretty
```

For auditable output with per-field confidence, source OCR, and all OCR text:

```powershell
python cola_label_extractor.py label.png --pretty --detailed --include-raw-text
```

Automatic deskew is enabled. If a difficult image needs manual correction:

```powershell
python cola_label_extractor.py label.png --rotation=-18 --pretty
```

Positive and negative values follow OpenCV's rotation convention. Use `--gpu`
only when PyTorch/CUDA is correctly installed.

## Python API

```python
from cola_label_extractor import COLALabelExtractor

# Reuse one instance for a batch so the OCR model is loaded only once.
extractor = COLALabelExtractor(gpu=False)

result = extractor.extract(
    "label.png",
    detailed=True,
    include_raw_text=True,
)
print(result["brand_name"]["value"])
print(result["bottler_producer"]["value"])
print(result["country_of_origin"]["value"])
```

## Complete COLA documents and batches

Use `batch_label_extractor.py` for TTB downloads containing multiple label
panels/pages, PDFs, multi-frame TIFFs, or a directory of mixed files:

```powershell
python batch_label_extractor.py cola.pdf --include-pages --pretty
python batch_label_extractor.py .\cola-downloads --recursive --pretty --output results.json
```

Its output includes page provenance and per-field confidence. Supported raster
formats include JPEG, PNG, TIFF, BMP, WebP, GIF, PPM/PGM/PBM, and JPEG 2000.

## Error handling

Errors and uncertain evidence are handled at the narrowest practical scope so
one problem does not unnecessarily stop an entire validation run:

- **Input checks:** Missing paths, unsupported formats, unreadable or corrupt
  files, invalid page/frame data, and invalid extraction settings produce clear
  error messages instead of unhandled tracebacks where the application can
  recover.
- **Uncertain extraction:** A field without sufficient visible support is
  returned as `null`; the extractor does not fabricate missing label text.
  Low-confidence values, conflicting page evidence, inconsistent alcohol
  measurements, and missing required fields are recorded in diagnostics and
  review reasons. The
  detailed batch result exposes these conditions through `review_required`.
- **Single-image isolation:** The Streamlit application catches extraction and
  validation exceptions, displays the exception for the active image, and
  leaves the form available for another upload or retry.
- **Batch isolation:** A failed file receives its own error result and FAIL
  verdict while later files continue processing. Successfully completed items
  remain available in the displayed and downloadable batch output.
- **Cache recovery:** Missing, truncated, or invalid cached results are ignored
  and recomputed. Content-, settings-, and code-based cache keys prevent a valid
  result for different extraction inputs from being reused accidentally.
- **Resource cleanup:** Image pages and native inference workspaces are released
  after each item, including failure paths. Adaptive detector sizing and the
  batch guidance below reduce memory pressure on constrained cloud hosts.
- **Process-level limits:** An operating-system or container out-of-memory kill
  can terminate Python before application-level error handling runs. If that
  occurs, restart the application and retry with fewer or smaller files; the
  recommended Community Cloud batch sizes appear below.

## Validation performance optimizations

The Streamlit page delays heavyweight OCR imports so the form can render first,
then warms one EasyOCR reader in a background thread. Single and batch validation
reuse that reader instead of loading separate models. CPU recognition processes
two text crops at a time and uses at most two PyTorch threads; set the
`COLA_OCR_THREADS` environment variable to `1` for a more memory-constrained host.

Both validation modes share the content-addressed cache under `.cola_ocr_cache`.
The cache key includes file contents, extraction settings, and extractor source
code, so renaming a file can reuse valid OCR while changed inputs, settings, or
code cannot return stale results. Cache lookup occurs before model loading, and
Streamlit also retains recent single-image results in memory. Use `--no-cache`
with the batch command to force OCR or `--cache-dir PATH` to select another cache
location.

Batch files remain sequential to control peak memory. Decoded pages and native
inference workspaces are released between items, while detection and recognition
use bounded canvases and batches. Batch detection adaptively limits EasyOCR's
detector enlargement to 1.25x only when a moderately deskewed image would create
a detector canvas above roughly 2.8 million pixels. Other batch and single-image
validations retain the higher-resolution 1.5x detector. Recognition
continues to read the original prepared-image crops. This targets the observed
cloud memory spike without changing field rules, confidence checks, or normal
image accuracy. Consequently, a new image still incurs OCR time, while repeated
content can return immediately from the content-addressed cache.

## Streamlit validation application

The web application in `app.py` supports single-label validation and batch
processing. Its sidebar accepts expected values for Brand Name, Category / Class
/ Type, Alcohol Content, ABV, Net Contents, Producer / Bottler, Country of Origin, and
Government Warning. The top of the application provides selectable Old Tom
Distillery Bourbon, Cascade Winery, and Imported Wine example images for quick
testing. Selecting an example makes it the active single-image source; users can
instead upload their own image. The active image is displayed in the main body.

Install dependencies and start the application:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Click **Label validation** to run the single-image extractor and compare every
entered value with OCR evidence. Results distinguish exact matches, close
matches, mismatches, missing input, and missing OCR evidence. Field mapping and
PASS/FAIL rules are defined under **Assumptions and limitations**.

Select several files under **Batch files** and click **Batch processing** to run
the document/batch extractor. Each file is validated against the values currently
entered in the sidebar. The visually prominent batch button helps distinguish
this operation from single-image validation. The batch summary displays the
derived Alcohol Content value alongside ABV, including an ABV fallback when no
other alcohol-content statement is available. Detailed measurement evidence
remains available in the downloadable UTF-8 JSON.

The interface has no fixed file-count limit, but Community Cloud batches should
normally contain 3-5 typical images, 1-3 large, rotated, or multi-page inputs, or
at most about 10 small images. Each document is limited to 20 pages or frames.
Split larger submissions into multiple batches because available cloud memory
and CPU allocation can vary.

Click **Clear** to remove the current sidebar values, uploaded files, and latest
single and batch validation results.
