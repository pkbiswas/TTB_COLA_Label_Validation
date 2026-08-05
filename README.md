# COLA beverage-label OCR

**Live application:** [Test the TTB COLA Label Validator](https://ttb-cola-label-validation.streamlit.app/)

## Public deployment

This folder is a self-contained Git repository payload for the Streamlit label
validation application. Push the **contents of this folder** to the root of a
public Git repository. In Streamlit Community Cloud, create an app from that
repository, select `app.py` as the entry point, and deploy. The platform installs
the Python packages from `requirements.txt`; EasyOCR downloads its English model
the first time the application initializes.

The deployed public application is available at
[https://ttb-cola-label-validation.streamlit.app/](https://ttb-cola-label-validation.streamlit.app/).

To verify the deployment package locally before pushing it:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application does not require API keys or committed secrets. Its OCR cache is
created at runtime and intentionally excluded from Git.

This project extracts these fields from a photographed beverage label:

- brand name
- category/class
- alcohol by volume (ABV)
- proof
- net volume
- government warning
- bottler/producer
- country of origin

When an explicit country statement is absent, `country_of_origin` contains the
producer/bottler location printed with the role statement (for example,
`Grand Rapids, MI 49546`). The extractor does not infer a country from a state.

The output is JSON. Missing or insufficiently supported fields are returned as
`null`; the extractor does not silently invent a brand or warning.

## Extraction and validation approach

The extractor prepares each TTB COLA image with OpenCV and then uses EasyOCR to
recover text together with confidence scores and page positions. Field-specific
rules combine wording, layout, and regulatory context to identify the brand,
beverage class, ABV, proof, volume, producer/bottler, origin, and government
warning. Numeric candidates are checked for credible units and ranges, while
low-confidence, missing, or conflicting evidence is retained for review instead
of being silently invented.

### Handling difficult images

- **Size and color format:** Grayscale and transparent images are converted to
  three-channel images. Small labels are enlarged with cubic interpolation to
  improve small-print recognition; very large images are reduced with area
  interpolation to bound memory and processing time.
- **Angles and rotation:** A fast text-detection pass estimates the dominant
  label angle from OCR polygons using a confidence- and width-weighted median.
  The estimate is deliberately conservative for curved bottles. Images skewed
  by at least two degrees are rotated on an expanded white canvas so text is not
  cropped, followed by one full recognition pass. A manual `--rotation` value is
  available when automatic deskew is insufficient.
- **Low contrast and small text:** OCR runs at increased magnification and canvas
  resolution, with a lower secondary text-detection threshold for faint text.
  EasyOCR also retries low-contrast text regions with contrast adjustment. OCR
  confidence is carried into field scoring, and weak candidates are rejected or
  flagged instead of being treated as reliable values.
- **Glare, shadows, and curved labels:** There is no destructive glare-removal or
  inpainting step, global binarization, blanket sharpening, or denoising because
  these operations can alter regulatory characters and numbers. The enlarged OCR
  pass, faint-text threshold, spatial grouping, contextual field rules, and
  confidence checks can recover text that remains visible around a reflection.
  Text fully obscured by glare, severe shadow, curvature, blur, or cropping is
  reported as missing or low confidence and should be reviewed or re-photographed.

For multi-page COLA documents and batches, every page is processed independently
and the strongest supported value for each field is selected with its page and
source-text provenance. Validation normalizes capitalization, punctuation,
spacing, and measurement-unit variations before computing text similarity
against the values entered by the user. An image passes only when at least one
field can be compared and every comparable field scores at least 85%; otherwise
it fails. Missing entered or extracted values remain visible but are not included
in the threshold calculation.

## Assumptions and limitations

The implementation relies on the following assumptions:

- **Document scope:** Inputs are images or documents of alcoholic-beverage labels
  associated with the TTB COLA process. The parser is tuned to common U.S.
  beverage-label vocabulary and regulatory layout, not arbitrary product images.
- **Language and characters:** The application uses English OCR by default and
  assumes the target fields are printed with Latin characters. Other EasyOCR
  languages can be configured through the Python or command-line APIs, but the
  Streamlit form currently uses English.
- **Image evidence:** At least part of each requested field must be visible,
  legible, and present in the submitted front/back label material. Text fully
  hidden by glare, blur, curvature, shadows, cropping, or overprinting cannot be
  recovered reliably and is returned as missing or low confidence.
- **Image correction:** Automatic deskew relies on credible text polygons between
  approximately 2 and 40 degrees and applies a conservative correction. More
  extreme, vertical, or inconsistent orientations may require the CLI's manual
  `--rotation` option. The code intentionally avoids global sharpening,
  binarization, denoising, and synthetic glare removal that could alter printed
  characters or numbers.
- **OCR confidence:** EasyOCR confidence is treated as evidence quality, not a
  probability that a field is legally correct. Field candidates below 0.25 are
  generally excluded, and document-level values below the default 0.55 review
  threshold are flagged for review.
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
- **Alcohol measurements:** ABV must be in the plausible range 0.1–100%, and proof
  must be in the range 1–200. Missing proof is not calculated from ABV, and
  missing ABV is not calculated from proof. When both are printed, a difference
  greater than one proof unit from twice the ABV is flagged for review.
- **Volume:** A volume must include a recognizable unit such as ml, cl, L, or
  fluid ounces. Common standard ml sizes receive a ranking bonus when OCR
  produces several candidates, but nonstandard printed sizes are still allowed.
  Units are normalized for display but quantities are not converted.
- **Government warning:** A warning anchor must be detected before warning text is
  returned. The canonical U.S. warning is reconstructed only when at least 55%
  of its distinctive vocabulary is present; otherwise the supported OCR text is
  returned rather than inventing the missing language.
- **Multi-page documents:** Pages and panels are processed independently, and the
  strongest supported page is selected separately for each field. Front-label
  identity and back-label regulatory fields may therefore come from different
  pages. Credible conflicting page values trigger review instead of being merged.
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
  `proof`, it is compared with extracted proof. Otherwise it is compared with
  ABV, falling back to proof only when ABV is unavailable.
- **Batch validation:** The same sidebar reference values are applied to every
  file in a submitted batch. A heterogeneous batch that requires different
  expected values should be split into separate runs.
- **Runtime and caching:** Public deployment assumes CPU inference, sufficient
  memory, and network access for EasyOCR's initial model download. Exact-input
  extraction results may be cached using file contents, settings, and source-code
  fingerprints; changing any of these invalidates the cache.
- **Human review:** OCR and heuristic parsing are assistive. A PASS verdict is not
  TTB approval, a legal-compliance determination, or a substitute for reviewing
  the original label and the extractor's source text/confidence evidence.

## Install

Python 3.10+ is recommended. Create a virtual environment, activate it, then:

```powershell
python -m pip install -r requirements.txt
```

EasyOCR downloads its English model on first use. Later runs use the local
model cache. On CPU, a high-resolution image usually takes tens of seconds;
an angled image takes two passes because the first pass estimates deskew.
The dependency file selects CPU-only PyTorch builds. Text boxes are detected on
a bounded canvas, then recognized one at a time from the original image pixels;
native CPU thread pools are also limited to bound public-cloud peak memory.

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

For production ingestion, reject or queue for human review fields below your
chosen confidence threshold. Label glare, curvature, tiny type, and incomplete
front/back photography cannot always be resolved by OCR alone.

## Complete COLA documents and batches

Use `batch_label_extractor.py` for TTB downloads containing multiple label
panels/pages, PDFs, multi-frame TIFFs, or a directory of mixed files:

```powershell
python batch_label_extractor.py cola.pdf --include-pages --pretty
python batch_label_extractor.py .\cola-downloads --recursive --pretty --output results.json
```

The document extractor OCRs each page independently and selects the strongest
page for each field. Its output includes page provenance, confidence, conflict
checks, missing-field notices, and a `review_required` flag. Supported raster
formats include JPEG, PNG, TIFF, BMP, WebP, GIF, PPM/PGM/PBM, and JPEG 2000.

For speed, deskew uses text detection without a redundant recognition pass and
the batch entry point keeps one OCR model loaded. Exact-input results are cached
under `.cola_ocr_cache`; the cache key includes the file contents, settings, and
extractor source code, so changed inputs or code cannot reuse stale results. Use
`--no-cache` to force OCR or `--cache-dir PATH` to choose another cache location.
Decoded page arrays and native inference workspaces are released between pages
and batch files so long-running public-cloud workers do not accumulate memory.

## Streamlit validation application

The web application in `app.py` supports single-label validation and batch
processing. Its sidebar accepts expected values for Brand Name, Category / Class
/ Type, Alcohol Content, ABV, Volume, Producer / Bottler, Country of Origin, and
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
matches, mismatches, missing input, and missing OCR evidence. Alcohol Content
uses proof when the entered value contains the word `proof`; otherwise it uses
ABV. Each image receives an overall **PASS** only when at least one field has
both an entered and extracted value and every such field has a similarity score
of at least 85%. Missing inputs and missing OCR values are excluded from the
threshold calculation; an image with no comparable fields receives **FAIL**.

Select several files under **Batch files** and click **Batch processing** to run
the document/batch extractor. Each file is validated against the values currently
entered in the sidebar and receives its own PASS/FAIL verdict using the same 85%
rule. The visually prominent batch button helps distinguish this operation from
single-image validation. Results can be inspected in the application or
downloaded as UTF-8 JSON. OCR models and exact-input results are reused across
Streamlit reruns for faster interaction.

Click **Clear** to remove the current sidebar values, uploaded files, and latest
single and batch validation results.
