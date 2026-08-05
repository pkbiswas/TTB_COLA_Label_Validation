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

## Install

Python 3.10+ is recommended. Create a virtual environment, activate it, then:

```powershell
python -m pip install -r requirements.txt
```

EasyOCR downloads its English model on first use. Later runs use the local
model cache. On CPU, a high-resolution image usually takes tens of seconds;
an angled image takes two passes because the first pass estimates deskew.

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
