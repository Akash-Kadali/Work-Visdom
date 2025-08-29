# VISDOM — Pixels to Reports

An end-to-end, production-ready visual inspection pipeline that turns raw wafer/laser die images in Google Drive into structured, verified CSV reports — with one-click batch jobs, strict idempotency, and lossless crops. The system is built for unattended daily operation and manual spot-verification when needed.&#x20;

---

## Why this exists

Factory teams drop image batches into Drive. VISDOM:

1. Detects and crops 20 laser dies + the unique ID “code” per full image via a remote model.
2. Splits each laser crop into Body/Critical/Laser (lossless PNG).
3. Runs YOLO + CRNN OCR, merges results into a single CSV, and surfaces mismatches for review.   &#x20;

---

## Key capabilities

* **Drive→Drive automation**: No browser uploads; jobs read and write directly to configured Drive folders.&#x20;
* **Idempotent, “run-once” semantics** with differential scanning (new inputs only) and single-flight locks per job. &#x20;
* **Lossless crops** for split outputs (PNG with preserved ICC), zero JPEG re-compression artifacts.&#x20;
* **Consistent naming** across all stages (`<base>_laser_die_{i}.jpg`, `<base>_code.png|jpg`, `<base>_body.png`, `<base>_critical.png`, `<base>_laser.png`). &#x20;
* **Robust remote inference calls** with retries/backoff and TLS error handling for Drive I/O and model endpoints.  &#x20;
* **Operator UI** to trigger jobs, poll status (`/job_status`), and manually verify only the disagreements.  &#x20;

---

## System architecture

```
[Drive: Full Images]
        │
        ▼
[Crop 20 + Code (Modal endpoint via CROP_URL)]
        │ produces:
        │  • <base>_laser_die_1..20.(jpg|png)
        │  • <base>_code.(jpg|png)
        ▼
[Drive: Laser-20 Folder]      [Drive: Code Folder]
        │                              │
        │ (idempotent diff)            │ (idempotent diff)
        ▼                              ▼
[Split 20→3, lossless PNGs]   [YOLO OCR] + [CRNN OCR]
        │                          │          │
        │                          └──────────┘  (merge)
        ▼
[Body / Critical / Laser PNGs]      ▼
                              [ocr_results.csv in OUTPUT_OCR_FOLDER_ID]
```

* Crop job driver & upload logic: `backend/inference_jobs/yolo_crop.py` (Drive listing, Modal POST, upload ZIP/split).&#x20;
* Split geometry & PNG writing: `backend/app.py` (`_make_crops`, `_to_png_bytes`).&#x20;
* OCR jobs: `backend/inference_jobs/yolo_ocr.py` + `backend/inference_jobs/crnn_ocr.py`, then merged CSV. &#x20;
* Web/API: Flask app + templates + JS polling.    &#x20;

---

## Repository layout

```
backend/
  app.py                     # Flask app, routes, Drive helpers, Split geometry & PNG save
  automation.py              # Orchestrates OCR pipeline (wired by app.py)
  drive_utils.py             # Google Drive auth/helpers
  inference_jobs/
    yolo_crop.py            # Drive→Drive Crop-20 + Code using Modal endpoint
    yolo_ocr.py             # YOLO OCR client & CSV persistence
    crnn_ocr.py             # CRNN OCR client & CSV persistence
  modal_request.py           # HTTP clients for remote endpoints
frontend/
  templates/
    crop.html, ocr.html, verify.html, index.html (or index.txt), done.html
  static/
    js/main.js               # Button wiring + status polling, toasts
    css/{base.css,components.css,animations.css,style.css}
.env                         # All folder IDs + model endpoint URLs etc.
```

See cited files for implementation details.         &#x20;

---

## Job flows

### 1) Crop 20 + Code (Drive→Drive)

* Reads **`INPUT_IMAGE_FOLDER_ID`**.
* POSTs each image to **`CROP_URL`** (FastAPI on Modal), expects a ZIP with 20 laser crops + 1 code.
* Upload mode:

  * Split ZIP into Code folder and Laser-20 folder with enforced names, skipping already-present outputs, or
  * Upload ZIP as a single artifact (optional).
* Writes a best-effort sentinel `CROP20_DONE.txt` to the code folder when complete.&#x20;

**Trigger**: UI “Run Drive Job” or `POST /crop_drive`. When no new inputs exist (diff via basename match), it no-ops with a “Nothing to do” message. Single-flight lock prevents concurrent starts. &#x20;

### 2) Split Laser-20 → Body/Critical/Laser (lossless)

For each `…_laser_die_{i}.(jpg|png)` input:

* **Body**: right 70% of image → `<base>_body.png`
* **Critical**: left 30% & **top 20%** → `<base>_critical.png`
* **Laser**: left 30% & **bottom 80%** → `<base>_laser.png`
  Crops are saved as PNG with preserved ICC profile and configurable compress level (still lossless).&#x20;

**Trigger**: UI “Split Laser 20 → 3 Folders” or `POST /split_drive`. A **preflight diff** determines missing outputs; only those are processed; stale sentinels are removed to avoid “done” illusions.&#x20;

### 3) OCR (YOLO + CRNN) and Merge

* YOLO decodes character detections left→right with confidence thresholding, X-axis de-dupe, and a strict character set; emits `yolo_results.csv`.&#x20;
* CRNN predicts text strings, sanitized/padded to 6 chars; emits `crnn_results.csv`.&#x20;
* A pipeline merges both into **`ocr_results.csv`** in **`OUTPUT_OCR_FOLDER_ID`**, and the UI shows only disagreements by default.  &#x20;

**Trigger**: UI “Run Full Pipeline (YOLO + CRNN)” or `POST /run_pipeline`. Idempotent: if no new code crops since last CSV, the run is skipped. Single-flight lock. &#x20;

---

## Web & API surface

### Endpoints (Flask)

* `GET /` — Dashboard (links to Crop, OCR, Verify). The home page clearly advertises “More features underway”.&#x20;
* `POST /crop_drive` — Start Crop 20 + Code (Drive job). Idempotent + locked.&#x20;
* `POST /split_drive` — Start Split 20→3 (lossless PNGs). Idempotent + locked.&#x20;
* `POST /run_pipeline` — Run YOLO+CRNN OCR and write merged CSV. Idempotent + locked.&#x20;
* `GET /job_status?job=crop|split|ocr` — Returns `running|done|idle|stale` with previews. “stale” means new inputs exist.&#x20;
* `GET /verify_data?only_disagreements=1|0` — JSON rows with image proxy URLs and predictions; UI defaults to disagreements.&#x20;
* `GET /image/<file_id>` — Proxies private Drive images to the UI without changing sharing settings.&#x20;

### Operator UI

* **Crop page**: One-click server-side crop; explains file outputs.&#x20;
* **OCR page**: One-click pipeline + link to manual verification.&#x20;
* **Verify page**: Keyboard-friendly viewer that loads only disagreements first; can toggle to “Show all rows”.&#x20;
* **JS runtime**: Buttons wire to API, show toasts/spinners, and **poll `/job_status`** until completion; also enforces “run-once” behavior in the UI.&#x20;

---

## Configuration

All configuration is via `.env` (example keys shown from a sample environment).&#x20;

**Drive auth**

* `CREDENTIALS_FILE` — OAuth client secret JSON
* `TOKEN_FILE` — OAuth token cache

**Drive folders**

* `INPUT_IMAGE_FOLDER_ID` — Full input images (source for crop job)
* `OUTPUT_LASER_FOLDER_ID` — Laser-20 crops destination
* `OUTPUT_CODE_FOLDER_ID` — Code crops destination
* `OUTPUT_BODY_FOLDER_ID`, `OUTPUT_CRITICAL_FOLDER_ID`, `OUTPUT_LASER2_FOLDER_ID` — Split outputs (PNG)
* `OUTPUT_OCR_FOLDER_ID` — Where `ocr_results.csv` is written
* Optional alias: `INPUT_FOLDER_ID` (kept equal to `INPUT_IMAGE_FOLDER_ID`)

**Model endpoints**

* `CROP_URL` — Modal FastAPI endpoint that returns ZIP of 20 lasers + code
* `YOLO_URL`, `CRNN_URL` — Modal OCR endpoints used by the pipeline

**Idempotency / naming**

* `FINAL_CSV_NAME` (default `ocr_results.csv`)
* `CROP_SENTINEL_NAME`, `SPLIT_SENTINEL_NAME`, `OCR_SENTINEL_NAME`
* `OCR_FORCE` — force re-run even if final CSV exists

**PNG save (lossless)**

* `PNG_COMPRESS_LEVEL`, `PNG_OPTIMIZE`

---

## Installation & local run

> Requires Python 3.10+ and a Google Cloud project with Drive API enabled.

1. **Clone & deps**

```bash
python -m venv .venv && source .venv/bin/activate
pip install flask google-api-python-client google-auth google-auth-oauthlib pillow requests python-dotenv
```

2. **Create `.env`** with your folder IDs and Modal URLs (see “Configuration”).&#x20;

3. **Place credentials**

```
backend/
  client_secret.json   # OAuth client (matches CREDENTIALS_FILE)
  token.json           # Will be generated on first auth flow
```

4. **Run**

```bash
python -m backend.app
# opens at http://127.0.0.1:5000
```

The app serves the UI pages and API endpoints described above.&#x20;

---

## Using the system

### From the browser

* **Crop 20 + Code** → “Run Drive Job” on the **Crop** page.
* **Split 20→3** → “Split Laser 20 → 3 Folders” on the **Crop** page.
* **Run OCR** → “Run Full Pipeline (YOLO + CRNN)” on the **OCR** page.
* **Verify** → “Compare” page to step through only disagreements first.  &#x20;

### From the command line (examples)

```bash
# Crop (Drive job)
curl -X POST http://127.0.0.1:5000/crop_drive

# Split (Drive job)
curl -X POST http://127.0.0.1:5000/split_drive

# OCR pipeline
curl -X POST http://127.0.0.1:5000/run_pipeline

# Status (poll)
curl "http://127.0.0.1:5000/job_status?job=crop"
```

Each returns JSON with status text and (when applicable) a preview of pending items.&#x20;

---

## Implementation highlights

* **Geometry**: `_make_crops` enforces Body=right 70%, Critical=left 30% top 20%, Laser=left 30% bottom 80%.&#x20;
* **PNG writer**: `_to_png_bytes` preserves ICC profile and uses configurable compression/optimize flags — always lossless.&#x20;
* **OCR decode**: YOLO sorts by X-center, applies confidence and X-axis de-duplication (px threshold), maps class→char, and pads/truncates to fixed length.&#x20;
* **CSV policy**: Per-model CSVs are updated in place; duplicates are removed; merged CSV is the single source for verification. &#x20;
* **Resilience**: Download calls handle Google `HttpError` with exponential backoff and known TLS faults like `WRONG_VERSION_NUMBER`.  &#x20;
* **Locks + status**: Per-job `threading.Lock()` prevents double starts; `/job_status` computes **stale** by diffing inputs/outputs, not just by sentinels.&#x20;

---

## Operational guidance

### Idempotency & naming

* Crop job **skips** images whose `<base>_code.*` and all 20 `<base>_laser_die_{i}.*` already exist, unless forced via env on the worker side.&#x20;
* Split job runs only for inputs where any of `<base>_body.png|_critical.png|_laser.png` is missing.&#x20;
* OCR jobs keep/merge existing per-model CSV rows to avoid thrashing; force re-run with env flags when needed. &#x20;

### Security

* Drive files are served to the UI via an authenticated **proxy** endpoint; no public links needed. Keep `client_secret.json` and `token.json` secure.&#x20;
* Never commit `.env` or credentials.

### Deploying to production

* Run behind a WSGI server (e.g., gunicorn) and a reverse proxy (e.g., Nginx).
* Keep Modal endpoints behind appropriate auth (e.g., signed URLs or private network).
* Externalize logs to your platform (Cloud Logging, ELK).

---

## Troubleshooting

* **“⏳ already running”**: You hit the single-flight lock; poll `/job_status` to observe progress. &#x20;
* **“✅ Nothing to do”**: The diff found no new inputs. Confirm folder IDs and naming conventions.&#x20;
* **TLS/SSL errors from Drive** (e.g., `WRONG_VERSION_NUMBER`): The clients automatically retry with backoff; if persistent, check network/proxy.  &#x20;
* **Modal error messages** in crop/OCR: The HTTP client surfaces non-200 responses and raises with the upstream text prefix; inspect logs on the modal app. &#x20;

---

## Frontend notes

* **CSS**: A clean, professional dark theme with brand-aligned gradient, sticky nav, elevation, responsive layout, and accessible focus rings. &#x20;
* **Animations**: Subtle page/element transitions, accessible spinner, and toast notifications.&#x20;
* **JS app**: Centralized fetch/JSON helpers, toasts, button “working” states, and periodic status polling wired per job.&#x20;

---

## Roadmap

The landing page outlines near-term additions: live job progress, keyboard-first verification, confidence-aware prioritization, and drift alerts.&#x20;

---

## License & attribution

Internal use at Ayar Labs (visual defect observation and monitoring). Update license headers as appropriate for your deployment. Branding in templates references Ayar Labs.  &#x20;

---

## Appendix — Environment keys (quick reference)

```
# OAuth
CREDENTIALS_FILE=client_secret.json
TOKEN_FILE=token.json

# Drive I/O
INPUT_IMAGE_FOLDER_ID=...
INPUT_FOLDER_ID=...
OUTPUT_LASER_FOLDER_ID=...
OUTPUT_CODE_FOLDER_ID=...
OUTPUT_BODY_FOLDER_ID=...
OUTPUT_LASER2_FOLDER_ID=...
OUTPUT_CRITICAL_FOLDER_ID=...
OUTPUT_OCR_FOLDER_ID=...

# Model endpoints
CROP_URL=...
YOLO_URL=...
CRNN_URL=...

# CSV / Sentinels / Behavior
FINAL_CSV_NAME=ocr_results.csv
CROP_SENTINEL_NAME=CROP20_DONE.txt
SPLIT_SENTINEL_NAME=SPLIT_DONE.txt
OCR_SENTINEL_NAME=OCR_DONE.txt
OCR_FORCE=0

# PNG save (lossless)
PNG_COMPRESS_LEVEL=0
PNG_OPTIMIZE=0
```

See `.env` for a working example.&#x20;

---

**Built like a product, not a prototype.**
