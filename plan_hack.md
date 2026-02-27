# Intel Robotic Welding Dataset Description

## Overview

- The dataset is organized into three roots:
- `good_weld`: labeled non-defect reference runs (`750` runs in `43` configuration folders).
- `defect_data_weld`: labeled defect runs (`1580` runs in `80` configuration folders).
- `test_data`: anonymized evaluation set (`90` samples named `sample_0001` ... `sample_0090`).
- Training-available labeled pool (`good_weld` + `defect_data_weld`) contains `2330` runs.
- Each run/sample is multimodal and typically includes:
- one sensor CSV, one FLAC audio file, one AVI video file, and an `images/` folder (usually 5 JPGs).
- In `defect_data_weld`, a small number of runs have missing still images (`0` JPG in some `images/` folders), but CSV/audio/video are present.

## Weld Configuration Metadata

- Weld setup/context is encoded in labeled folder names (for example: joint type such as `butt` or `plane_plate`, material tags such as `Fe410` or `BSK46`, and date-like tokens).
- Run IDs follow a pattern like `04-03-23-0010-11`, where the final two-digit suffix (`11`) is the defect/quality code.
- For fair hackathon evaluation, `test_data` has anonymized folder names (`sample_XXXX`) and neutral filenames (`sensor.csv`, `weld.flac`, `weld.avi`) to reduce label leakage.
- Ground-truth linkage for evaluation is kept separately in `test_data_ground_truth.csv`.

## Labels and Defect Definitions

| Code | Label | Practical definition | Train pool count (`good_weld`+`defect_data_weld`) |
|---|---|---|---:|
| `00` | `good_weld` | Acceptable weld (no target defect) | 750 |
| `01` | `excessive_penetration` | Over-penetration through joint root | 479 |
| `02` | `burn_through` | Severe over-penetration causing burn-through/hole | 317 |
| `06` | `overlap` | Weld metal overlap without proper fusion at toe/root region | 155 |
| `07` | `lack_of_fusion` | Incomplete fusion between weld metal and base material | 320 |
| `08` | `excessive_convexity` | Excessively convex weld bead profile | 159 |
| `11` | `crater_cracks` | Cracks near weld crater/termination zone | 150 |

## Folder Structure

```text
good_weld/
  <configuration_folder>/
    <run_id>/
      <run_id>.csv
      <run_id>.flac
      <run_id>.avi
      images/*.jpg

defect_data_weld/
  <defect_configuration_folder>/
    <run_id>/
      <run_id>.csv
      <run_id>.flac
      <run_id>.avi
      images/*.jpg

test_data/
  sample_0001/
    sensor.csv
    weld.flac
    weld.avi
    images/*.jpg
  ...
  sample_0090/
```

- Evaluation helper files:
- `test_data_manifest.csv`: file paths for each anonymized sample.
- `test_data_ground_truth.csv`: mapping from `sample_id` to true label (for evaluator use).

## Sensor Features (Number of Features)

- Original labeled CSV schema has `10` columns:
- `Date`, `Time`, `Part No`, `Pressure`, `CO2 Weld Flow`, `Feed`, `Primary Weld Current`, `Wire Consumed`, `Secondary Weld Voltage`, `Remarks`.
- Core process/sensor channels used for modeling are typically `6` numeric features:
- `Pressure`, `CO2 Weld Flow`, `Feed`, `Primary Weld Current`, `Wire Consumed`, `Secondary Weld Voltage`.
- `test_data/sensor.csv` intentionally removes `Part No` to prevent ID leakage, so evaluation CSVs have `9` columns.

## Concise and Informative Guideline

1. Use `good_weld` and `defect_data_weld` only for training and validation.
2. Split by run/group (not random rows) to avoid leakage across near-duplicate temporal segments.
3. Treat each run/sample as multimodal (`sensor.csv` + audio + video + images), then build binary (`defect` vs `good`) and multi-class (`defect type`) models.
4. Run final inference only on anonymized `test_data` and export predictions keyed by `sample_id`.
5. Keep `test_data_ground_truth.csv` strictly for organizer-side scoring or final offline evaluation after predictions are frozen.
6. Report both performance and confidence quality (for example: F1, Macro-F1, calibration/ECE) and include failure-case examples.

## Phase 1: Data preparation + dataset + dashboard + overall analysis + feature engineering

**Goal:** produce a clean, reproducible dataset + an analysis dashboard that explains what’s in the data and what signals might matter.

**What they must do**

* **Ingest the dataset**

  * Load video + audio + labels/metadata.
  * Validate files: missing, corrupt, mismatched IDs, inconsistent durations.
* **Define the unit of prediction**

  * Decide what one “sample” means (whole weld, fixed-length segment, windowed chunks, etc.).
  * Ensure labels align to that unit.
* **Create a reproducible split**

  * Train/validation/test split that avoids leakage (split by session/part/run if applicable).
  * Save split files so results are repeatable.
* **Preprocess and standardize**

  * Make audio/video consistent (sampling rate/FPS, resizing, normalization, trimming/padding policy).
  * Handle variable length (padding, cropping, pooling, sliding windows).
* **Feature engineering (optional, but if used it must be documented)**

  * Produce derived signals/features from audio/video/metadata (any representation is fine).
  * Keep a clear mapping from raw inputs → engineered inputs.
* **Dashboard (must show)**

  * Dataset overview: counts, durations, missing/corrupt stats.
  * Label distributions: defect vs non-defect, defect-type counts.
  * Representative examples: video preview + audio preview (waveform/spectrogram or equivalent).
  * Basic data quality indicators: class imbalance, outliers, noise, sync issues (if relevant).
  * Exportable reports: ability to save plots/tables or generate a summary.

**Phase 1 output package**

* `dataset/` or loader pipeline that can recreate the dataset
* split definition files
* dashboard app/notebook
* short “data card” summary (1 page) describing assumptions and preprocessing choices

---

## Phase 2: Defect detection (binary classification) with confidence

**Goal:** build a model that outputs **defect vs non-defect** plus a **confidence score** for each prediction.

**What they must do**

* **Train a binary classifier**

  * Input: audio/video (and any engineered features) per sample.
  * Output: probability/score for “defect”.
* **Produce confidence**

  * Define what confidence means (typically a calibrated probability).
  * Confidence must be reported per prediction.
* **Set a decision rule**

  * Thresholding policy to convert score → defect/non-defect.
  * Threshold must be fixed for test-time scoring (not adjusted after seeing test labels).
* **Evaluate on validation**

  * Report core binary metrics (listed below).
  * Show error breakdown (false positives/false negatives) and examples.
* **Create an inference pipeline**

  * Script that takes the test split and writes predictions in the required format.

**Phase 2 output package**

* trained model checkpoint(s)
* inference script (one command run)
* `predictions_binary.csv` (or combined file) with:

  * `sample_id`, `p_defect`, `pred_defect`, `confidence`
* evaluation report/plots in the dashboard

---

## Phase 3: Defect type classification (multi-class)

**Goal:** if a weld is defective, predict **which defect type**, with confidence.

**What they must do**

* **Train a defect-type classifier**

  * Input: same sample representation.
  * Output: defect type probabilities (or scores).
* **Define handling of non-defect samples**

  * Either:

    * classify defect type **only when defect is predicted/known**, OR
    * include “none” as a class.
  * Whichever they choose, it must match the evaluation spec and be consistent.
* **Report confidence for type**

  * Provide a confidence score for the chosen defect type (top-1 probability or calibrated).
* **Evaluate**

  * Report multi-class metrics (listed below), especially per-class results due to imbalance.
* **Integrate with Phase 2**

  * Final output should be coherent: non-defect → type is “none”; defect → type predicted.

**Phase 3 output package**

* model checkpoint(s)
* inference script producing:

  * `pred_defect_type`, `p_type_*` (optional), `type_confidence`
* evaluation report (per-type performance)

---

# Evaluation criteria (what you’ll score)

## A) Model metrics (primary)

### Submission CSV (required)

Teams must submit one CSV file with this exact schema:

```csv
sample_id,pred_label_code,p_defect
sample_0001,11,0.94
sample_0002,00,0.08
...
sample_0090,06,0.81
```

Submission rules:

* Exactly `90` rows (one row per sample in `test_data_manifest.csv`)
* `sample_id` must match exactly (`sample_0001` ... `sample_0090`), with no duplicates
* `pred_label_code` must be one of: `00`, `01`, `02`, `04`, `06`, `11`
* `p_defect` must be numeric in `[0,1]`

Scoring interpretation:

* Binary prediction is derived as: `pred_defect = (pred_label_code != "00")`
* Type prediction is the submitted `pred_label_code`

### 1) Defect vs non-defect (binary)

Use these as the core:

* **F1 (Defect as positive class)**
* **Precision / Recall (Defect)**
* **ROC-AUC**
* **PR-AUC**
* **Confusion matrix counts** (TP/FP/FN/TN)

### 2) Defect type (multi-class)

Use these:

* **Macro F1** (treats each defect type equally, good for imbalance)
* **Per-class Precision/Recall/F1**
* **Weighted F1** (secondary)

### 3) Confidence quality (recommended to include)

Because “confidence” that’s just vibes is worthless:

* **Calibration metric**: **ECE (Expected Calibration Error)** (binary at minimum)

**Suggested single final score (simple and fair):**

* `FinalScore = 0.6 * Binary_F1 + 0.4 * Type_MacroF1`
* Optional small penalty for bad confidence:

  * `FinalScore = FinalScore - 0.05 * ECE`

(You can tweak weights, but pick them once and don’t change mid-hackathon unless you enjoy riots.)

---

## B) Engineering & product quality (secondary)

### UI / Dashboard (clean and usable)

* Clear navigation, readable plots/tables, consistent labels
* Shows the required dataset stats + evaluation views
* Fast enough to use during a demo (no 5-minute refreshes)

### Clean code & reproducibility

* One-command run for training/inference (or documented steps)
* Clear folder structure, requirements/environment file
* No hardcoded paths, no mystery constants without comments
* Reproducible splits + fixed random seeds (where relevant)

### Presentation & explanation

* Clear statement of:

  * sample definition
  * preprocessing assumptions
  * model outputs and how confidence is computed
  * strengths/weaknesses and common failure cases
* Demo includes: dashboard + a few correctly/incorrectly predicted examples
