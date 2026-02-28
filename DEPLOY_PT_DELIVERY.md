o# Deploy `.pt` Delivery Guide (Audio MIL)

This is the minimal handoff process to deliver one deployable model artifact and verify it end-to-end.

## 0) Where training happens and how `best_model.pt` is produced

Training entrypoint (from project root):

```bash
conda run -n therness_env python -m audio.run_audio --config configs/master_config.json
```

Where this is implemented:

- `audio/run_audio.py` builds loaders/model and calls `run_training_mil(...)` when `audio.training.sequence_mil.enabled=true`
- `audio/run_train_mil.py` runs epochs and validation

How best checkpoint is decided:

- after each epoch, validation computes `val_f1` (Macro F1)
- if current `val_f1` is higher than previous best, code saves:
  - `.../last_model.pt` (always)
  - `.../best_model.pt` (only on improvement)
  - `.../best_models/<timestamp>_epXXX_f1_..._vl_.../model.pt` (snapshot)
- threshold used for binary decision is also stored in checkpoint (`threshold` key)

Important folders in this repo:

- baseline run: `checkpoints/audio/`
- promoted run already present: `checkpoints/audio_confirm_run_07_topk0.05_gw0.25_lr0.0001/`

## 0.1) How to decide the best model and parameters (practical workflow)

Use this 3-level rule to avoid choosing by one noisy epoch:

1. **Within one run (one config):**
  - use `best_model.pt` from that run directory (already selected by best val Macro F1)

2. **Across different configs/runs:**
  - for each run, export deploy `.pt` and evaluate on the same held-out set
  - rank runs primarily by `F1_DEFECT` (or Macro F1 if you evaluate multiclass)
  - use `PRECISION_DEFECT` / `RECALL_DEFECT` as tie-breakers depending on your objective:
    - fewer false alarms -> prefer higher precision
    - fewer missed defects -> prefer higher recall

3. **Stability check before final pick:**
  - prefer configs whose top snapshots in `best_models/` have similar scores (stable), not one lucky spike
  - if two configs are very close, choose simpler/safer one (lower variance, cleaner confusion matrix)

Recommended minimal tuning dimensions for MIL in this repo:

- `lr`
- `sequence_mil.topk_ratio_pos`
- `sequence_mil.good_window_weight`
- optionally `sequence_mil.eval_pool_ratio` and `patience`

Suggested quick selection protocol:

1. create 3-6 configs changing only these parameters
2. train each config with the same `seed` and split policy
3. evaluate each exported deploy model with the same evaluation script
4. choose winner by metric priority (usually highest `F1_DEFECT` with acceptable precision)
5. freeze that run dir and export final `deploy_single_label.pt`

## 1) Export the deployable `.pt`

From project root:

```bash
conda run -n therness_env python -m audio.export_deploy_pt \
  --checkpoint checkpoints/audio/best_model.pt \
  --output checkpoints/audio/deploy_single_label.pt
```

Expected output (example):

- `Exported deploy model: checkpoints/audio/deploy_single_label.pt`
- `chunk_samples=... | defect_idx=... | eval_pool_ratio=... | threshold=...`

## 2) Quick single-file sanity test

Use notebook: `deploy_single_window_playground.ipynb`

Run cells in order:

1. Setup + config-driven parameters
2. Helper functions
3. Mode A (full audio -> one label)
4. Mode B (single window -> one label)

Expected:

- one output label dict for full audio
- one output label dict for single window

## 3) Optional batch sanity scan

In `deploy_single_window_playground.ipynb`, run the batch cells:

- batch over files + offsets
- save CSV/JSON

Generated files:

- `checkpoints/audio/deploy_window_scan_results.csv`
- `checkpoints/audio/deploy_window_scan_results.json`

## 4) Optional full-dataset evaluation script (binary)

If you want a quick aggregate check similar to what we ran:

```bash
/home/alolli/miniconda3/envs/therness_env/bin/python /tmp/eval_deploy_dataset.py
```

Example metrics obtained in this repo run:

- `TOTAL_FILES=1551`
- `TP=675 FP=377 TN=354 FN=145`
- `ACCURACY=0.663443`
- `PRECISION_DEFECT=0.641635`
- `RECALL_DEFECT=0.823171`
- `F1_DEFECT=0.721154`

## 5) What to send to your friend

Required:

- `checkpoints/audio/deploy_single_label.pt`
- `checkpoints/audio/config.json`

Recommended:

- this guide
- `deploy_single_window_playground.ipynb`
- exported batch report (`csv/json`) if available

## 6) Input/Output contract for the deploy model

Input:

- raw waveform tensor of shape `(channels, samples)`
- can be full audio or one exact window

Output dict:

- `label` -> `0` (`good_weld`) or `1` (`defect`)
- `p_defect` -> defect probability after model pooling logic

## 7) Reproducibility (recommended)

Before sharing, save artifact hash:

```bash
sha256sum checkpoints/audio/deploy_single_label.pt
```

Include the hash in your handoff message.
