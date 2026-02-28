"""Evaluation and submission CSV generation."""

import csv

import torch
from tqdm import tqdm

from audio.train import validate_epoch
from audio.run_train_mil import validate_epoch_mil, validate_epoch_mil_multiclass


def run_test(model, dataloader, criterion, device, checkpoint_path):
    """Evaluate a model on a test set using a saved checkpoint.

    Args:
        model: nn.Module (same architecture used during training).
        dataloader: test dataloader yielding (inputs, targets).
        criterion: loss function.
        device: torch device.
        checkpoint_path: path to the checkpoint file.

    Returns:
        dict with "loss", "predictions", "targets".
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return validate_epoch(model, dataloader, criterion, device)


def run_test_mil(
    model,
    dataloader,
    device,
    checkpoint_path,
    task="binary",
    defect_idx=0,
    good_idx=1,
    topk_ratio_pos=0.05,
    topk_ratio_neg=0.2,
    eval_pool_ratio=0.05,
    threshold=0.5,
    auto_threshold=False,
):
    """Evaluate MIL model on a file-level dataloader using saved checkpoint.

    Works for both binary and multiclass tasks.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if task == "binary":
        eval_threshold = float(checkpoint.get("threshold", threshold))
        return validate_epoch_mil(
            model=model,
            dataloader=dataloader,
            device=device,
            defect_idx=defect_idx,
            good_idx=good_idx,
            topk_ratio_pos=topk_ratio_pos,
            topk_ratio_neg=topk_ratio_neg,
            eval_pool_ratio=eval_pool_ratio,
            threshold=eval_threshold,
            auto_threshold=auto_threshold,
        )
    else:
        return validate_epoch_mil_multiclass(
            model=model,
            dataloader=dataloader,
            device=device,
            eval_pool_ratio=eval_pool_ratio,
        )


def generate_submission(
    model,
    dataloader,
    device,
    checkpoint_path,
    label_map,
    output_path="submission.csv",
):
    """Run inference and produce the hackathon submission CSV.

    Args:
        model: nn.Module (same architecture used during training).
        dataloader: test dataloader yielding (inputs, sample_ids) tuples.
            inputs: Tensor or dict. sample_ids: list/tuple of sample ID strings.
        device: torch device.
        checkpoint_path: path to the checkpoint file.
        label_map: dict mapping class index (int) to label code string,
            e.g. {0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11"}.
            The key for "good_weld" (code "00") is used to derive p_defect.
        output_path: path to write the submission CSV.

    Returns:
        list of dicts with "sample_id", "pred_label_code", "p_defect".
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Find which class index corresponds to "good_weld" (code "00")
    good_weld_idx = None
    for idx, code in label_map.items():
        if code == "00":
            good_weld_idx = idx
            break

    num_classes = len(label_map)
    rows = []

    with torch.no_grad():
        for inputs, sample_ids in tqdm(dataloader, desc="Inference", leave=False):
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)

            # Binary case: single output, use sigmoid
            if num_classes <= 2 and outputs.dim() == 2 and outputs.size(1) == 1:
                probs_defect = torch.sigmoid(outputs).squeeze(1).cpu()
                for i, sid in enumerate(sample_ids):
                    p_defect = probs_defect[i].item()
                    if p_defect >= 0.5:
                        # Pick the defect class (first non-"00" code)
                        pred_code = [c for c in label_map.values() if c != "00"][0]
                    else:
                        pred_code = "00"
                    rows.append({
                        "sample_id": sid,
                        "pred_label_code": pred_code,
                        "p_defect": round(p_defect, 4),
                    })

            # Multi-class case: softmax over classes
            else:
                probs = torch.softmax(outputs, dim=1).cpu()
                pred_indices = probs.argmax(dim=1)
                for i, sid in enumerate(sample_ids):
                    pred_idx = pred_indices[i].item()
                    pred_code = label_map[pred_idx]
                    if good_weld_idx is not None:
                        p_defect = 1.0 - probs[i, good_weld_idx].item()
                    else:
                        p_defect = float(pred_code != "00")
                    rows.append({
                        "sample_id": sid,
                        "pred_label_code": pred_code,
                        "p_defect": round(p_defect, 4),
                    })

    # Sort by sample_id for consistent output
    rows.sort(key=lambda r: r["sample_id"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "pred_label_code", "p_defect"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Submission saved to {output_path} ({len(rows)} rows)")
    return rows


def predict_chunk_probs(
    model,
    dataloader,
    device,
    checkpoint_path,
    defect_idx=0,
):
    """Return per-chunk defect probabilities.

    Args:
        model: nn.Module.
        dataloader: yields (inputs, sample_ids) or (inputs, _).
        device: torch device.
        checkpoint_path: model checkpoint.
        defect_idx: class index corresponding to "defect" in softmax output.

    Returns:
        list of dict rows: {"sample_id", "p_defect"}.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    rows = []
    running_idx = 0

    with torch.no_grad():
        for inputs, sample_ids in tqdm(dataloader, desc="Chunk inference", leave=False):
            if not isinstance(sample_ids, (list, tuple)):
                sample_ids = [f"chunk_{running_idx + i}" for i in range(len(inputs))]

            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)

            if outputs.dim() == 2 and outputs.size(1) == 1:
                probs_defect = torch.sigmoid(outputs).squeeze(1).cpu()
            else:
                probs = torch.softmax(outputs, dim=1).cpu()
                probs_defect = probs[:, defect_idx]

            for i, sid in enumerate(sample_ids):
                rows.append({
                    "sample_id": sid,
                    "p_defect": float(probs_defect[i].item()),
                })
            running_idx += len(sample_ids)

    return rows


def apply_live_trigger(
    probs,
    on_threshold=0.6,
    off_threshold=0.4,
    min_consecutive_on=3,
    min_consecutive_off=5,
):
    """Stateful live alarm logic over chunk probabilities.

    The alarm turns ON after `min_consecutive_on` high-probability chunks and turns
    OFF after `min_consecutive_off` low-probability chunks.
    """
    states = []
    alarm_on = False
    on_count = 0
    off_count = 0

    for p in probs:
        if not alarm_on:
            if p >= on_threshold:
                on_count += 1
            else:
                on_count = 0
            if on_count >= min_consecutive_on:
                alarm_on = True
                off_count = 0
        else:
            if p <= off_threshold:
                off_count += 1
            else:
                off_count = 0
            if off_count >= min_consecutive_off:
                alarm_on = False
                on_count = 0

        states.append(int(alarm_on))

    return states
