"""Evaluation utilities for video models."""

import torch
from video.train import validate_epoch


def run_test(model, dataloader, criterion, device, checkpoint_path):
    """Load checkpoint and run full validation."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    return validate_epoch(model, dataloader, criterion, device)
