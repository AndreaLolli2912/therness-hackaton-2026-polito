"""
Export video and audio models to ONNX format with optional INT8 quantization.

Supports:
  - ONNX export for cross-platform inference (ONNX Runtime, TensorRT, etc.)
  - Static INT8 post-training quantization via onnxruntime.quantization
  - PyTorch dynamic quantization for CPU-only deployment

Usage:
    # Export video model to ONNX
    python export_onnx.py --model video \
        --checkpoint checkpoints/video_classifier.pth \
        --output checkpoints/video_classifier.onnx

    # Export + INT8 quantize
    python export_onnx.py --model video \
        --checkpoint checkpoints/video_classifier.pth \
        --output checkpoints/video_classifier.onnx \
        --quantize

    # Export audio model
    python export_onnx.py --model audio \
        --checkpoint checkpoints/audio/audio_model.pth \
        --output checkpoints/audio/audio_model.onnx
"""
import os
import argparse

import torch
import torch.nn as nn

from src.models.video_model import StreamingVideoClassifier


def _resolve_device(device_arg: str) -> torch.device:
    dev = str(device_arg).strip().lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested, but CUDA is not available")
        return torch.device("cuda")
    if dev == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_arg}. Use one of: auto, cuda, cpu")


# Inline AudioCNN to avoid circular imports
class AudioCNN(nn.Module):
    """3-block CNN for audio classification."""
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


class VideoModelSingleFrame(nn.Module):
    """
    Wrapper that exports the video model in single-frame mode.
    Input: (B, 3, 224, 224) — one frame
    Output: (B, num_classes) — class logits
    
    GRU hidden state is initialized to zeros internally.
    For streaming, the hidden state would be managed externally.
    """
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        # x: [B, 3, 224, 224] → add time dimension → [B, 1, 3, 224, 224]
        x = x.unsqueeze(1)
        logits, _ = self.model(x, h=None)
        return logits


def export_video_onnx(checkpoint, output, num_classes=7, hidden_size=128, device="auto"):
    """Export video model to ONNX for single-frame inference."""
    export_device = _resolve_device(device)
    model = StreamingVideoClassifier(
        num_classes=num_classes,
        hidden_size=hidden_size,
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location=export_device, weights_only=True)
    model.load_state_dict(state)
    model.to(export_device).eval()

    wrapper = VideoModelSingleFrame(model)
    wrapper.to(export_device).eval()

    # Dummy input: single frame
    dummy = torch.randn(1, 3, 224, 224, device=export_device)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, output,
        input_names=["frame"],
        output_names=["logits"],
        dynamic_axes={
            "frame": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported video model to {output}")
    print(f"  Export device: {export_device}")
    print(f"  File size: {os.path.getsize(output) / 1024 / 1024:.2f} MB")
    return output


def export_audio_onnx(checkpoint, output, num_classes=7, dropout=0.3, device="auto"):
    """Export audio model to ONNX."""
    export_device = _resolve_device(device)
    model = AudioCNN(num_classes=num_classes, dropout=dropout)
    state = torch.load(checkpoint, map_location=export_device, weights_only=True)
    model.load_state_dict(state)
    model.to(export_device).eval()

    # Dummy input: (B, 1, n_mels=40, T=24) for a 0.5s chunk
    dummy = torch.randn(1, 1, 40, 24, device=export_device)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    torch.onnx.export(
        model, dummy, output,
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch", 3: "time"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported audio model to {output}")
    print(f"  Export device: {export_device}")
    print(f"  File size: {os.path.getsize(output) / 1024 / 1024:.2f} MB")
    return output


def quantize_onnx(onnx_path):
    """Apply INT8 dynamic quantization to an ONNX model."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("WARNING: onnxruntime.quantization not available. "
              "Install with: pip install onnxruntime")
        return None

    base, ext = os.path.splitext(onnx_path)
    quantized_path = f"{base}_int8{ext}"

    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QInt8,
    )

    fp32_size = os.path.getsize(onnx_path) / 1024 / 1024
    int8_size = os.path.getsize(quantized_path) / 1024 / 1024
    print(f"Quantized → {quantized_path}")
    print(f"  FP32: {fp32_size:.2f} MB → INT8: {int8_size:.2f} MB "
          f"({int8_size/fp32_size*100:.0f}%)")
    return quantized_path


def quantize_pytorch(checkpoint, num_classes=7, hidden_size=128):
    """Apply PyTorch dynamic quantization (CPU only)."""
    model = StreamingVideoClassifier(
        num_classes=num_classes,
        hidden_size=hidden_size,
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.GRU}, dtype=torch.qint8
    )

    base, ext = os.path.splitext(checkpoint)
    quantized_path = f"{base}_int8{ext}"
    torch.save(quantized.state_dict(), quantized_path)
    print(f"PyTorch INT8 model saved to {quantized_path}")
    return quantized_path


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX + quantize")
    parser.add_argument("--model", choices=["video", "audio"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to .pth file")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--quantize", action="store_true",
                        help="Also produce INT8 quantized model")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Export device for loading and tracing: auto|cuda|cpu")
    args = parser.parse_args()

    if args.model == "video":
        onnx_path = export_video_onnx(
            args.checkpoint, args.output,
            args.num_classes, args.hidden_size, args.device
        )
    else:
        onnx_path = export_audio_onnx(
            args.checkpoint, args.output,
            args.num_classes, args.dropout, args.device
        )

    if args.quantize:
        quantize_onnx(onnx_path)
        if args.model == "video":
            quantize_pytorch(args.checkpoint, args.num_classes, args.hidden_size)


if __name__ == "__main__":
    main()
