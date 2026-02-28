"""Video model re-exports for the reorganized video/ package."""

from models.video_backbone import VideoCNNBackbone, ResBlock

__all__ = ["VideoCNNBackbone", "ResBlock"]
