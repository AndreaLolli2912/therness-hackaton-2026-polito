import torch
import torch.nn as nn
from torchvision import models


class WindowVideoClassifier(nn.Module):
    """
    Sliding-window video classifier (no temporal RNN).

    For each window of N frames the MobileNetV3-Small backbone extracts
    spatial features, the features are mean-pooled across the window,
    and a small MLP head produces logits.

    This mirrors how the AudioCNN classifies fixed-size spectral chunks.
    """

    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()

        # 1. Spatial backbone (MobileNetV3-Small)
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        self.feature_extractor = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # MobileNetV3-Small feature dim
        self.feature_dim = 576

        # 2. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, pre_extracted=False):
        """
        Args:
            x: If pre_extracted=False: (batch, n_frames, 3, H, W)
               If pre_extracted=True:  (batch, n_frames, 576)
            pre_extracted: bool, whether x is already MobileNetV3 features
        Returns:
            logits: (batch, num_classes)
        """
        if pre_extracted:
            # x is [B, N, 576]
            if x.dim() == 2: # [B, 576] single frame
                x = x.unsqueeze(1)
            features = x
        else:
            if x.dim() == 4:  # single frame (batch, 3, H, W)
                x = x.unsqueeze(1)

            batch_size, n_frames, c, h, w = x.size()
            x = x.view(batch_size * n_frames, c, h, w)

            features = self.feature_extractor(x)
            features = self.avgpool(features)
            features = torch.flatten(features, 1) # (B*N, 576)
            features = features.view(batch_size, n_frames, -1) # (B, N, 576)

        # Mean-pool across the temporal window
        features = features.mean(dim=1)  # (B, 576)
        logits = self.classifier(features)
        return logits
