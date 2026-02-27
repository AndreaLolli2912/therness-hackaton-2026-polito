import torch
import torch.nn as nn
from torchvision import models

class VideoAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(VideoAutoencoder, self).__init__()
        
        # Encoder: Using MobileNetV3 Small as a backbone
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder_features = mobilenet.features
        
        # MobileNetV3 Small features output 576 channels for 7x7 if input is 224x224
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(576 * 7 * 7, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 576 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (576, 7, 7)),
            nn.ConvTranspose2d(576, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 224x224
            nn.Sigmoid() # Normalize output to [0, 1]
        )

    def forward(self, x):
        # x: [batch, 3, 224, 224]
        features = self.encoder_features(x)
        latent = self.fc_encode(self.flatten(features))
        
        reconstructed = self.fc_decode(latent)
        reconstructed = self.decoder(reconstructed)
        return reconstructed

    def get_anomaly_score(self, x):
        reconstructed = self.forward(x)
        # Using MSE as the anomaly score
        score = torch.mean((x - reconstructed) ** 2, dim=(1, 2, 3))
        return score
