import torch
import torch.nn as nn
from torchvision import models

class StreamingVideoClassifier(nn.Module):
    def __init__(self, num_classes=7, hidden_size=128, pretrained=True):
        """
        Hybrid CNN-RNN architecture for real-time welding defect detection.
        Balances MobileNetV3 efficiency with GRU temporal awareness.

        7-class output: good_weld(00), excessive_penetration(01), burn_through(02),
                        overlap(06), lack_of_fusion(07), excessive_convexity(08),
                        crater_cracks(11).
        """
        super(StreamingVideoClassifier, self).__init__()
        
        # 1. Spatial Feature Extractor (MobileNetV3 Small)
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.feature_extractor = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # MobileNetV3 Small feature dimension is 576
        self.feature_dim = 576
        
        # 2. Temporal Aggregator (GRU)
        # batch_first=True expects [batch, seq_len, features]
        self.gru = nn.GRU(
            input_size=self.feature_dim, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, h=None):
        """
        Args:
            x: [batch, seq_len, 3, 224, 224] - A sequence of frames.
            h: [1, batch, hidden_size] - Optional hidden state for streaming.
        Returns:
            logits: [batch, num_classes] - Prediction based on the last frame's context.
            h: [1, batch, hidden_size] - Updated hidden state.
        """
        if x.dim() == 4: # Handle single frame input [batch, 3, 224, 224]
            x = x.unsqueeze(1)
            
        batch_size, seq_len, c, h_img, w_img = x.size()
        
        # Flatten batch and seq_len for backbone processing
        x = x.view(batch_size * seq_len, c, h_img, w_img)
        
        # Extract spatial features
        features = self.feature_extractor(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Reshape back to sequence format [batch, seq_len, feature_dim]
        features = features.view(batch_size, seq_len, -1)
        
        # Temporal pass
        # output: [batch, seq_len, hidden_size]
        output, h = self.gru(features, h)
        
        # We classify based on the "state" at the final frame of the sequence
        last_step_features = output[:, -1, :]
        logits = self.classifier(last_step_features)
        
        return logits, h

    @torch.no_grad()
    def predict_stream(self, frame, hidden_state=None):
        """
        Inference helper for real-time streaming (1 frame at a time).
        """
        self.eval()
        logits, next_h = self.forward(frame.unsqueeze(1), hidden_state)
        probs = torch.softmax(logits, dim=1)
        return probs, next_h
