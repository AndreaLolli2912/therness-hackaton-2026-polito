import torch
import torch.nn as nn
from .video_model import StreamingVideoClassifier
from .sensor_model import SensorClassifier

class MultimodalFusionClassifier(nn.Module):
    def __init__(self, num_classes=7, video_hidden=128, sensor_hidden=64, fusion_hidden=64):
        super(MultimodalFusionClassifier, self).__init__()
        
        # 1. Video Branch
        # We reuse the video model but we will extract the GRU output
        self.video_branch = StreamingVideoClassifier(num_classes=num_classes, hidden_size=video_hidden)
        # Note: StreamingVideoClassifier has: feature_extractor -> avgpool -> gru -> classifier
        
        # 2. Sensor Branch
        self.sensor_branch = SensorClassifier(input_size=6, hidden_size=sensor_hidden, num_classes=num_classes)
        
        # 3. Fusion Head (Late Fusion)
        # We concatenate the last hidden state of Video GRU and Sensor LSTM
        self.fusion_layer = nn.Sequential(
            nn.Linear(video_hidden + sensor_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, video_seq, sensor_seq):
        """
        video_seq: [batch, v_seq_len, 3, 224, 224]
        sensor_seq: [batch, s_seq_len, 6]
        """
        # Get Video Features (last GRU hidden state)
        # We need to bypass the original video classifier
        # Redefining logic here to be explicit
        batch_size, v_seq_len, c, h, w = video_seq.size()
        v_flat = video_seq.view(batch_size * v_seq_len, c, h, w)
        v_feat = self.video_branch.feature_extractor(v_flat)
        v_feat = self.video_branch.avgpool(v_feat)
        v_feat = torch.flatten(v_feat, 1)
        v_feat = v_feat.view(batch_size, v_seq_len, -1)
        v_output, _ = self.video_branch.gru(v_feat)
        v_last = v_output[:, -1, :] # [batch, video_hidden]
        
        # Get Sensor Features (last LSTM hidden state)
        # SensorClassifier returns logits, we need the hidden state
        s_output, (s_h, s_c) = self.sensor_branch.lstm(sensor_seq)
        s_last = s_h[-1] # [batch, sensor_hidden]
        
        # Concatenate and Classify
        fused = torch.cat((v_last, s_last), dim=1)
        logits = self.fusion_layer(fused)
        
        return logits
