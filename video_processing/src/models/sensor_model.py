import torch
import torch.nn as nn

class SensorAutoencoder(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, latent_dim=16):
        super(SensorAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_encode = nn.Linear(hidden_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        _, (h_n, _) = self.encoder(x)
        latent = self.fc_encode(h_n.squeeze(0))
        
        # Reconstruct sequence
        h_d = self.fc_decode(latent).unsqueeze(0)
        # Repeat h_d for each timestep in the sequence
        # For simplicity, we can use the same input for each step of decoder or use zero-inputs
        seq_len = x.size(1)
        decoder_input = torch.zeros(x.size(0), seq_len, x.size(2)).to(x.device) # This is a placeholder
        # More standard: repeat latent or use it as hidden state
        # Here we'll just use the hidden state and a dummy input
        decoder_input = h_d.transpose(0, 1).repeat(1, seq_len, 1)
        
        output, _ = self.decoder(decoder_input)
        reconstructed = self.output_layer(output)
        return reconstructed

    def get_anomaly_score(self, x):
        reconstructed = self.forward(x)
        score = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return score
