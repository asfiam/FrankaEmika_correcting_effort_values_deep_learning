# model_seq.py
import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2),  # assuming RGB/depth stacked into 3-chunks
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(32 * 4 * 4, output_dim)

    def forward(self, x):
        # x: (batch, seq, C, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features.view(B, S, -1)


class CNNLSTMNet(nn.Module):
    def __init__(self, numeric_dim, cnn_out_dim=64, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.2):
        super().__init__()
        self.cnn_extractor = CNNFeatureExtractor(output_dim=cnn_out_dim)
        self.lstm_input_dim = numeric_dim + cnn_out_dim

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, numeric_seq, image_seq):
        cnn_features = self.cnn_extractor(image_seq)  # (B, S, cnn_out_dim)
        combined = torch.cat([numeric_seq, cnn_features], dim=-1)  # (B, S, total_dim)

        out, _ = self.lstm(combined)
        last_out = out[:, -1, :]  # last timestep
        return self.fc(last_out)
