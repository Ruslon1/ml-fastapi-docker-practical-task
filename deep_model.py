import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


class IrisNet(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 16, output_size: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, features):
        return self.layers(features)


class TorchIrisClassifier:
    def __init__(self, model: IrisNet, scaler: StandardScaler):
        self.model = model.eval()
        self.scaler = scaler

    def predict(self, features):
        scaled_features = self.scaler.transform(np.asarray(features, dtype=np.float32))
        tensor = torch.tensor(scaled_features, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(tensor)

        return logits.argmax(dim=1).numpy()
