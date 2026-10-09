import torch
import torch.nn as nn
import torchvision.models as models


class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        features = self.cnn(x)
        features = features.view(B, T, 512)

        _, (hn, _) = self.lstm(features)

        out = self.dropout(hn[-1])
        out = self.fc(out)

        return out