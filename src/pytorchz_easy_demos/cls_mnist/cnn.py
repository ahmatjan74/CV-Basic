import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),# 32 -> 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 28 -> 14
        )
        self.fc = nn.Linear(14 * 14 * 32, 10)
        
    def forward(self, x):
        out = self.conv(x) # 14
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
    