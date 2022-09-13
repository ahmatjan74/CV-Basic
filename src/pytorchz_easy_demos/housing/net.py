from turtle import forward
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(Net, self).__init__()
        self.hidden = nn.Linear(in_features=in_features, out_features=100)
        self.predict = nn.Linear(in_features=100, out_features=out_features)
        
    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out
        
    