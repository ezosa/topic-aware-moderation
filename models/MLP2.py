import torch
import torch.nn as nn


class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        y_prob = self.layers(x)
        return y_prob
