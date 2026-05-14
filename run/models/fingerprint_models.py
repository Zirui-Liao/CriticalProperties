from __future__ import annotations

from torch import nn


class MLPMultiTask(nn.Module):
    """Fingerprint MLP used by the multitask fingerprint Optuna script."""

    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 128), dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_size
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
