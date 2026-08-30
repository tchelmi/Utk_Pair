import torch
import torch.nn as nn


class PairwiseAgeMLP(nn.Module):
    """
    MLP che riceve due vettori di feature (derivati da una rete
    di age regression) e predice chi è più giovane.

    Input: concat(f1, f2, |f1 - f2|) di dimensione 3 * d
    Output: 2 logit (classe 0/1).
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = feature_dim * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        x = torch.cat([f1, f2, diff], dim=1)
        return self.net(x)

