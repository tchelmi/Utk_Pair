import torch
import torch.nn as nn
from torchvision.models import resnet18


class AgeResNet18(nn.Module):
    """
    ResNet18 modificata per regressione dell'età (uscita scalare).
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        # Regressione di età come valore continuo
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ritorna età stimata (non vincolata)
        return self.backbone(x).squeeze(1)


class AgeFeatureExtractor(nn.Module):
    """
    Estrae un vettore di feature da una ResNet18 pre-addestrata per age regression.

    Usa il backbone della stessa rete ma sostituisce il layer finale
    con nn.Identity, restituendo quindi il vettore di embedding.
    """

    def __init__(self, pretrained_backbone: AgeResNet18) -> None:
        super().__init__()
        # Copiamo il backbone e rimuoviamo il layer di regressione
        backbone = pretrained_backbone.backbone
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output: [B, out_dim]
        return self.backbone(x)

