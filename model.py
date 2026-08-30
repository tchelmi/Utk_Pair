from typing import cast

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetPairwiseAge(nn.Module):
    """
    Modello basato su ResNet18 che riceve due immagini facciali
    concatenate lungo la dimensione dei canali:

      img1: [B, 3, H, W]
      img2: [B, 3, H, W]
      concat -> [B, 6, H, W]

    e produce i logit per 2 classi:
      - 0: la prima immagine è più giovane
      - 1: la prima immagine è più vecchia (o coetanea)
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        base = resnet18(pretrained=pretrained)

        # Adattiamo il primo layer conv da 3 -> 6 canali
        old_conv = base.conv1
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=cast(tuple[int, int], old_conv.kernel_size),
            stride=cast(tuple[int, int], old_conv.stride),
            padding=cast(tuple[int, int] | str, old_conv.padding),
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            # Inizializziamo copiando i pesi sui primi 3 canali
            # e duplicandoli sugli altri 3
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = old_conv.weight

        base.conv1 = new_conv

        # Sostituiamo il classificatore finale con uno a 2 classi
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, 2)

        self.backbone = base

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Concatenazione lungo i canali
        x = torch.cat([img1, img2], dim=1)
        logits = self.backbone(x)
        return logits

