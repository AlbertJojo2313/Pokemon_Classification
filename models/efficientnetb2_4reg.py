# Contains the model architecture for all the supported regions (Kanto-Johto-Hoenn-Sinnoh)
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


class PokemonClassifierEfficientNetB2_4reg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = (EfficientNet_B2_Weights.DEFAULT)

        self. model = efficientnet_b2(weights=weights)

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes),
        )
    # Freeze all layers except the classifier
    def freeze_backbone(self):
        for param in self.model.features.parameters():
            param.requires_grad = False
    
    # Unfreeze all layers
    def unfreeze_backbone(self):
        for param in self.model.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)