import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class PokemonClassifierEfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT.transforms()
        )  # Avoids preprocessing issues with normalization
        self.model = efficientnet_b0(weights=weights)

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    # Freeze all layers except the classifier
    def freeze_backbone(self):

        for param in self.model.features.parameters():
            param.requires_grad = False

    # Unfreeze all layers for fine-tuning
    def unfreeze_backbone(self):
        for param in self.model.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
