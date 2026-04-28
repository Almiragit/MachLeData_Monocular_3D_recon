import torch
import torch.nn as nn
import torchvision.models as models

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        
        # Use ResNet18 as the Encoder (extracting features from the image)
        # We exclude the last two layers (Global Average Pooling and Fully Connected layer)
        backbone = models.resnet18(weights=None) 
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        
        # Simple Decoder to reconstruct the depth map from features
        # It upsamples the feature map back to a higher resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Final layer to produce a single-channel depth map
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Normalizes depth values between 0 and 1
        )

    def forward(self, x):
        # x shape: (batch_size, 3, H, W)
        features = self.encoder(x)
        # depth_map shape: (batch_size, 1, H_new, W_new)
        depth_map = self.decoder(features)
        return depth_map