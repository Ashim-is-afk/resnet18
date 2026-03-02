import os
import torch
import torch.nn as nn
from monai.networks.nets import resnet18

class EndometrialResNet(nn.Module):
    def __init__(self, num_status, num_figo, weights_path=None, in_channels=1):
        super().__init__()
        
        # 1. Update n_input_channels (set to 2 if using image + mask)
        self.backbone = resnet18(
            spatial_dims=3, 
            n_input_channels=in_channels, 
            num_classes=512
        )
        
        # 2. Load Pretrained MedicalNet Weights
        if weights_path and os.path.exists(weights_path):
            print(f"Loading pretrained weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            # Remove 'module.' prefix and ignore original fc layer
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'fc' not in k}
            
            # CRITICAL: If in_channels > 1, the first layer weights won't match.
            # We must skip the first conv layer loading and let it initialize randomly,
            # or manually expand the weights.
            if in_channels > 1:
                print("Multi-channel input detected: Re-initializing first conv layer.")
                new_state_dict = {k: v for k, v in new_state_dict.items() if "conv1.weight" not in k}
            
            self.backbone.load_state_dict(new_state_dict, strict=False)

        # 3. Dropout Layer for Regularization
        self.dropout = nn.Dropout(p=0.3)

        # 4. Multi-Task Heads
        self.status_head = nn.Linear(512, num_status) # Normal, Benign, Malignant
        self.figo_head = nn.Linear(512, num_figo)     # FIGO Stages I-IV

    def forward(self, x):
        # x shape: [Batch, Channels, D, H, W]
        features = self.backbone(x)
        features = self.dropout(features)
        
        status_out = self.status_head(features)
        figo_out = self.figo_head(features)
        
        return status_out, figo_out