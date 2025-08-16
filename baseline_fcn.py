import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class FCNBaseline(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", classes=1):
        super().__init__()
        
        # Load only the encoder part from SMP
        encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        self.encoder = encoder
        
        # 1x1 conv to map final encoder features → output mask channels
        self.conv1x1 = nn.Conv2d(encoder.out_channels[-1], classes, kernel_size=1)
        
    def forward(self, x):
        features = self.encoder(x)  # list of feature maps at different depths
        last_feat = features[-1]    # final, lowest-res feature map
        
        x = self.conv1x1(last_feat)  # predict mask at low resolution
        x = nn.functional.interpolate(x, size=(x.shape[2]*32, x.shape[3]*32), mode="bilinear", align_corners=False)
        
        return x

