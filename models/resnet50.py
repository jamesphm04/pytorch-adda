from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torch import nn
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class ResnetEncoder(nn.Module):
    """Resnet 50 encoder model for ADDA."""
    def __init__(self, pretrained=False):
        """Init Resnet 50 encoder."""
        super(ResnetEncoder, self).__init__()
        
        model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

        # Get the number of input features for the classifier
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        # Get the numbner of output channels for the Mask Predictor
        dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

        # Replace the box predictor
        model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=2)

        # Replace the mask predictor
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=2)
        device = torch.device("cpu")
        model.to(device)
        
        if pretrained:
            model.load_state_dict(torch.load('weights/maskrcnn_resnet50_fpn_v2.pth', map_location=device))
        
        self.encoder = model.backbone.body
        
        # Add a final fully connected layer to match LeNetEncoder output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Just in case
        self.fc1 = nn.Linear(256, 500)
        
    def forward(self, x):
        features = self.encoder(x)  # features is an OrderedDict
        x = features["0"]  # Pick the "highest resolution" feature map (you can also try "1", "2", "3")
        x = self.avgpool(x)  # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [B, 2048]
        feat = self.fc1(x)  # [B, 500]
        return feat