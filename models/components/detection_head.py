import torch
import torch.nn as nn
from jaxtyping import Float

class DetectionHead(nn.Module):
    """
    Per-frame object detection head.
    Predicts bounding boxes (x, y, w, h) and classes per frame.
    For simplicity, it pools spatial patches to predict fixed number of bounding boxes per frame.
    """
    def __init__(self, dim_in: int = 768, num_classes: int = 10, num_boxes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        
        # Predict box coords [num_boxes, 4] and logits [num_boxes, num_classes] from average spatial feature
        self.box_mlp = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.ReLU(),
            nn.Linear(256, num_boxes * 4)
        )
        
        self.cls_mlp = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.ReLU(),
            nn.Linear(256, num_boxes * num_classes)
        )

    def forward(self, patch_tokens: Float[torch.Tensor, "B T D P"]) -> tuple[
        Float[torch.Tensor, "B T num_boxes 4"], 
        Float[torch.Tensor, "B T num_boxes num_classes"]
    ]:
        B, T, D, P = patch_tokens.shape
        
        # Global average pool across spatial patches (P): [B, T, D]
        pooled = patch_tokens.mean(dim=-1)
        
        boxes = self.box_mlp(pooled).view(B, T, self.num_boxes, 4)
        # Normalize coordinates to [0, 1] using sigmoid
        boxes = torch.sigmoid(boxes)
        
        logits = self.cls_mlp(pooled).view(B, T, self.num_boxes, self.num_classes)
        
        return boxes, logits
