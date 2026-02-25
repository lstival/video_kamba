import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from jaxtyping import Float

from models.components.dinov3_wrapper import DinoV3Wrapper
from models.components.kanga_ssm import KangaSSM
from models.components.classification_head import ClassificationHead
from models.components.detection_head import DetectionHead

class VideoMambaSystem(L.LightningModule):
    def __init__(self, dim_in: int = 768, dim_out: int = 256, num_classes: int = 10, num_boxes: int = 10, target_size: int = 224, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.feature_extractor = DinoV3Wrapper(freeze=True)
        self.temporal_model = KangaSSM(d_model=dim_in)
        self.clf_head = ClassificationHead(dim_in=dim_in, num_classes=num_classes)
        self.detection_head = DetectionHead(dim_in=dim_in, num_classes=num_classes, num_boxes=num_boxes)
        
    def forward(self, x: Float[torch.Tensor, "B T C H W"]) -> tuple[
        Float[torch.Tensor, "B num_classes"],
        Float[torch.Tensor, "B T num_boxes 4"],
        Float[torch.Tensor, "B T num_boxes num_classes"]
    ]:
        # 1. Spatial feature extraction (DinoV3)
        cls_tokens, patch_tokens = self.feature_extractor(x)
        
        # 2. Temporal modeling (KANGA SSM)
        # We contextualize the sequence of cls tokens
        ssm_out = self.temporal_model(cls_tokens)
        
        # 3. Heads
        # Classification uses the temporal context
        logits_clf = self.clf_head(ssm_out)
        
        # Object Detection uses the patch tokens infused with temporal context
        B, T, D, P = patch_tokens.shape
        ssm_context = ssm_out.unsqueeze(-1) # [B, T, D, 1]
        infused_patches = patch_tokens + ssm_context
        
        pred_boxes, pred_box_logits = self.detection_head(infused_patches)
        
        return logits_clf, pred_boxes, pred_box_logits

    def _shared_step(self, batch, batch_idx, prefix="train"):
        frames, labels, boxes_gt, box_labels_gt = batch
        
        # Forward pass
        logits_clf, pred_boxes, pred_box_logits = self(frames)
        
        # Loss calculation
        loss_clf = F.cross_entropy(logits_clf, labels)
        
        # Detection Loss: L1 for bounding boxes, CrossEntropy for box classes
        loss_box = F.l1_loss(pred_boxes, boxes_gt)
        loss_box_cls = F.cross_entropy(
            pred_box_logits.view(-1, self.hparams.num_classes), 
            box_labels_gt.view(-1).long()
        )
        loss_detection = loss_box + loss_box_cls
        
        loss = loss_clf + loss_detection
        
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_loss_clf", loss_clf)
        self.log(f"{prefix}_loss_det", loss_detection)
        self.log(f"{prefix}_loss_box", loss_box)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
