import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from jaxtyping import Float
from torchmetrics import Accuracy, F1Score, Recall, JaccardIndex

from models.components.dinov3_wrapper import DinoV3Wrapper
from models.components.kanga_ssm import KangaSSM
from models.components.classification_head import ClassificationHead
from models.components.detection_head import DetectionHead
from models.components.segmentation_decoder import SegmentationDecoder

class VideoMambaSystem(L.LightningModule):
    def __init__(self, dim_in: int = 768, dim_out: int = 256, num_classes: int = 10, num_boxes: int = 10, target_size: int = 224, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.feature_extractor = DinoV3Wrapper(freeze=True)
        self.temporal_model = KangaSSM(d_model=dim_in)
        self.clf_head = ClassificationHead(dim_in=dim_in, num_classes=num_classes)
        self.detection_head = DetectionHead(dim_in=dim_in, num_classes=num_classes, num_boxes=num_boxes)
        self.seg_decoder = SegmentationDecoder(dim_in=dim_in, num_classes=num_classes, target_size=target_size)
        
        # Metrics
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_miou = JaccardIndex(task="multiclass", num_classes=num_classes)
        
    def forward(self, x: Float[torch.Tensor, "B T C H W"]) -> tuple[
        Float[torch.Tensor, "B num_classes"],
        Float[torch.Tensor, "B T num_boxes 4"],
        Float[torch.Tensor, "B T num_boxes num_classes"],
        Float[torch.Tensor, "B T num_classes H W"]
    ]:
        # 1. Spatial feature extraction (DinoV3)
        cls_tokens, patch_tokens = self.feature_extractor(x)
        
        # 2. Temporal modeling (KANGA SSM)
        # We contextualize the sequence of cls tokens
        ssm_out = self.temporal_model(cls_tokens)
        
        # 3. Heads
        # Classification uses the temporal context
        logits_clf = self.clf_head(ssm_out)
        
        # Object Detection and Segmentation use the patch tokens infused with temporal context
        B, T, D, P = patch_tokens.shape
        ssm_context = ssm_out.unsqueeze(-1) # [B, T, D, 1]
        infused_patches = patch_tokens + ssm_context
        
        pred_boxes, pred_box_logits = self.detection_head(infused_patches)
        logits_seg = self.seg_decoder(infused_patches)
        
        return logits_clf, pred_boxes, pred_box_logits, logits_seg

    def _shared_step(self, batch, batch_idx, prefix="train"):
        frames = batch[0]
        
        # Forward pass
        logits_clf, pred_boxes, pred_box_logits, logits_seg = self(frames)
        
        loss = 0.0
        
        # 1. Classification Loss (HMDB51 style)
        if len(batch) >= 2 and batch[1].dim() == 1:
            labels = batch[1]
            loss_clf = F.cross_entropy(logits_clf, labels)
            loss += loss_clf
            self.log(f"{prefix}_loss_clf", loss_clf)
            
            if prefix == "test":
                self.test_acc(logits_clf, labels)
                self.test_f1(logits_clf, labels)
                self.test_recall(logits_clf, labels)
            
        # 2. Segmentation Loss (DAVIS style)
        # Assuming batch is (images, masks) for DAVIS
        if len(batch) == 2 and batch[1].dim() == 4: # [B, T, H, W]
            masks = batch[1]
            # logits_seg: [B, T, num_classes, H, W]
            # Flatten to [BT, C, H, W] and [BT, H, W] for CrossEntropy
            BT, C, H, W = logits_seg.shape[0] * logits_seg.shape[1], logits_seg.shape[2], logits_seg.shape[3], logits_seg.shape[4]
            loss_seg = F.cross_entropy(logits_seg.view(BT, C, H, W), masks.view(BT, H, W))
            loss += loss_seg
            self.log(f"{prefix}_loss_seg", loss_seg)
            
            if prefix == "test":
                # Reshape for JaccardIndex: [BT, H, W] for both preds and targets
                self.test_miou(torch.argmax(logits_seg.view(BT, C, H, W), dim=1), masks.view(BT, H, W))
            
        # 3. Detection Loss (Original style: frames, labels, boxes, box_labels)
        if len(batch) == 4:
            boxes_gt, box_labels_gt = batch[2], batch[3]
            loss_box = F.l1_loss(pred_boxes, boxes_gt)
            loss_box_cls = F.cross_entropy(
                pred_box_logits.view(-1, self.hparams.num_classes), 
                box_labels_gt.view(-1).long()
            )
            loss_detection = loss_box + loss_box_cls
            loss += loss_detection
            self.log(f"{prefix}_loss_det", loss_detection)
            self.log(f"{prefix}_loss_box", loss_box)
        
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, prefix="test")
        return loss

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute())
        self.log("test_f1", self.test_f1.compute())
        self.log("test_recall", self.test_recall.compute())
        self.log("test_miou", self.test_miou.compute())
        
        # Reset metrics
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_recall.reset()
        self.test_miou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
