import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from jaxtyping import Float
from torchmetrics import Accuracy, F1Score, Recall, JaccardIndex
from utils.davis_metrics import DAVISMetric

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
        
        # Object Memory
        self.mask_embedding = nn.Embedding(num_classes + 1, dim_in)
        
        # Metrics
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_miou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.davis_metric = DAVISMetric()
        
    def forward(self, x: torch.Tensor, ref_frame: torch.Tensor = None, ref_mask: torch.Tensor = None) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # 1. Spatial feature extraction (DinoV3)
        query_cls, query_patch = self.feature_extractor(x)
        
        if ref_frame is not None and ref_mask is not None:
            # Extract ref features
            ref_cls, ref_patch = self.feature_extractor(ref_frame.unsqueeze(1))
            
            # Embed the reference integer mask and pool it to match CLS token dimension
            # ref_mask is [B, H, W]. Output of embedding is [B, H, W, dim_in]
            ref_mask_emb = self.mask_embedding(ref_mask).mean(dim=(1, 2)).unsqueeze(1) # [B, 1, dim_in]
            
            # Infuse the reference class token with the mask information
            ref_cls_infused = ref_cls + ref_mask_emb
            
            # Prepend context to the sequence
            cls_tokens = torch.cat([ref_cls_infused, query_cls], dim=1)
        else:
            cls_tokens = query_cls
            ref_patch = None
        
        # 2. Temporal modeling (KANGA SSM)
        # We contextualize the sequence of cls tokens
        ssm_out = self.temporal_model(cls_tokens)
        
        if ref_frame is not None and ref_mask is not None:
            # Discard the memory sequence element for task decoding
            ssm_out = ssm_out[:, 1:, :]
        
        # 3. Heads
        # Classification uses the temporal context
        logits_clf = self.clf_head(ssm_out)
        
        # Object Detection and Segmentation use the patch tokens infused with temporal context
        B, T, D, P = query_patch.shape
        ssm_context = ssm_out.unsqueeze(-1) # [B, T, D, 1]
        infused_patches = query_patch + ssm_context
        
        pred_boxes, pred_box_logits = self.detection_head(infused_patches)
        logits_seg = self.seg_decoder(infused_patches)
        
        return logits_clf, pred_boxes, pred_box_logits, logits_seg

    def _shared_step(self, batch, batch_idx, prefix="train"):
        loss = 0.0
        
        # Handle DAVIS Semi-Supervised format: (ref_img, ref_mask, query_images, query_masks)
        if len(batch) == 4 and batch[0].dim() == 4 and batch[1].dim() == 3 and batch[2].dim() == 5:
            ref_img, ref_mask, query_images, query_masks = batch
            logits_clf, pred_boxes, pred_box_logits, logits_seg = self(query_images, ref_frame=ref_img, ref_mask=ref_mask)
            
            BT, C, H, W = logits_seg.shape[0] * logits_seg.shape[1], logits_seg.shape[2], logits_seg.shape[3], logits_seg.shape[4]
            loss_seg = F.cross_entropy(logits_seg.view(BT, C, H, W), query_masks.view(BT, H, W))
            loss += loss_seg
            self.log(f"{prefix}_loss_seg", loss_seg)
            
            if prefix in ["val", "test"]:
                preds = torch.argmax(logits_seg, dim=2) # [B, T, H, W]
                self.davis_metric.update(preds, query_masks)
                
            self.log(f"{prefix}_loss", loss, prog_bar=True)
            return loss
            
        # Handle HMDB51 or other standard formats
        frames = batch[0]
        
        # Forward pass without reference context
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
        
        davis_res = self.davis_metric.compute()
        if davis_res["J&F"] > 0:
            self.log("test_J", davis_res["J"])
            self.log("test_F", davis_res["F"])
            self.log("test_J_and_F", davis_res["J&F"])
        
        # Reset metrics
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_recall.reset()
        self.test_miou.reset()
        self.davis_metric.reset()
        
    def on_validation_epoch_end(self):
        davis_res = self.davis_metric.compute()
        if davis_res["J&F"] > 0:
            self.log("val_J", davis_res["J"])
            self.log("val_F", davis_res["F"])
            self.log("val_J_and_F", davis_res["J&F"])
        self.davis_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
