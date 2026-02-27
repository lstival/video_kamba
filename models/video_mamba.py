import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from jaxtyping import Float
from torchmetrics import Accuracy, F1Score, Recall, JaccardIndex
from utils.davis_metrics import DAVISMetric
from utils.vos_loss import HybridVOSLoss

from models.components.dinov3_wrapper import DinoV3Wrapper
from models.components.kanga_ssm import KangaSSM
from models.components.classification_head import ClassificationHead
from models.components.detection_head import DetectionHead
from models.components.segmentation_decoder import SegmentationDecoder

class VideoMambaSystem(L.LightningModule):
    """Multi-task video understanding model.

    Supports three task families configured purely through YAML:

    * **Action classification** – HMDB51 style ``(frames, labels)``.
    * **DAVIS semi-supervised VOS** – ``(ref_img, ref_mask, query_imgs, query_masks)``.
    * **Multi-object VOS** – YouTube-VOS / MOSE style 6-element batches
      ``(ref_img, ref_mask, query_imgs, query_masks, obj_present, meta)``.

    Args:
        dim_in:          Feature dimension from DinoV3 (default 768).
        dim_out:         Projection dimension (unused by default, reserved).
        num_clf_classes: Number of action-classification output classes
                         (e.g. 51 for HMDB51).
        num_seg_classes: Number of segmentation output channels
                         (background + n_id, e.g. 11 for n_id=10).
        num_boxes:       Fixed number of predicted bounding boxes per frame.
        target_size:     Spatial resolution of segmentation output.
        learning_rate:   AdamW base learning rate.
        vos_loss_beta:   Beta for HybridVOSLoss (BCE weight; default 0.5).
    """

    def __init__(
        self,
        dim_in: int = 768,
        dim_out: int = 256,
        num_clf_classes: int = 51,
        num_seg_classes: int = 11,
        num_boxes: int = 10,
        target_size: int = 224,
        learning_rate: float = 1e-4,
        vos_loss_beta: float = 0.5,
        use_ref_context: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = DinoV3Wrapper(freeze=True)
        self.temporal_model = KangaSSM(d_model=dim_in)
        self.clf_head = ClassificationHead(dim_in=dim_in, num_classes=num_clf_classes)
        self.detection_head = DetectionHead(dim_in=dim_in, num_classes=num_clf_classes, num_boxes=num_boxes)
        self.seg_decoder = SegmentationDecoder(dim_in=dim_in, num_classes=num_seg_classes, target_size=target_size)

        # Object Memory: one embedding per segmentation channel (bg + objects)
        self.mask_embedding = nn.Embedding(num_seg_classes, dim_in)

        # Hybrid VOS loss
        self.vos_loss_fn = HybridVOSLoss(beta=vos_loss_beta, from_logits=True)
        
        # Metrics
        self.test_acc = Accuracy(task="multiclass", num_classes=num_clf_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_clf_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_clf_classes, average="macro")
        self.test_miou = JaccardIndex(task="multiclass", num_classes=num_seg_classes)
        self.davis_metric = DAVISMetric()
        self.vos_val_metric = DAVISMetric()  # reused for YouTube-VOS / MOSE val
        
    # ------------------------------------------------------------------
    # Batch type detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_vos_batch(batch) -> bool:
        """Return True iff ``batch`` is a 6-element multi-object VOS batch.

        Multi-object VOS batches from ``MultiObjectVOSDataModule`` carry:
        ``(ref_img, ref_mask, query_images, query_masks, obj_present, meta)``.
        The 5th element (index 4) is a boolean ``obj_present`` tensor, which
        distinguishes this format from the standard DAVIS 4-tuple and from
        HMDB51 2-tuples.
        """
        return (
            isinstance(batch, (list, tuple))
            and len(batch) == 6
            and isinstance(batch[4], torch.Tensor)
            and batch[4].dtype == torch.bool
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, ref_frame: torch.Tensor = None, ref_mask: torch.Tensor = None) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Main inference process for multi-task video understanding.

        Performs:
        1. Feature extraction using DinoV3.
        2. Sequence contextualization using KangaSSM.
        3. Task decoding via specialized heads.

        Args:
            x: Query video sequence of shape [B, T, C, H, W].
            ref_frame: Optional reference frame [B, 1, C, H, W] for VOS.
            ref_mask: Optional reference mask [B, H, W] for VOS.

        Returns:
            A tuple of (logits_clf, pred_boxes, pred_box_logits, logits_seg).
        """
        # 1. Spatial feature extraction (DinoV3)
        B, T = x.shape[0], x.shape[1]
        query_cls, query_patch = self.feature_extractor(x) # query_patch: [B, T, D, P]
        
        if self.hparams.use_ref_context and ref_frame is not None and ref_mask is not None:
            # Extract ref features
            ref_cls, ref_patch = self.feature_extractor(ref_frame.unsqueeze(1)) # ref_patch: [B, 1, D, P]
            
            # Robust extraction of dimensions
            B_ref, T_ref, D_feat, P_feat = ref_patch.shape
            H_p = W_p = int(P_feat**0.5) 
            
            # Spatial Infusion: Map mask to the patch grid
            ref_mask_small = F.interpolate(
                ref_mask.unsqueeze(1).float(), 
                size=(H_p, W_p), 
                mode="nearest"
            ).long() # [B, 1, H_p, W_p]
            
            ref_mask_safe = ref_mask_small.clone()
            ref_mask_safe[ref_mask_small == 255] = 0
            
            # Embed mask patches -> [B, P, D]
            ref_mask_patch_emb = self.mask_embedding(ref_mask_safe.squeeze(1)) 
            ref_mask_patch_emb = ref_mask_patch_emb.reshape(B, H_p * W_p, -1) 
            
            # Correct Transposition: ref_patch is [B, 1, D, P]. We need [B, 1, P, D]
            ref_patch_p = ref_patch.transpose(2, 3) # [B, 1, P, D]
            
            # Infuse patch tokens
            ref_patch_infused = ref_patch_p + ref_mask_patch_emb.unsqueeze(1) # [B, 1, P, D]
            
            # --- PATCH-SEQUENCE SSM ---
            # Instead of flattening patches into one sequence, treat each (x,y) location as a sequence
            # Query patches: [B, T, D, P] -> [B, T, P, D]
            query_patch_p = query_patch.transpose(2, 3) 
            
            # Full sequence per patch: [RefPatch, Q1Patch, Q2Patch, ..., QTPatch]
            all_patches = torch.cat([ref_patch_infused, query_patch_p], dim=1) # [B, T+1, P, D]
            
            # Group by patch location: [B, P, T+1, D] -> [B*P, T+1, D]
            all_patches = all_patches.permute(0, 2, 1, 3).reshape(B * P_feat, T + 1, D_feat)
            
            # Temporal modeling
            ssm_out_raw = self.temporal_model(all_patches) # [B*P, T+1, D]
            
            # Reshape back and extract query part: [B, P, T+1, D]
            ssm_out_full = ssm_out_raw.reshape(B, P_feat, T + 1, D_feat)
            ssm_out_query = ssm_out_full[:, :, 1:, :] # [B, P, T, D]
            
            # Final infused patches for decoder: [B, T, D, P]
            infused_patches = ssm_out_query.permute(0, 2, 3, 1)
            
            # Global representation for ClassificationHead: pool over patches
            # [B, P, T, D] -> [B, T, D]
            ssm_cls = ssm_out_query.mean(dim=1) 
        else:
            # Fallback for no-ref case (e.g. action recognition)
            # Treat each patch as a sequence independently
            P_feat = query_patch.shape[-1]
            all_patches = query_patch.transpose(2, 3).reshape(B * P_feat, T, -1)
            ssm_out_raw = self.temporal_model(all_patches)
            infused_patches = ssm_out_raw.view(B, P_feat, T, -1).permute(0, 2, 3, 1)
            ssm_cls = ssm_out_raw.view(B, P_feat, T, -1).mean(dim=1)
            ref_patch = None
        
        # 3. Heads
        logits_clf = self.clf_head(ssm_cls)
        pred_boxes, pred_box_logits = self.detection_head(infused_patches)
        logits_seg = self.seg_decoder(infused_patches)
        
        return logits_clf, pred_boxes, pred_box_logits, logits_seg

    # ------------------------------------------------------------------
    # Multi-object VOS step (YouTube-VOS / MOSE)
    # ------------------------------------------------------------------

    def _vos_step(
        self,
        batch,
        batch_idx: int,
        prefix: str,
    ) -> torch.Tensor:
        """Training / validation step for multi-object VOS batches.

        Batch format: ``(ref_img, ref_mask, query_images, query_masks,
        obj_present, meta)``.

        Args:
            batch:     6-element tuple from ``MultiObjectVOSDataModule``.
            batch_idx: Batch index (unused, kept for API consistency).
            prefix:    Log prefix – ``"train"`` or ``"val"``.

        Returns:
            Scalar hybrid VOS loss.
        """
        ref_img, ref_mask, query_images, query_masks, obj_present, _meta = batch

        # Forward pass with reference memory
        _logits_clf, _pred_boxes, _pred_box_logits, logits_seg = self(
            query_images, ref_frame=ref_img, ref_mask=ref_mask
        )
        # logits_seg: [B, T, num_seg_classes, H, W]

        loss = self.vos_loss_fn(logits_seg, query_masks, obj_present)

        assert not torch.isnan(loss), "_vos_step: NaN loss detected."

        self.log(f"{prefix}_loss_vos", loss, prog_bar=(prefix == "val"), on_epoch=True, on_step=(prefix == "train"))
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_epoch=True, on_step=(prefix == "train"))

        if prefix in ("val", "test"):
            preds = torch.argmax(logits_seg, dim=2)  # [B, T, H, W]
            self.vos_val_metric.update(preds, query_masks)

        return loss

    # ------------------------------------------------------------------
    # Shared step (dispatches by batch format)
    # ------------------------------------------------------------------

    def _shared_step(self, batch, batch_idx, prefix="train"):
        """Shared logic for training, validation, and test steps.

        Dispatches the batch to the appropriate task-specific processing (VOS, Detection, or Classification)
        based on the 'Batch Type Detection' process.

        Args:
            batch: The input batch from the dataloader.
            batch_idx: Index of the current batch.
            prefix: Log prefix ('train', 'val', 'test').

        Returns:
            The computed loss for the step.
        """
        loss = 0.0
        label_smoothing = 0.1 if prefix == "train" else 0.0

        # ── Multi-object VOS (YouTube-VOS / MOSE) ────────────────────────
        if self._is_vos_batch(batch):
            return self._vos_step(batch, batch_idx, prefix)

        # Handle DAVIS Semi-Supervised format: (ref_img, ref_mask, query_images, query_masks)
        if len(batch) == 4 and batch[0].dim() == 4 and batch[1].dim() == 3 and batch[2].dim() == 5:
            ref_img, ref_mask, query_images, query_masks = batch
            logits_clf, pred_boxes, pred_box_logits, logits_seg = self(query_images, ref_frame=ref_img, ref_mask=ref_mask)
            
            BT, C, H, W = logits_seg.shape[0] * logits_seg.shape[1], logits_seg.shape[2], logits_seg.shape[3], logits_seg.shape[4]
            loss_seg = F.cross_entropy(
                logits_seg.view(BT, C, H, W), 
                query_masks.view(BT, H, W), 
                ignore_index=255,
                label_smoothing=label_smoothing
            )
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
            loss_clf = F.cross_entropy(logits_clf, labels, label_smoothing=label_smoothing)
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
            loss_seg = F.cross_entropy(logits_seg.view(BT, C, H, W), masks.view(BT, H, W), label_smoothing=label_smoothing)
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
                pred_box_logits.view(-1, self.hparams.num_clf_classes),
                box_labels_gt.view(-1).long(),
                label_smoothing=label_smoothing
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

        vos_res = self.vos_val_metric.compute()
        if vos_res["J&F"] > 0:
            self.log("test_vos_J", vos_res["J"])
            self.log("test_vos_F", vos_res["F"])
            self.log("test_vos_J_and_F", vos_res["J&F"])

        # Reset metrics
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_recall.reset()
        self.test_miou.reset()
        self.davis_metric.reset()
        self.vos_val_metric.reset()

    def on_validation_epoch_end(self):
        davis_res = self.davis_metric.compute()
        if davis_res["J&F"] > 0:
            self.log("val_J", davis_res["J"])
            self.log("val_F", davis_res["F"])
            self.log("val_J_and_F", davis_res["J&F"])
        self.davis_metric.reset()

        vos_res = self.vos_val_metric.compute()
        if vos_res["J&F"] > 0:
            self.log("val_vos_J", vos_res["J"])
            self.log("val_vos_F", vos_res["F"])
            self.log("val_vos_J_and_F", vos_res["J&F"])
        self.vos_val_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=1e-2
        )
        return optimizer
