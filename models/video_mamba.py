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
        identity_mode: str = "add", # "add", "concat", "modulate"
        online_context: bool = False,
        consistency_weight: float = 0.0,
        ssm_d_state: int = 16,
        ssm_layers: int = 1,
        compress_skip: bool = False,
        use_checkpointing: bool = False,
        fusion_mode: str = "concat",
        modulator_type: str = "kan",
        propagation_mode: str = "soft_mask", # "soft_mask", "direct_feature", "teacher_forcing"
        scheduled_sampling_rate: float = 0.0, # 0.0 = always use GT (teacher forcing), 1.0 = always use prediction
        # Encoder fine-tuning: 0.0 = frozen, >0.0 = backbone LR multiplier (e.g. 0.1 → 1/10th of base LR)
        encoder_lr_scale: float = 0.0,
        # LR scheduler: "none", "cosine", "cosine_warmup"
        lr_scheduler: str = "cosine_warmup",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Adjusted dimension for identity injection
        self.dim_infused = dim_in * 2 if identity_mode == "concat" else dim_in
        
        self.feature_extractor = DinoV3Wrapper(freeze=(encoder_lr_scale == 0.0))
        self.temporal_model = KangaSSM(
            d_model=self.dim_infused,
            d_state=ssm_d_state,
            num_layers=ssm_layers,
            use_checkpointing=use_checkpointing,
            modulator_type=modulator_type,
        )
        self.clf_head = ClassificationHead(dim_in=self.dim_infused, num_classes=num_clf_classes)
        self.detection_head = DetectionHead(dim_in=self.dim_infused, num_classes=num_clf_classes, num_boxes=num_boxes)
        self.seg_decoder = SegmentationDecoder(
            dim_ssm=self.dim_infused, 
            dim_dinov2=dim_in, 
            num_classes=num_seg_classes, 
            target_size=target_size, 
            compress_skip=compress_skip,
            fusion_mode=fusion_mode
        )

        # Object Memory: one embedding per segmentation channel (bg + objects)
        self.mask_embedding = nn.Embedding(num_seg_classes, dim_in)

        # Projection for direct feature feedback: maps SSM output back to mask embedding space
        if propagation_mode == "direct_feature":
            self.feat_proj = nn.Linear(self.dim_infused, dim_in)

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

    def _get_mask_embedding(self, mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Helper to embed a mask into the patch grid. 
        Supports both hard label masks [B, H, W] and soft probability masks [B, C, H, W].
        """
        B = mask.shape[0]
        
        if mask.ndim == 3: # Hard mask [B, H, W]
            mask_small = F.interpolate(
                mask.unsqueeze(1).float(), 
                size=(h, w), 
                mode="nearest"
            ).long() # [B, 1, h, w]
            
            mask_safe = mask_small.clone()
            mask_safe[mask_small == 255] = 0
            
            # Embed mask patches -> [B, h*w, D]
            emb = self.mask_embedding(mask_safe.squeeze(1)) 
            return emb.reshape(B, h * w, -1)
        else: # Soft probabilities [B, C, H, W]
            mask_small = F.interpolate(
                mask.float(), 
                size=(h, w), 
                mode="area"
            ) # [B, C, h, w]
            
            # [B, h*w, C]
            mask_flat = mask_small.view(B, mask.shape[1], h * w).transpose(1, 2)
            
            # Multiply by embedding weights: [B, h*w, C] @ [C, D] -> [B, h*w, D]
            emb = torch.matmul(mask_flat, self.mask_embedding.weight)
            return emb

    def _infuse_identity(self, patch: torch.Tensor, mask_emb: torch.Tensor) -> torch.Tensor:
        """Helper to infuse identity information into patches based on identity_mode."""
        if self.hparams.identity_mode == "concat":
            # mask_emb is [B, P, D] or [B, T, P, D]
            # patch is [B, 1, P, D] or [B, T, P, D]
            if mask_emb.dim() == 3 and patch.dim() == 4:
                mask_emb = mask_emb.unsqueeze(1)
            elif mask_emb.dim() == 4 and patch.dim() == 3:
                patch = patch.unsqueeze(1)
            return torch.cat([patch, mask_emb], dim=-1)
        elif self.hparams.identity_mode == "modulate":
            if mask_emb.dim() == 3 and patch.dim() == 4:
                mask_emb = mask_emb.unsqueeze(1)
            return patch * torch.sigmoid(mask_emb)
        else: # "add"
            if mask_emb.dim() == 3 and patch.dim() == 4:
                mask_emb = mask_emb.unsqueeze(1)
            return patch + mask_emb

    def forward(self, x: torch.Tensor, ref_frame: torch.Tensor = None, ref_mask: torch.Tensor = None) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Recursive inference for VOS with mask feedback and multi-scale fusion."""
        B, T, C, H, W = x.shape
        # 1. Spatial feature extraction (DinoV3 - Multi-scale)
        query_cls, query_features_ms = self.feature_extractor(x) 
        # query_features_ms: {"layer_3": [B, T, D, P], ...}
        
        # We use the deepest layer (layer_11) for temporal propagation
        query_patch = query_features_ms["layer_11"]
        D, P = query_patch.shape[2], query_patch.shape[3]
        h = w = int(P**0.5)

        if self.hparams.use_ref_context and ref_frame is not None and ref_mask is not None:
            # 2. Reference Initializer
            ref_cls, ref_features_ms = self.feature_extractor(ref_frame.unsqueeze(1))
            ref_patch = ref_features_ms["layer_11"]
            
            # Embed the reference mask
            ref_mask_emb = self._get_mask_embedding(ref_mask, h, w) # [B, P, D]
            
            # Infuse identity into ref patches
            ref_patch_p = ref_patch.transpose(2, 3) # [B, 1, P, D]
            ref_patch_infused = self._infuse_identity(ref_patch_p, ref_mask_emb)
            
            # 3. Recursive Sequence Processing
            # For 80% potential, we feed the PREVIOUS mask back into the CURRENT frame
            # This turns the SSM into a stateful tracker
            
            all_preds_seg = []
            # Initialize recursive mask with the reference mask
            prev_mask_emb = ref_mask_emb # [B, P, D]
            
            # We process query frames one by one to allow mask feedback
            # Note: During training, we can parallelize if we don't have feedback, 
            # but for "Memory Bank" we need recursion.
            
            # Initial sequence state: Start with reference
            current_seq = ref_patch_infused # [B, 1, P, D]
            
            for t in range(T):
                # Current frame patches
                curr_patch = query_patch[:, t:t+1].transpose(2, 3) # [B, 1, P, D]
                
                # INFUSE PREVIOUS MASK into CURRENT FRAME
                # This is the "Memory Bank" feedback loop
                curr_infused = self._infuse_identity(curr_patch, prev_mask_emb)
                
                # Update sequence
                current_seq = torch.cat([current_seq, curr_infused], dim=1)
                
                # Apply SSM to the whole sequence so far (or just the step)
                # For efficiency we group patches: [B*P, len, D]
                seq_len = current_seq.shape[1]
                seq_flattened = current_seq.permute(0, 2, 1, 3).reshape(B * P, seq_len, -1)
                
                # SSM temporal update
                ssm_out = self.temporal_model(seq_flattened) # [B*P, len, D]
                
                # Get the latest frame's features
                last_feat = ssm_out[:, -1:, :].reshape(B, P, 1, -1).permute(0, 2, 3, 1) # [B, 1, D, P]
                
                # 4. Hierarchical Decoding (using all scale features for frame t)
                # Extract frame-specific ms features
                frame_ms = {k: v[:, t:t+1] for k, v in query_features_ms.items()}
                
                # Decode mask
                logits_t = self.seg_decoder(last_feat, frame_ms) # [B, 1, C, H, W]
                all_preds_seg.append(logits_t)
                
                # 5. Update Recursive Mask for next step (based on propagation_mode)
                prop_mode = self.hparams.propagation_mode
                sr = self.hparams.scheduled_sampling_rate

                if prop_mode == "direct_feature":
                    # Pass raw SSM features directly - no mask bottleneck
                    # last_feat is [B, 1, D, P] -> reshape to [B, P, D]
                    prev_mask_emb = self.feat_proj(
                        last_feat.squeeze(1).permute(0, 2, 1)  # [B, P, D_infused]
                    )  # [B, P, dim_in]

                elif prop_mode == "teacher_forcing" and self.training:
                    # Use ground truth mask embedding during training (sampled by scheduled_sampling_rate)
                    # Note: gt_mask_t is the query mask at timestep t (passed in during training)
                    # Fallback to soft_mask if no GT available
                    use_pred = (torch.rand(1).item() < sr)
                    if use_pred:
                        pred_probs_t = torch.softmax(logits_t.squeeze(1), dim=1)
                        prev_mask_emb = self._get_mask_embedding(pred_probs_t, h, w)
                    else:
                        # For teacher forcing we use prev soft mask as no GT access in forward
                        pred_probs_t = torch.softmax(logits_t.squeeze(1), dim=1)
                        prev_mask_emb = self._get_mask_embedding(pred_probs_t, h, w)

                else: # soft_mask (default)
                    pred_probs_t = torch.softmax(logits_t.squeeze(1), dim=1) # [B, C, H, W]
                    prev_mask_emb = self._get_mask_embedding(pred_probs_t, h, w)
            
            logits_seg = torch.cat(all_preds_seg, dim=1)
            ssm_cls = query_cls # Simplification for classification heads in VOS mode
            
            # If concat mode, pad ssm_cls to match dim_infused
            if self.hparams.identity_mode == "concat":
                zeros = torch.zeros_like(ssm_cls)
                ssm_cls = torch.cat([ssm_cls, zeros], dim=-1)

        else:
            # Fallback for no-ref case (standard feedforward)
            P_feat = query_patch.shape[-1]
            all_patches = query_patch.transpose(2, 3).reshape(B * P_feat, T, -1)
            
            # If concat mode, we must zero-pad to match dim_infused (2*D)
            if self.hparams.identity_mode == "concat":
                zeros = torch.zeros_like(all_patches)
                all_patches = torch.cat([all_patches, zeros], dim=-1)
                
            ssm_out_raw = self.temporal_model(all_patches)
            infused_patches = ssm_out_raw.view(B, P_feat, T, -1).permute(0, 2, 3, 1)
            ssm_cls = ssm_out_raw.view(B, P_feat, T, -1).mean(dim=1)
            
            # Simple decoding without multi-scale for fallback
            # (In practice, you'd want to handle this better)
            logits_seg = self.seg_decoder(infused_patches, query_features_ms)
            
        # 3. Heads
        logits_clf = self.clf_head(ssm_cls)
        # For simplicity, we only run detection on final features
        # Note: infused_patches is not defined in recursive path yet, we'd use the last ssm output
        # pred_boxes, pred_box_logits = self.detection_head(infused_patches)
        pred_boxes = torch.zeros(B, T, 10, 4, device=x.device) # Dummies
        pred_box_logits = torch.zeros(B, T, 10, self.hparams.num_clf_classes, device=x.device)
        
        # Resize logits_seg back to input resolution if needed
        if logits_seg.shape[-2:] != (H, W):
            L_B, L_T, L_C, L_H, L_W = logits_seg.shape
            logits_seg = F.interpolate(
                logits_seg.view(L_B * L_T, L_C, L_H, L_W),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).view(L_B, L_T, L_C, H, W)

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

        self.log(f"{prefix}_loss_vos", loss, prog_bar=(prefix == "val"), on_epoch=True, on_step=(prefix == "train"), batch_size=ref_img.shape[0])
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_epoch=True, on_step=(prefix == "train"), batch_size=ref_img.shape[0])

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
            bs = ref_img.shape[0]
            logits_clf, pred_boxes, pred_box_logits, logits_seg = self(query_images, ref_frame=ref_img, ref_mask=ref_mask)
            
            BT, C, H, W = logits_seg.shape[0] * logits_seg.shape[1], logits_seg.shape[2], logits_seg.shape[3], logits_seg.shape[4]
            loss_seg = F.cross_entropy(
                logits_seg.view(BT, C, H, W), 
                query_masks.view(BT, H, W), 
                ignore_index=255,
                label_smoothing=label_smoothing
            )
            loss += loss_seg
            self.log(f"{prefix}_loss_seg", loss_seg, batch_size=bs)
            
            # 2. Temporal Consistency Loss (Tactic E)
            if self.hparams.consistency_weight > 0 and query_images.shape[1] > 1:
                # Encourage subsequent frames to have similar predictions
                probs = torch.softmax(logits_seg, dim=2)
                loss_cons = F.mse_loss(probs[:, 1:], probs[:, :-1])
                loss += self.hparams.consistency_weight * loss_cons
                self.log(f"{prefix}_loss_consistency", loss_cons, batch_size=bs)
            
            if prefix in ["val", "test"]:
                preds = torch.argmax(logits_seg, dim=2) # [B, T, H, W]
                self.davis_metric.update(preds, query_masks)
                
            self.log(f"{prefix}_loss", loss, prog_bar=True, batch_size=bs)
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
        # Skip calculations if we haven't updated metrics (e.g. during sanity check)
        if self.trainer.sanity_checking:
            return

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
        lr = self.hparams.learning_rate
        encoder_lr_scale = self.hparams.encoder_lr_scale

        # Build parameter groups: optionally fine-tune encoder at a much smaller LR
        encoder_params = list(self.feature_extractor.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        decoder_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]

        if encoder_lr_scale > 0.0:
            param_groups = [
                {"params": decoder_params, "lr": lr, "name": "decoder"},
                {"params": encoder_params, "lr": lr * encoder_lr_scale, "name": "encoder"},
            ]
        else:
            # Encoder is fully frozen — only pass trainable (decoder) params
            param_groups = [{"params": decoder_params, "lr": lr, "name": "decoder"}]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

        scheduler_name = self.hparams.lr_scheduler
        if scheduler_name == "none":
            return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(0.05 * total_steps))  # 5% warmup

        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=lr * 1e-2
            )
        else:  # "cosine_warmup" (default)
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=lr * 1e-2)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
