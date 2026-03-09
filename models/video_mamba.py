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
from models.components.memory_bank import MemoryBank
from models.components.propagation_attention import PropagationAttention

class VideoMambaSystem(L.LightningModule):
    """Multi-task video understanding model — Phase 2 (Hiera backbone).

    Supports three task families configured purely through YAML:

    * **Action classification** – HMDB51 style ``(frames, labels)``.
    * **DAVIS semi-supervised VOS** – ``(ref_img, ref_mask, query_imgs, query_masks)``.
    * **Multi-object VOS** – YouTube-VOS / MOSE style 6-element batches
      ``(ref_img, ref_mask, query_imgs, query_masks, obj_present, meta)``.

    Phase 2 changes vs Phase 1 (Reverted to DINO):
    - Backbone: DINOv2 ViT-B/14 (768-dim).
    - MemoryBank dual-scale keys: Layer 11 (semantic) + Layer 9 (fine-grained).
    - SegmentationDecoder: hierarchical fusion with DINO Layer 6/9/11 skip connections.

    Args:
        dim_in:              Primary feature dimension — Stage 3 channels (384).
        dim_out:             Reserved projection dimension.
        num_clf_classes:     Action-classification output classes.
        num_seg_classes:     Segmentation channels (background + n_id).
        num_boxes:           Fixed predicted bounding boxes per frame.
        target_size:         Spatial resolution of segmentation output.
        learning_rate:       AdamW base learning rate.
        vos_loss_beta:       Beta for HybridVOSLoss (BCE weight).
        prop_use_dual_scale: Use Stage 2 fine-grained keys in MemoryBank.
    """

    def __init__(
        self,
        dim_in: int = 768,             # DINOv2 ViT-B/14 dim (was 384 Hiera)
        dim_out: int = 256,
        num_clf_classes: int = 51,
        num_seg_classes: int = 11,
        num_boxes: int = 10,
        target_size: int = 224,
        learning_rate: float = 1e-4,
        vos_loss_beta: float = 0.5,
        use_ref_context: bool = True,
        # Memory-attention propagation
        prop_d_key: int = 256,
        prop_d_value: int = 256,
        prop_n_heads: int = 8,
        prop_dropout: float = 0.1,
        max_mem_frames: int = 5,
        memory_update_freq: int = 1,
        # Scheduled sampling: 0.0 = always use GT mask for memory update (teacher
        # forcing), 1.0 = always use predicted mask (pure autoregressive).
        scheduled_sampling_rate: float = 0.0,
        # KAN-SSM
        ssm_d_state: int = 16,
        ssm_layers: int = 1,
        use_checkpointing: bool = False,
        fusion_mode: str = "kan_spatial",
        modulator_type: str = "kan",
        # Regularisation
        consistency_weight: float = 0.0,
        # Phase 2: dual-scale memory keys (Stage 3 semantic + Stage 2 fine-grained)
        prop_use_dual_scale: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # DINO layers mapped to hierarchical stages:
        # stage_3 (semantic)   <- layer_11 (768)
        # stage_2 (fine)       <- layer_9  (768)
        # stage_1 (spatial)    <- layer_6  (768)
        dim_fine = dim_in  # Layer 9: 768
        dim_s1   = dim_in  # Layer 6: 768

        self.feature_extractor = DinoV3Wrapper(freeze=True)
        self.temporal_model = KangaSSM(
            d_model=dim_in,
            d_state=ssm_d_state,
            num_layers=ssm_layers,
            use_checkpointing=use_checkpointing,
            modulator_type=modulator_type,
        )
        self.clf_head = ClassificationHead(dim_in=dim_in, num_classes=num_clf_classes)
        self.detection_head = DetectionHead(dim_in=dim_in, num_classes=num_clf_classes, num_boxes=num_boxes)
        self.seg_decoder = SegmentationDecoder(
            dim_ssm=dim_in,
            skip_dims=(dim_in, dim_fine, dim_s1),  # Stage 3 / 2 / 1 channels
            num_classes=num_seg_classes,
            target_size=target_size,
            fusion_mode=fusion_mode,
        )
        # Propagation: MemoryBank holds explicit K/V pairs per frame;
        # PropagationAttention cross-attends query patches to the bank.
        self.memory_bank = MemoryBank(
            d_model=dim_in,
            d_model_fine=dim_fine,
            d_key=prop_d_key,
            d_value=prop_d_value,
            n_objects=num_seg_classes - 1,
            max_mem_frames=max_mem_frames,
            use_dual_scale=prop_use_dual_scale,
        )
        self.propagation_attention = PropagationAttention(
            d_model=dim_in,
            d_key=prop_d_key,
            d_value=prop_d_value,
            n_heads=prop_n_heads,
            dropout=prop_dropout,
        )

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

    def forward(
        self,
        x: torch.Tensor,
        ref_frame: torch.Tensor = None,
        ref_mask: torch.Tensor = None,
        query_masks: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: memory-attention propagation followed by KAN-SSM refinement.

        For VOS (with ``ref_frame``/``ref_mask``):

        1. Extract DINOv2 features for all query frames (backbone frozen).
        2. Encode the reference frame into the ``MemoryBank`` (K/V pairs with
           object ID embeddings injected into the values).
        3. For each query frame *t*:

           a. Cross-attend query patches to memory  →  propagated features.
           b. Feed propagated features through the KAN-SSM  →  temporal context.
           c. Decode to segmentation logits.
           d. Update the memory bank with the current frame (GT or predicted
              mask chosen via ``scheduled_sampling_rate``).

        Args:
            x:           Query video frames ``[B, T, C, H, W]``.
            ref_frame:   Reference (first) frame ``[B, C, H, W]``.
            ref_mask:    Integer segmentation mask for the reference frame
                         ``[B, H, W]`` (0 = background, 1..N = objects,
                         255 = void).
            query_masks: Ground-truth query masks ``[B, T, H, W]``.  Required
                         only during training for scheduled-sampling on memory
                         updates; may be ``None`` at evaluation.

        Returns:
            Tuple ``(logits_clf, pred_boxes, pred_box_logits, logits_seg)``
            where ``logits_seg`` is ``[B, T, num_seg_classes, H, W]``.
        """
        B, T, C, H, W = x.shape

        # ── 1. Spatial feature extraction ──────────────────────────────
        query_cls, query_features_raw = self.feature_extractor(x)
        # Map DINO layers to the stage keys expected by downstream modules
        query_features_ms = {
            "stage_3": query_features_raw["layer_11"],
            "stage_2": query_features_raw["layer_9"],
            "stage_1": query_features_raw["layer_6"],
        }
        query_patch = query_features_ms["stage_3"]  # [B, T, 768, 196]
        D, P = query_patch.shape[2], query_patch.shape[3]

        if self.hparams.use_ref_context and ref_frame is not None and ref_mask is not None:
            # ── 2. Reference feature extraction ────────────────────────
            _ref_cls, _ref_features_raw = self.feature_extractor(ref_frame.unsqueeze(1))
            _ref_features_ms = {
                "stage_3": _ref_features_raw["layer_11"],
                "stage_2": _ref_features_raw["layer_9"],
                "stage_1": _ref_features_raw["layer_6"],
            }
            ref_patch = _ref_features_ms["stage_3"]                 # [B, 1, 768, 196]
            # [B, 1, D, P] → [B, P, D]
            ref_patch_p = ref_patch.squeeze(1).permute(0, 2, 1)

            # Fine-scale reference features for dual-scale memory key.
            ref_patch_fine = _ref_features_ms["stage_2"].squeeze(1).permute(0, 2, 1)  # [B, P2, 768]

            # ── 3. Initialise memory bank with reference frame ──────────
            self.memory_bank.encode_reference(ref_patch_p, ref_mask, ref_patch_fine)

            # Spatial decoder guidance: binary objectness from reference mask
            ref_mask_safe = ref_mask.clone()
            ref_mask_safe[ref_mask == 255] = 0
            prev_guide_mask = (ref_mask_safe > 0).float().unsqueeze(1)  # [B, 1, H, W]

            all_preds_seg: list[torch.Tensor] = []
            ssm_states = None

            for t in range(T):
                # Current frame patches: [B, D, P] → [B, P, D]
                curr_patch = query_patch[:, t].permute(0, 2, 1)  # [B, P, 384]
                # Fine-scale patches for dual-scale memory update.
                curr_patch_fine = query_features_ms["stage_2"][:, t].permute(0, 2, 1)  # [B, P2, 192]

                # ── 4. Propagation: cross-attend to memory bank ─────────
                K_mem, V_mem = self.memory_bank.get_memory()
                prop_feat = self.propagation_attention(
                    curr_patch, K_mem, V_mem
                )  # [B, P, 384]

                # ── 5. KAN-SSM temporal refinement (patch-parallel) ─────
                # Feed as [B*P, 1, D] — one token per step preserves the
                # SSM's recurrent hidden-state across the clip.
                prop_flat = prop_feat.reshape(B * P, 1, -1)  # [B*P, 1, 384]
                ssm_out_flat, ssm_states = self.temporal_model(
                    prop_flat, prev_states=ssm_states, return_last_state=True
                )
                # [B*P, 1, D] → [B, 1, D, P] for decoder
                last_feat = (
                    ssm_out_flat.squeeze(1)   # [B*P, D]
                    .reshape(B, P, D)         # [B, P, D]
                    .permute(0, 2, 1)         # [B, D, P]
                    .unsqueeze(1)             # [B, 1, D, P]
                )

                # ── 6. Hierarchical decoding ────────────────────────────
                # frame_ms passes all Hiera stages to the decoder (stage_1/2/3).
                frame_ms = {k: v[:, t:t+1] for k, v in query_features_ms.items()}
                logits_t = self.seg_decoder(
                    last_feat, frame_ms, prev_mask=prev_guide_mask
                )  # [B, 1, num_classes, H, W]
                all_preds_seg.append(logits_t)

                # ── 7. Update memory bank ───────────────────────────────
                if t % self.hparams.memory_update_freq == 0:
                    use_gt = (
                        self.training
                        and query_masks is not None
                        and torch.rand(1).item() > self.hparams.scheduled_sampling_rate
                    )
                    if use_gt:
                        gt_t = query_masks[:, t]
                        if gt_t.ndim == 4:  # [B, 1, H, W]
                            gt_t = gt_t.squeeze(1)
                        mem_mask = gt_t  # [B, H, W]
                    else:
                        # Soft predicted mask — detached to prevent graph growth
                        mem_mask = (
                            torch.softmax(logits_t, dim=2).squeeze(1).detach()
                        )  # [B, C, H, W]
                    self.memory_bank.add_frame(
                        curr_patch.detach(),
                        mem_mask,
                        curr_patch_fine.detach(),
                    )

                # Update spatial guide: total objectness across all object channels
                pred_soft = torch.softmax(logits_t, dim=2)  # [B, 1, C, H, W]
                prev_guide_mask = (
                    pred_soft[:, 0, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0).detach()
                )  # [B, 1, H, W]

            logits_seg = torch.cat(all_preds_seg, dim=1)  # [B, T, C, H, W]
            ssm_cls = query_cls  # [B, T, 384]

        else:
            # ── Fallback: no reference context ─────────────────────────
            # Process all patches in parallel across time.
            # [B, T, D, P] → [B, P, T, D] → [B*P, T, D]
            all_patches = query_patch.permute(0, 3, 1, 2).reshape(B * P, T, D)
            ssm_out_raw = self.temporal_model(all_patches)  # [B*P, T, D]
            # [B*P, T, D] → [B, P, T, D] → [B, T, D, P]
            infused_patches = ssm_out_raw.reshape(B, P, T, D).permute(0, 2, 3, 1)
            ssm_cls = ssm_out_raw.reshape(B, P, T, D).mean(dim=1)  # [B, T, D]
            logits_seg = self.seg_decoder(infused_patches, query_features_ms)

        # ── 8. Classification and detection heads ───────────────────────
        logits_clf = self.clf_head(ssm_cls)
        pred_boxes = torch.zeros(B, T, 10, 4, device=x.device)
        pred_box_logits = torch.zeros(
            B, T, 10, self.hparams.num_clf_classes, device=x.device
        )

        # Resize segmentation logits to original spatial resolution if needed
        if logits_seg.shape[-2:] != (H, W):
            L_B, L_T, L_C, L_H, L_W = logits_seg.shape
            logits_seg = F.interpolate(
                logits_seg.view(L_B * L_T, L_C, L_H, L_W),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
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

        # Forward pass with reference memory and ground truth masks (for scheduled sampling)
        _logits_clf, _pred_boxes, _pred_box_logits, logits_seg = self(
            query_images, ref_frame=ref_img, ref_mask=ref_mask, query_masks=query_masks
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
            logits_clf, pred_boxes, pred_box_logits, logits_seg = self(
                query_images, ref_frame=ref_img, ref_mask=ref_mask, query_masks=query_masks
            )
            
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

        # Separate propagation parameters — they need a higher LR because
        # the decoder's frozen-backbone skip connections provide a gradient
        # shortcut that prevents Q-K alignment in the attention path.
        # Diagnostic: attention entropy = 0.978 (near-uniform) confirmed this.
        prop_params = list(self.memory_bank.parameters()) + list(self.propagation_attention.parameters())
        prop_ids = {id(p) for p in prop_params}
        base_params = [p for p in self.parameters() if id(p) not in prop_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": prop_params, "lr": lr * 10},
                {"params": base_params, "lr": lr},
            ],
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _get_mask_embedding(self, mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Converts a mask (hard or soft) into a patch-level identity embedding [B, P, D].
        mask: [B, 1, H, W], [B, H, W], or [B, 1, 1, H, W]
        """
        # Ensure 4D [N, 1, H, W]
        if mask.ndim == 5:
            mask = mask.view(-1, 1, mask.shape[-2], mask.shape[-1])
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
            
        # Downsample to patch resolution (e.g. 14x14)
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        
        # mask is [B, 1, h, w]. Flatten to [B, 1, P]
        B, _, h_p, w_p = mask.shape
        mask_flat = mask.reshape(B, 1, h_p * w_p).transpose(1, 2) # [B, P, 1]
        
        # Map foreground (channel 1) to "Object" embedding (index 1)
        # and background to "BG" embedding (index 0).
        # We use a linear interpolation between BG and OBJ embeddings for soft masks.
        bg_emb = self.mask_embedding(torch.zeros(B, h_p * w_p, device=mask.device).long()) # [B, P, D]
        obj_emb = self.mask_embedding(torch.ones(B, h_p * w_p, device=mask.device).long())  # [B, P, D]
        
        return (1.0 - mask_flat) * bg_emb + mask_flat * obj_emb

    def _infuse_identity(self, patch: torch.Tensor, mask_emb: torch.Tensor) -> torch.Tensor:
        """
        Mixes frame patches and identity embeddings.
        patch: [B, 1, P, D] or [B, T, P, D]
        mask_emb: [B, P, D]
        """
        if patch.ndim == 4:
            mask_emb = mask_emb.unsqueeze(1) # [B, 1, P, D]
            
        if self.hparams.identity_mode == "add":
            return patch + mask_emb
        elif self.hparams.identity_mode == "concat":
            # [B, T, P, 2*D]
            return torch.cat([patch, mask_emb.expand_as(patch)], dim=-1)
        elif self.hparams.identity_mode == "modulate":
            return patch * torch.sigmoid(mask_emb)
        return patch
