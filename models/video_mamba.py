import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from jaxtyping import Float
from torchmetrics import Accuracy, F1Score, Recall, JaccardIndex
from utils.davis_metrics import DAVISMetric
from utils.vos_loss import HybridVOSLoss

from models.components.dinov3_wrapper import DinoV3Wrapper
from models.components.mobilenetv2_wrapper import MobileNetV2Wrapper
from models.components.vision_mamba_tiny_wrapper import VisionMambaTinyWrapper
from models.components.kanga_ssm import KangaSSM
from models.components.kan_key_adapter import KANKeyAdapter
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
        dim_in: int = 768,             # DINOv2 ViT-B/14 dim; e.g. 256(MV2), 192(ViM-tiny)
        dim_out: int = 256,
        num_clf_classes: int = 51,
        num_seg_classes: int = 11,
        num_boxes: int = 10,
        target_size: int = 224,
        learning_rate: float = 1e-4,
        vos_loss_beta: float = 0.5,
        use_ref_context: bool = True,
        # ── Encoder ────────────────────────────────────────────────────────
        # "dino"        : DINOv2 ViT-B/14 (frozen), 768-dim, single-scale 14×14
        # "mobilenetv2" : AOT-style MobileNetV2 (trainable), 256-dim, multi-scale
        encoder_type: str = "dino",
        # Fine-scale dim fed into MemoryBank dual-scale key (-1 → same as dim_in).
        # MobileNetV2: 96, Vision-Mamba tiny: usually 96.
        dim_in_fine: int = -1,
        # Decoder up2 skip dim (-1 → same as dim_in).
        # MobileNetV2: 32, Vision-Mamba tiny: usually 64.
        dim_in_s2: int = -1,
        # Decoder up3 skip dim (-1 → same as dim_in).
        # MobileNetV2: 24, Vision-Mamba tiny: usually 48.
        dim_in_s1: int = -1,
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
        dec_dim_up1: int = 256,
        dec_dim_up2: int = 128,
        dec_dim_up3: int = 64,
        modulator_type: str = "kan",
        # Regularisation
        consistency_weight: float = 0.0,
        # Phase 2: dual-scale memory keys (Stage 3 semantic + Stage 2 fine-grained)
        prop_use_dual_scale: bool = True,
        # MobileNetV2-specific
        mv2_output_stride: int = 16,
        mv2_freeze_at: int = 0,
        mv2_pretrained: bool = True,
        # Vision-Mamba tiny-specific
        vim_stem_dim: int = 24,
        vim_stage0_dim: int = 48,
        vim_stage1_dim: int = 64,
        vim_stage2_dim: int = 96,
        vim_spatial_d_state: int = 8,
        vim_spatial_layers: int = 1,
        # Training schedule
        max_epochs: int = 20,
        # KAN-SSM Memory Key Adapter
        # When True, a lightweight KANKeyAdapter refines non-reference memory
        # bank keys recurrently over the clip to model appearance drift.
        use_kan_key_adapter: bool = True,
        kan_adapter_d_state: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Resolve per-encoder skip dims ───────────────────────────────
        # dim_in_fine : fine-scale dim for MemoryBank dual-scale key
        # dim_in_s2   : decoder up2 skip (one level coarser than stage_3 spatially)
        # dim_in_s1   : decoder up3 skip (two levels coarser)
        if encoder_type == "mobilenetv2":
            _dim_fine = MobileNetV2Wrapper.DIM_16X if dim_in_fine < 0 else dim_in_fine
            _dim_s2 = MobileNetV2Wrapper.DIM_8X if dim_in_s2 < 0 else dim_in_s2
            _dim_s1 = MobileNetV2Wrapper.DIM_4X if dim_in_s1 < 0 else dim_in_s1
        elif encoder_type == "vision_mamba_tiny":
            _dim_fine = vim_stage2_dim if dim_in_fine < 0 else dim_in_fine
            _dim_s2 = vim_stage1_dim if dim_in_s2 < 0 else dim_in_s2
            _dim_s1 = vim_stage0_dim if dim_in_s1 < 0 else dim_in_s1
        else:
            _dim_fine = dim_in if dim_in_fine < 0 else dim_in_fine
            _dim_s2 = dim_in if dim_in_s2 < 0 else dim_in_s2
            _dim_s1 = dim_in if dim_in_s1 < 0 else dim_in_s1

        # ── Encoder ────────────────────────────────────────────────────
        if encoder_type == "mobilenetv2":
            self.feature_extractor = MobileNetV2Wrapper(
                output_stride=mv2_output_stride,
                freeze_at=mv2_freeze_at,
                pretrained=mv2_pretrained,
            )
        elif encoder_type == "vision_mamba_tiny":
            self.feature_extractor = VisionMambaTinyWrapper(
                out_dim=dim_in,
                stem_dim=vim_stem_dim,
                stage0_dim=_dim_s1,
                stage1_dim=_dim_s2,
                stage2_dim=_dim_fine,
                ssm_d_state=vim_spatial_d_state,
                ssm_layers=vim_spatial_layers,
                modulator_type=modulator_type,
            )
        elif encoder_type == "dino":
            self.feature_extractor = DinoV3Wrapper(freeze=True)
        else:
            raise ValueError(
                f"Unsupported encoder_type '{encoder_type}'. "
                "Expected one of {'dino', 'mobilenetv2', 'vision_mamba_tiny'}."
            )
        self.temporal_model = KangaSSM(
            d_model=dim_in,
            d_state=ssm_d_state,
            num_layers=ssm_layers,
            use_checkpointing=use_checkpointing,
            modulator_type=modulator_type,
        )
        self.clf_head = ClassificationHead(dim_in=dim_in, num_classes=num_clf_classes)
        self.detection_head = DetectionHead(dim_in=dim_in, num_classes=num_clf_classes, num_boxes=num_boxes)
        # For multi-scale trainable encoders the decoder skips use
        # stage_2 / stage_1 / stage_0 for up1 / up2 / up3.
        # For DINO all three stages are at the same 14×14 resolution.
        _dec_up1_skip = _dim_fine  # up1: semantic refinement at same spatial as SSM
        _dec_up2_skip = _dim_s2   # up2: first genuine upsample level
        _dec_up3_skip = _dim_s1   # up3: second genuine upsample level

        self.seg_decoder = SegmentationDecoder(
            dim_ssm=dim_in,
            skip_dims=(_dec_up1_skip, _dec_up2_skip, _dec_up3_skip),
            decoder_dims=(dec_dim_up1, dec_dim_up2, dec_dim_up3),
            num_classes=num_seg_classes,
            target_size=target_size,
            fusion_mode=fusion_mode,
        )
        # Propagation: MemoryBank holds explicit K/V pairs per frame;
        # PropagationAttention cross-attends query patches to the bank.
        # ── KAN Key Adapter (optional) ──────────────────────────────────
        _key_adapter = None
        if use_kan_key_adapter:
            _key_adapter = KANKeyAdapter(
                d_key=prop_d_key,
                d_state=kan_adapter_d_state,
                num_layers=1,
            )

        self.memory_bank = MemoryBank(
            d_model=dim_in,
            d_model_fine=_dim_fine,
            d_key=prop_d_key,
            d_value=prop_d_value,
            n_objects=num_seg_classes - 1,
            max_mem_frames=max_mem_frames,
            use_dual_scale=prop_use_dual_scale,
            key_adapter=_key_adapter,
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
    # Encoder-agnostic feature helpers
    # ------------------------------------------------------------------

    def _map_ms_features(self, raw: dict) -> dict:
        """Map raw encoder output to the canonical stage keys used for SSM/memory-bank input.

        Returns a dict with keys ``"stage_3"``, ``"stage_2"``, ``"stage_1"``
        representing the primary, secondary and tertiary features **for SSM /
        reference-memory purposes**.  Decoder skip connections use
        ``_build_dec_ms()`` instead, which routes the encoder-native multi-scale
        features to the correct decoder levels.

        For DINOv2 all three are at 14×14 (same channels) — same skip and SSM
        features. For lightweight multi-scale encoders (MobileNetV2 and
        Vision-Mamba tiny), ``stage_3`` is the projected SSM input while
        ``stage_2`` and ``stage_1`` are lower-level features used by the
        fallback (no-reference) path.
        """
        if self.hparams.encoder_type in ("mobilenetv2", "vision_mamba_tiny"):
            return {
                "stage_3": raw["stage_3"],
                "stage_2": raw["stage_1"],
                "stage_1": raw["stage_0"],
            }
        # DINOv2 (default)
        return {
            "stage_3": raw["layer_11"],
            "stage_2": raw["layer_9"],
            "stage_1": raw["layer_6"],
        }

    def _get_fine_features(self, raw: dict, t: int | None = None) -> torch.Tensor:
        """Return fine-scale patch tokens for dual-scale MemoryBank updates.

        For DINOv2  → layer_9 (14×14).
        For MobileNetV2 / Vision-Mamba tiny → stage_2 (pre-projection stride-16).

        If ``t`` is given, selects frame ``t`` and permutes to ``[B, P, Ch]``.
        Otherwise returns ``[B, T, Ch, P]`` without permuting.
        """
        key = "stage_2" if self.hparams.encoder_type in ("mobilenetv2", "vision_mamba_tiny") else "layer_9"
        feat = raw[key]
        if t is not None:
            return feat[:, t].permute(0, 2, 1)  # [B, P, Ch]
        return feat  # [B, T, Ch, P]

    def _build_dec_ms(self, raw: dict, t: int) -> dict:
        """Build the decoder skip-connection dict for frame ``t``.

        The SegmentationDecoder expects keys ``"stage_3"`` / ``"stage_2"`` /
        ``"stage_1"`` as up1 / up2 / up3 skip features.

        For DINOv2 these are the same multi-scale features used internally.
        For MobileNetV2 and Vision-Mamba tiny the decoder skips are the
        *pre-projection* and lower-stride features, giving genuine multi-scale
        skip connections:

            up1 skip  "stage_3"  ← MV2 stage_2  96-ch  14×14  (same-scale refinement)
            up2 skip  "stage_2"  ← MV2 stage_1  32-ch  28×28  (first genuine upsample)
            up3 skip  "stage_1"  ← MV2 stage_0  24-ch  56×56  (second genuine upsample)
        """
        if self.hparams.encoder_type in ("mobilenetv2", "vision_mamba_tiny"):
            return {
                "stage_3": raw["stage_2"][:, t:t+1],
                "stage_2": raw["stage_1"][:, t:t+1],
                "stage_1": raw["stage_0"][:, t:t+1],
            }
        ms = self._map_ms_features(raw)
        return {k: v[:, t:t+1] for k, v in ms.items()}

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

        1. Extract backbone features for all query frames.
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
        # Map raw encoder output → canonical stage keys for SSM / decoder
        query_features_ms = self._map_ms_features(query_features_raw)
        query_patch = query_features_ms["stage_3"]  # [B, T, dim_in, P]
        D, P = query_patch.shape[2], query_patch.shape[3]

        if self.hparams.use_ref_context and ref_frame is not None and ref_mask is not None:
            # ── 2. Reference feature extraction ────────────────────────
            _ref_cls, _ref_features_raw = self.feature_extractor(ref_frame.unsqueeze(1))
            _ref_features_ms = self._map_ms_features(_ref_features_raw)
            ref_patch = _ref_features_ms["stage_3"]  # [B, 1, dim_in, P]
            # [B, 1, D, P] → [B, P, D]
            ref_patch_p = ref_patch.squeeze(1).permute(0, 2, 1)

            # Fine-scale reference features for dual-scale memory key.
            # DINOv2: layer_9 | multi-scale encoders: stage_2
            ref_patch_fine = self._get_fine_features(_ref_features_raw, t=0)  # [B, P2, Ch]

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
                curr_patch = query_patch[:, t].permute(0, 2, 1)  # [B, P, dim_in]
                # Fine-scale patches for dual-scale memory update.
                # DINOv2: layer_9 | multi-scale encoders: stage_2
                curr_patch_fine = self._get_fine_features(query_features_raw, t=t)  # [B, P2, Ch]

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
                # Decoder skip connections differ by encoder type:
                # DINOv2: all stages at 14×14 (same resolution)
                # Multi-scale encoders: stage_2(14x14), stage_1(28x28), stage_0(56x56).
                frame_ms = self._build_dec_ms(query_features_raw, t)
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
            # For multi-scale encoders the decoder skips differ from the SSM input:
            #   query_features_ms["stage_3"] = 256-ch (SSM input), but
            #   the decoder's up1 block expects 96-ch (raw stage_2, same spatial).
            # query_features_ms is only correct for DINOv2 (all stages 768-ch, same spatial).
            if self.hparams.encoder_type in ("mobilenetv2", "vision_mamba_tiny"):
                dec_ms_full = {
                    "stage_3": query_features_raw["stage_2"],
                    "stage_2": query_features_raw["stage_1"],
                    "stage_1": query_features_raw["stage_0"],
                }
            else:
                dec_ms_full = query_features_ms
            logits_seg = self.seg_decoder(infused_patches, dec_ms_full)

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
        # the attention Q-K alignment starts from random init while the backbone
        # features are already informative.
        # Diagnostic: attention entropy = 0.978 (near-uniform) confirmed this.
        prop_params = list(self.memory_bank.parameters()) + list(self.propagation_attention.parameters())
        prop_ids = {id(p) for p in prop_params}

        # For trainable encoders (MobileNetV2 / Vision-Mamba tiny) the backbone must use a much
        # lower LR than the randomly-initialised heads to preserve the pretrained
        # ImageNet features.  Applying the full lr=1e-4 corrupts the backbone
        # early in training and causes the characteristic val-J&F plateau.
        backbone_params = list(self.feature_extractor.parameters())
        backbone_ids = {id(p) for p in backbone_params}

        base_params = [
            p for p in self.parameters()
            if id(p) not in prop_ids and id(p) not in backbone_ids
        ]

        trainable_backbones = {"mobilenetv2", "vision_mamba_tiny"}
        backbone_lr = lr * 0.1 if self.hparams.encoder_type in trainable_backbones else lr * 0.01

        optimizer = torch.optim.AdamW(
            [
                {"params": prop_params,    "lr": lr * 5},   # 10→5: was overfitting to GT references
                {"params": base_params,    "lr": lr},
                {"params": backbone_params, "lr": backbone_lr},
            ],
            weight_decay=1e-2,
        )
        # CosineAnnealingLR decays LR every epoch regardless of any metric,
        # preventing the overshot that occurs when ReduceLROnPlateau never fires
        # because val_loss keeps monotonically decreasing even as J&F collapses.
        # Read T_max robustly: prefer trainer.max_epochs if already bound, else
        # fall back to the hparam (pass max_epochs=20 via ++model.max_epochs=20)
        # or a hardcoded default so the schedule is always well-defined.
        if self.trainer is not None and getattr(self.trainer, "max_epochs", None):
            t_max = self.trainer.max_epochs
        else:
            t_max = getattr(self.hparams, "max_epochs", 20)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
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
