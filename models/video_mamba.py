from __future__ import annotations

import math
from typing import Any, Tuple, Dict, List, Optional, Union

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
from models.components.classification_head import ClassificationHead
from models.components.detection_head import DetectionHead
from models.components.segmentation_decoder import SegmentationDecoder
from models.components.memory_state_bank import MemoryStateBank
from models.components.feature_fusion import FeatureFusion
from models.components.fast_kan_layer import FastKANLayer

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
        # Memory update frequency (kept for compat)
        memory_update_freq: int = 1,
        # Scheduled sampling: 0.0 = always use GT mask for memory update (teacher
        # forcing), 1.0 = always use predicted mask (pure autoregressive).
        scheduled_sampling_rate: float = 0.0,
        # Optional schedule for scheduled sampling. If both are >= 0, the
        # effective rate is linearly interpolated from start -> end over epochs
        # after warmup; otherwise `scheduled_sampling_rate` is used as a fixed value.
        scheduled_sampling_start: float = -1.0,
        scheduled_sampling_end: float = -1.0,
        scheduled_sampling_warmup_epochs: int = 0,
        # KAN-SSM
        ssm_d_state: int = 16,
        ssm_layers: int = 1,
        use_checkpointing: bool = False,
        fusion_mode: str = "kan_spatial",
        dec_dim_up1: int = 128,
        dec_dim_up2: int = 64,
        dec_dim_up3: int = 32,
        modulator_type: str = "kan",
        # Regularisation
        consistency_weight: float = 0.0,
        # Phase 2: dual-scale memory keys (Stage 3 semantic + Stage 2 fine-grained)
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
        vim_bidirectional: bool = True, # Item 3: Enable bidirectional spatial scan by default
        # Training schedule
        max_epochs: int = 20,
        propagation_lr_multiplier: float = 1.0,
        # Separate multiplier for the temporal SSM (KangaSSM).
        # Kept lower than the decoder (1.0) because temporal_model.out_proj
        # is a primary explosion site — giving it less LR reduces the update
        # magnitude even before gradient clipping intervenes.
        temporal_lr_multiplier: float = 0.5,
        trainable_backbone_lr_multiplier: float = 0.05,
        frozen_backbone_lr_multiplier: float = 0.01,
        optimizer_weight_decay: float = 1e-2,
        scheduler_eta_min: float = 1e-6,
        # Linear LR warmup before the cosine decay kicks in.
        # Set to >0 epochs to ramp LR from (lr * warmup_start_factor) → lr
        # before handing off to CosineAnnealingLR.  Strongly recommended for
        # fine-tuning from a pretrained checkpoint on a new domain.
        lr_warmup_epochs: int = 0,
        lr_warmup_start_factor: float = 0.1,
        # Option 1: Dynamic Tracker
        use_identity_modulation: bool = False,
        modulator_grid_size: int = 4,
        # ── Dual-Timescale State Memory (DTSM) ─────────────────────────────
        # A slow SSM path that accumulates long-range context within a video
        # without any attention mechanism.  The slow path is strictly causal
        # (unidirectional) with near-1 state decay so it acts as a persistent
        # memory consolidator over hundreds of frames.
        # See design note: "Dual-Timescale State Memory for Long-Range VOS".
        use_dual_timescale_memory: bool = False,
        slow_ssm_d_state: int = 64,           # larger state for long-range capacity
        slow_ssm_log_a_init: float = -3.0,    # log_A init: A_bar ≈ exp(-exp(-3)*δ) ≈ 0.995
        **kwargs: Any,
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
                bidirectional=vim_bidirectional, # Item 3
            )
        elif encoder_type == "dino":
            self.feature_extractor = DinoV3Wrapper(freeze=True)
        else:
            raise ValueError(
                f"Unsupported encoder_type '{encoder_type}'. "
                "Expected one of {'dino', 'mobilenetv2', 'vision_mamba_tiny'}."
            )
        self.temporal_model_local = KangaSSM(
            d_model=dim_in,
            d_state=ssm_d_state,
            num_layers=ssm_layers,
            use_checkpointing=use_checkpointing,
            modulator_type=modulator_type,
            identity_dim=dim_in if use_identity_modulation else None,
            grid_size=modulator_grid_size,
            bidirectional=True, # Item 3: Temporal bidirectional for offline/fallback
        )
        self.temporal_model_global = KangaSSM(
            d_model=dim_in,
            d_state=ssm_d_state,
            num_layers=ssm_layers,
            use_checkpointing=use_checkpointing,
            modulator_type=modulator_type,
            identity_dim=dim_in if use_identity_modulation else None,
            grid_size=modulator_grid_size,
            bidirectional=True, # Item 3
        )
        self.ssm_output_fusion = nn.Linear(dim_in * 2, dim_in)
        self.memory_bank_local = MemoryStateBank()
        self.memory_bank_global = MemoryStateBank()

        # Item 2: KAN Dynamic Temporal Fusion (Local vs Global)
        # Replacing Linear with FastKANLayer for higher expressivity
        self.temporal_gate = FastKANLayer(dim_in * 2, dim_in, grid_size=modulator_grid_size)

        # ── Dual-Timescale State Memory (DTSM) ──────────────────────────────
        # Slow consolidation path: unidirectional KangaSSM with near-1 decay.
        # Key design decisions:
        #   - bidirectional=False: state must flow forward causally to persist
        #     across clips; bidirectional would require seeing future frames.
        #   - d_state=slow_ssm_d_state (64): 4× the fast paths (16) for capacity.
        #   - num_layers=1: one layer is sufficient; the fast paths already
        #     provide multi-layer temporal refinement.
        #   - grid_size=4: smaller KAN basis reduces params on the slow path.
        #   - log_A init to slow_ssm_log_a_init (-3.0): gives A_bar ≈ 0.995,
        #     near-lossless carry — the slow SSM "forgets" at ~0.5% per frame.
        #   - slow_residual_proj near-zero init: slow path starts as a null
        #     correction and learns to contribute incrementally.
        # No cross-attention or softmax anywhere in this path.
        self.use_dual_timescale_memory = use_dual_timescale_memory
        if use_dual_timescale_memory:
            self.temporal_model_slow = KangaSSM(
                d_model=dim_in,
                d_state=slow_ssm_d_state,
                num_layers=1,
                use_checkpointing=use_checkpointing,
                modulator_type=modulator_type,
                grid_size=4,
                bidirectional=False,
            )
            for layer in self.temporal_model_slow.layers:
                if hasattr(layer, "log_A"):
                    nn.init.constant_(layer.log_A, slow_ssm_log_a_init)
            self.memory_bank_slow = MemoryStateBank()
            self.slow_residual_proj = nn.Linear(dim_in, dim_in, bias=False)
            nn.init.normal_(
                self.slow_residual_proj.weight,
                std=0.02 / math.sqrt(2),
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
        # Note: self.memory_bank is now replaced by dual local/global banks.
        # Keeping self.memory_bank as a reference to self.memory_bank_local for compat if needed.
        self.memory_bank = self.memory_bank_local
        self.feature_fusion = FeatureFusion(d_model=dim_in)

        # Spatial identity encoding: one-hot object masks → patch-level embeddings.
        # Conv2d with stride=16 downsamples naturally to the patch grid — no attention.
        # Injected additively into query features at every propagation frame, giving
        # the SSM a persistent per-object signal analogous to AOTT's id_bank.
        # See: AOTT (Hierarchical Propagation), Kim et al., NeurIPS 2022.
        # Input : [B, num_seg_classes, H, W]  (one-hot mask at full resolution)
        # Output: [B, dim_in, H//16, W//16]   (patch-grid identity embeddings)
        self.patch_wise_id_bank = nn.Conv2d(
            num_seg_classes, dim_in, kernel_size=17, stride=16, padding=8, bias=False
        )
        nn.init.normal_(self.patch_wise_id_bank.weight, std=0.02)

        # Inter-block carry state for sliding-window eval (reset_memory=False path).
        # Not model parameters — cleared between sequences by reset_carry_state().
        self._carry_prev_mask_int: Optional[torch.Tensor] = None  # [B, H, W] cpu
        self._carry_prev_patch: Optional[torch.Tensor] = None      # [B, P, D] cpu
        self._carry_gh_gw: Optional[tuple[int, int]] = None

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
                "stage_3_hw": raw.get("stage_3_hw"),
                "stage_2_hw": raw.get("stage_1_hw"),
                "stage_1_hw": raw.get("stage_0_hw"),
            }
        # DINOv2 (default)
        return {
            "stage_3": raw["layer_11"],
            "stage_2": raw["layer_9"],
            "stage_1": raw["layer_6"],
            "stage_3_hw": raw.get("layer_11_hw"),
            "stage_2_hw": raw.get("layer_9_hw"),
            "stage_1_hw": raw.get("layer_6_hw"),
        }

    def _get_fine_features(
        self,
        raw: dict,
        t: int | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int] | None]:
        """Return fine-scale patch tokens for dual-scale MemoryBank updates.

        For DINOv2  → layer_9 (14×14).
        For MobileNetV2 / Vision-Mamba tiny → stage_2 (pre-projection stride-16).

        If ``t`` is given, selects frame ``t`` and permutes to ``[B, P, Ch]``.
        Otherwise returns ``[B, T, Ch, P]`` without permuting.

        Returns:
            Tuple ``(features, feat_hw)`` where ``feat_hw`` is ``(Hf, Wf)`` when
            available in the raw encoder output.
        """
        if self.hparams.encoder_type in ("mobilenetv2", "vision_mamba_tiny"):
            key, hw_key = "stage_2", "stage_2_hw"
        else:
            key, hw_key = "layer_9", "layer_9_hw"

        feat = raw[key]
        feat_hw = raw.get(hw_key)
        if t is not None:
            return feat[:, t].permute(0, 2, 1), feat_hw  # [B, P, Ch]
        return feat, feat_hw  # [B, T, Ch, P]

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
                "stage_3_hw": raw.get("stage_2_hw"),
                "stage_2_hw": raw.get("stage_1_hw"),
                "stage_1_hw": raw.get("stage_0_hw"),
            }
        ms = self._map_ms_features(raw)
        return {
            "stage_3": ms["stage_3"][:, t:t+1],
            "stage_2": ms["stage_2"][:, t:t+1],
            "stage_1": ms["stage_1"][:, t:t+1],
            "stage_3_hw": ms.get("stage_3_hw"),
            "stage_2_hw": ms.get("stage_2_hw"),
            "stage_1_hw": ms.get("stage_1_hw"),
        }

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

    def _get_scheduled_sampling_rate(self) -> float:
        """Return the effective scheduled-sampling rate for the current epoch.

        If `scheduled_sampling_start` and `scheduled_sampling_end` are both set
        (>= 0), applies a linear epoch-wise ramp after warmup.
        Otherwise returns the fixed `scheduled_sampling_rate`.
        """
        fixed = float(getattr(self.hparams, "scheduled_sampling_rate", 0.0))
        start = float(getattr(self.hparams, "scheduled_sampling_start", -1.0))
        end = float(getattr(self.hparams, "scheduled_sampling_end", -1.0))

        if start < 0.0 or end < 0.0:
            return max(0.0, min(1.0, fixed))

        warmup = max(0, int(getattr(self.hparams, "scheduled_sampling_warmup_epochs", 0)))
        current_epoch = int(getattr(self, "current_epoch", 0))

        if current_epoch <= warmup:
            return max(0.0, min(1.0, start))

        if self.trainer is not None and getattr(self.trainer, "max_epochs", None):
            total_epochs = int(self.trainer.max_epochs)
        else:
            total_epochs = int(getattr(self.hparams, "max_epochs", 1))

        ramp_span = max(1, total_epochs - warmup - 1)
        progress = max(0.0, min(1.0, (current_epoch - warmup) / ramp_span))
        rate = start + (end - start) * progress
        return max(0.0, min(1.0, rate))

    # ------------------------------------------------------------------
    # Inter-block carry state management (eval / long-video inference)
    # ------------------------------------------------------------------

    def reset_carry_state(self) -> None:
        """Clear the inter-block carry state.

        Must be called between sequences when using ``reset_memory=False`` in
        :meth:`forward` for sliding-window inference, so that stale state from
        one sequence does not bleed into the next.
        """
        self._carry_prev_mask_int = None
        self._carry_prev_patch = None
        self._carry_gh_gw = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        ref_frame: torch.Tensor = None,
        ref_mask: torch.Tensor = None,
        query_masks: torch.Tensor = None,
        reset_memory: bool = True,
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
            reset_memory: When ``True`` (default / training), seeds the SSM
                         memory banks from the reference frame and clears the
                         carry state.  Set ``False`` for all blocks after the
                         first during sliding-window long-video inference so
                         that DTSM slow-path state and the prev-frame context
                         carry across block boundaries.

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
        query_patch_hw = query_features_ms.get("stage_3_hw")
        D, P = query_patch.shape[2], query_patch.shape[3]

        if self.hparams.use_ref_context and ref_frame is not None and ref_mask is not None:

            _use_carry = (
                not reset_memory
                and self._carry_prev_mask_int is not None
                and self._carry_gh_gw is not None
            )

            if not _use_carry:
                # ── 2. Reference feature extraction ──────────────────────────────
                _ref_cls, _ref_features_raw = self.feature_extractor(ref_frame.unsqueeze(1))
                _ref_features_ms = self._map_ms_features(_ref_features_raw)
                ref_patch = _ref_features_ms["stage_3"]   # [B, 1, dim_in, P]
                ref_patch_hw = _ref_features_ms.get("stage_3_hw")
                # [B, 1, D, P] → [B, P, D]
                ref_patch_p = ref_patch.squeeze(1).permute(0, 2, 1)

                # Patch-grid spatial dimensions (for identity embedding alignment)
                if ref_patch_hw is not None:
                    gh, gw = ref_patch_hw
                else:
                    gh, gw = H // 16, W // 16
                self._carry_gh_gw = (gh, gw)

                # ── Spatial identity encoding from reference mask ─────────────────
                ref_mask_safe = ref_mask.clone()
                ref_mask_safe[ref_mask == 255] = 0
                ref_onehot = (
                    F.one_hot(ref_mask_safe.long(), num_classes=self.hparams.num_seg_classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )  # [B, num_seg_classes, H, W]
                ref_id_embed = self.patch_wise_id_bank(ref_onehot)  # [B, D, ~gh, ~gw]
                if ref_id_embed.shape[-2:] != (gh, gw):
                    ref_id_embed = F.adaptive_avg_pool2d(ref_id_embed, (gh, gw))
                ref_id_flat = ref_id_embed.flatten(2).permute(0, 2, 1)  # [B, P, D]

                # ── 3. Initialise memory bank from reference frame ────────────────
                # Identity-enhanced self-fusion seeds the SSM with appearance + position.
                ref_fused = self.feature_fusion(ref_patch_p + ref_id_flat, ref_patch_p)
                ref_flat = ref_fused.reshape(B * P, 1, D)

                id_seeding = None
                if self.hparams.use_identity_modulation:
                    id_seeding = (
                        ref_id_embed.mean(dim=[-2, -1])
                        .unsqueeze(1).repeat(1, P, 1).view(B * P, D)
                    )

                _, ref_h_local = self.temporal_model_local(
                    ref_flat, prev_states=None, return_last_state=True, identity=id_seeding
                )
                _, ref_h_global = self.temporal_model_global(
                    ref_flat, prev_states=None, return_last_state=True, identity=id_seeding
                )

                self.memory_bank_local.reset()
                self.memory_bank_global.reset()
                self.memory_bank_local.set_state(ref_h_local)
                self.memory_bank_global.set_state(ref_h_global)

                if self.use_dual_timescale_memory:
                    _, ref_h_slow = self.temporal_model_slow(
                        ref_flat, prev_states=None, return_last_state=True
                    )
                    self.memory_bank_slow.reset()
                    self.memory_bank_slow.set_state(ref_h_slow)

                prev_mask_int = ref_mask_safe.long()  # [B, H, W]
                prev_patch = ref_patch_p.detach()     # [B, P, D]
            else:
                # ── Carry state from previous block (no reference re-seeding) ─────
                # SSM states in memory_bank_* already hold end-of-last-block state.
                # Only prev_mask_int / prev_patch / gh,gw are restored from carry.
                gh, gw = self._carry_gh_gw  # type: ignore[misc]
                prev_mask_int = self._carry_prev_mask_int.to(x.device)  # type: ignore
                prev_patch = self._carry_prev_patch.to(x.device)        # type: ignore

            prev_guide_mask = (prev_mask_int > 0).float().unsqueeze(1)  # [B, 1, H, W]
            ss_rate = self._get_scheduled_sampling_rate()
            all_preds_seg: list[torch.Tensor] = []
            ssm_states = None

            for t in range(T):
                # Current frame patches: [B, D, P] → [B, P, D]
                curr_patch = query_patch[:, t].permute(0, 2, 1)  # [B, P, D]

                # ── Identity injection: choose mask for this frame ───────────────
                # Training:  ss_rate > 0 → Bernoulli mix of GT (teacher forcing)
                #            and predicted (autoregressive) masks per sample.
                #            ss_rate == 0 → pure teacher forcing (GT only).
                # Inference: always propagate from the predicted mask.
                if self.training and query_masks is not None and ss_rate > 0.0:
                    use_pred = torch.rand(B, device=x.device) < ss_rate  # [B] bool
                    gt_int = query_masks[:, t].long().clamp(
                        0, self.hparams.num_seg_classes - 1
                    )
                    mask_for_id = torch.where(
                        use_pred.view(B, 1, 1), prev_mask_int, gt_int
                    )
                elif self.training and query_masks is not None:
                    # ss_rate == 0: teacher-force with GT masks
                    mask_for_id = query_masks[:, t].long().clamp(
                        0, self.hparams.num_seg_classes - 1
                    )
                else:
                    # Inference: fully autoregressive
                    mask_for_id = prev_mask_int

                mask_for_id_safe = mask_for_id.clone()
                mask_for_id_safe[mask_for_id_safe == 255] = 0

                id_onehot = (
                    F.one_hot(mask_for_id_safe, num_classes=self.hparams.num_seg_classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )  # [B, num_seg_classes, H, W]
                curr_id_embed = self.patch_wise_id_bank(id_onehot)  # [B, D, ~gh, ~gw]
                if curr_id_embed.shape[-2:] != (gh, gw):
                    curr_id_embed = F.adaptive_avg_pool2d(curr_id_embed, (gh, gw))
                curr_id_flat = curr_id_embed.flatten(2).permute(0, 2, 1)  # [B, P, D]

                # ── 4 & 5. Feature fusion and KAN-SSM temporal refinement ────────
                # Fuse current features (+ identity signal) against the PREVIOUS
                # frame — enabling short-range motion propagation via FeatureFusion
                # while long-range context flows through SSM hidden states.
                fused = self.feature_fusion(curr_patch + curr_id_flat, prev_patch)
                fused_flat = fused.reshape(B * P, 1, D)

                # SSM identity conditioning: spatial-mean of current id embedding.
                curr_identity = None
                if self.hparams.use_identity_modulation:
                    curr_identity = (
                        curr_id_embed.mean(dim=[-2, -1])                   # [B, D]
                        .unsqueeze(1).repeat(1, P, 1).reshape(B * P, -1)  # [B*P, D]
                    )

                prev_h_local = self.memory_bank_local.get_state()
                prev_h_global = self.memory_bank_global.get_state()

                out_local, next_h_local = self.temporal_model_local(
                    fused_flat, prev_states=prev_h_local, return_last_state=True,
                    identity=curr_identity,
                )
                out_global, next_h_global = self.temporal_model_global(
                    fused_flat, prev_states=prev_h_global, return_last_state=True,
                    identity=curr_identity,
                )

                self.memory_bank_local.update_state(next_h_local)
                self.memory_bank_global.update_state(next_h_global)

                # KAN Dynamic Temporal Fusion: gated mix of local (motion) and global (re-id)
                ssm_cat = torch.cat([out_local, out_global], dim=-1)  # [B*P, 1, D*2]
                ssm_gate_flat = torch.sigmoid(self.temporal_gate(ssm_cat.reshape(-1, D * 2)))
                ssm_gate = ssm_gate_flat.view(B * P, 1, D)
                ssm_fused = (ssm_gate * out_global) + ((1 - ssm_gate) * out_local)

                # ── DTSM slow consolidation path ─────────────────────────────────
                # Persistent causal SSM carries long-range context across frames.
                # Injected as a residual correction — no attention / softmax.
                if self.use_dual_timescale_memory:
                    prev_h_slow = self.memory_bank_slow.get_state()
                    out_slow, next_h_slow = self.temporal_model_slow(
                        fused_flat, prev_states=prev_h_slow, return_last_state=True
                    )
                    self.memory_bank_slow.update_state(next_h_slow)
                    ssm_fused = ssm_fused + self.slow_residual_proj(out_slow)

                # [B*P, 1, D] → [B, 1, D, P] for decoder
                last_feat = (
                    ssm_fused.squeeze(1)   # [B*P, D]
                    .reshape(B, P, D)      # [B, P, D]
                    .permute(0, 2, 1)      # [B, D, P]
                    .unsqueeze(1)          # [B, 1, D, P]
                )

                # ── 6. Hierarchical decoding ─────────────────────────────────────
                frame_ms = self._build_dec_ms(query_features_raw, t)
                logits_t = self.seg_decoder(
                    last_feat, frame_ms, prev_mask=prev_guide_mask
                )  # [B, 1, num_classes, H, W]
                all_preds_seg.append(logits_t)

                # ── Update propagation state for next frame ───────────────────────
                pred_soft = torch.softmax(logits_t, dim=2)  # [B, 1, C, H, W]
                prev_guide_mask = (
                    pred_soft[:, 0, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0).detach()
                )  # [B, 1, H, W]

                # Integer predicted mask for identity encoding next frame.
                # Resize to input resolution so patch_wise_id_bank strides correctly.
                prev_mask_int = logits_t.squeeze(1).argmax(dim=1).detach()  # [B, lH, lW]
                if prev_mask_int.shape[-2:] != (H, W):
                    prev_mask_int = F.interpolate(
                        prev_mask_int.unsqueeze(1).float(), size=(H, W), mode="nearest"
                    ).squeeze(1).long()

                # Previous-frame features for fusion next iteration (detached).
                prev_patch = curr_patch.detach()

            # Persist end-of-clip state so the next forward() call can carry it
            # across when reset_memory=False (sliding-window long-video eval).
            self._carry_prev_mask_int = prev_mask_int.cpu()
            self._carry_prev_patch = prev_patch.cpu()

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
                    "stage_3_hw": query_features_raw.get("stage_2_hw"),
                    "stage_2_hw": query_features_raw.get("stage_1_hw"),
                    "stage_1_hw": query_features_raw.get("stage_0_hw"),
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

        if prefix == "train" and batch_idx == 0:
            self.log(
                "train_scheduled_sampling_rate",
                self._get_scheduled_sampling_rate(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=ref_img.shape[0],
            )

        # Forward pass with reference memory and ground truth masks (for scheduled sampling)
        _logits_clf, _pred_boxes, _pred_box_logits, logits_seg = self(
            query_images, ref_frame=ref_img, ref_mask=ref_mask, query_masks=query_masks
        )
        # logits_seg: [B, T, num_seg_classes, H, W]

        # Pad obj_present from [B, T, n_data] to [B, T, n_model] when the dataset has
        # fewer object slots than the model (e.g. BL30K single-object vs DAVIS 10-object).
        # Extra channels are False (0.0), so the loss masks them out automatically.
        n_model = logits_seg.shape[2] - 1  # n_id = C - 1 (background excluded)
        if obj_present.shape[2] < n_model:
            pad = torch.zeros(
                obj_present.shape[0], obj_present.shape[1],
                n_model - obj_present.shape[2],
                dtype=obj_present.dtype, device=obj_present.device,
            )
            obj_present = torch.cat([obj_present, pad], dim=2)

        loss = self.vos_loss_fn(logits_seg, query_masks, obj_present)

        assert not torch.isnan(loss), "_vos_step: NaN loss detected."

        # Per-sample CE proxy for cv-tracking callbacks (PerSampleLossTrajectoryTracker,
        # TRAKInfluenceCallback).  Averaged over T, H, W per sample in the batch.
        # Shape: [B] — detached so the computation graph is not retained.
        if prefix == "train":
            B, T, C, H, W = logits_seg.shape
            per_sample_loss = torch.nn.functional.cross_entropy(
                logits_seg.view(B * T, C, H, W),
                query_masks.view(B * T, H, W).long(),
                ignore_index=255,
                reduction="none",
            )  # [B*T, H, W]
            self.last_per_sample_losses = (
                per_sample_loss.view(B, T, H, W).mean(dim=(1, 2, 3)).detach()
            )  # [B]

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
        label_smoothing_clf = 0.1 if prefix == "train" else 0.0
        label_smoothing_seg = 0.0

        # ── Multi-object VOS (YouTube-VOS / MOSE) ────────────────────────
        if self._is_vos_batch(batch):
            return self._vos_step(batch, batch_idx, prefix)

        # Handle DAVIS Semi-Supervised format: (ref_img, ref_mask, query_images, query_masks, seq_names)
        # seq_names (list[str]) is passed through for spike diagnostics but not used in the loss.
        if len(batch) == 5 and batch[0].dim() == 4 and batch[1].dim() == 3 and batch[2].dim() == 5:
            ref_img, ref_mask, query_images, query_masks, _seq_names = batch
            bs = ref_img.shape[0]
            logits_clf, pred_boxes, pred_box_logits, logits_seg = self(
                query_images, ref_frame=ref_img, ref_mask=ref_mask, query_masks=query_masks
            )
            
            BT, C, H, W = logits_seg.shape[0] * logits_seg.shape[1], logits_seg.shape[2], logits_seg.shape[3], logits_seg.shape[4]
            loss_seg = F.cross_entropy(
                logits_seg.view(BT, C, H, W), 
                query_masks.view(BT, H, W), 
                ignore_index=255,
                label_smoothing=label_smoothing_seg
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
            loss_clf = F.cross_entropy(logits_clf, labels, label_smoothing=label_smoothing_clf)
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
            loss_seg = F.cross_entropy(logits_seg.view(BT, C, H, W), masks.view(BT, H, W), label_smoothing=label_smoothing_seg)
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
                label_smoothing=label_smoothing_clf
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
        learning_rate = self.hparams.learning_rate

        # ── Temporal SSM: KangaSSM (Local + Global) ───────────────────
        # Separate from "base" so temporal_lr_multiplier can throttle it.
        temporal_params = (
            list(self.temporal_model_local.parameters())
            + list(self.temporal_model_global.parameters())
            + list(self.ssm_output_fusion.parameters())
        )
        if self.use_dual_timescale_memory:
            temporal_params = temporal_params + (
                list(self.temporal_model_slow.parameters())
                + list(self.slow_residual_proj.parameters())
            )
        temporal_ids = {id(p) for p in temporal_params}

        # ── Encoder backbone ─────────────────────────────────────────
        backbone_params = list(self.feature_extractor.parameters())
        backbone_ids = {id(p) for p in backbone_params}

        # ── Decoder + everything else ────────────────────────────────
        base_params = [
            p for p in self.parameters()
            if id(p) not in temporal_ids
            and id(p) not in backbone_ids
        ]

        trainable_backbones = {"mobilenetv2", "vision_mamba_tiny"}
        if self.hparams.encoder_type in trainable_backbones:
            backbone_lr = learning_rate * self.hparams.trainable_backbone_lr_multiplier
        else:
            backbone_lr = learning_rate * self.hparams.frozen_backbone_lr_multiplier

        param_groups = [
            {
                "params": temporal_params,
                "lr": learning_rate * self.hparams.temporal_lr_multiplier,
                "name": "temporal",
            },
            {"params": base_params, "lr": learning_rate, "name": "base"},
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        ]

        # ── Optimizer ──────────────────────────────────────────────────
        if "opt" in self.hparams and "optimizer" in self.hparams.opt:
            import hydra
            from functools import partial
            opt_cfg = self.hparams.opt.optimizer
            if isinstance(opt_cfg, partial):
                optimizer = opt_cfg(params=param_groups)
            else:
                optimizer = hydra.utils.instantiate(opt_cfg, params=param_groups)
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.hparams.optimizer_weight_decay,
            )

        # ── LR Scheduler ───────────────────────────────────────────────
        if self.trainer is not None and getattr(self.trainer, "max_epochs", None):
            t_max = self.trainer.max_epochs
        else:
            t_max = self.hparams.max_epochs

        if "opt" in self.hparams and "lr_scheduler" in self.hparams.opt:
            import hydra
            from functools import partial
            sched_cfg = self.hparams.opt.lr_scheduler
            
            # For OneCycleLR and similar, we need to resolve total_steps
            is_onecycle = "OneCycleLR" in getattr(sched_cfg, "_target_", "") or (isinstance(sched_cfg, partial) and "OneCycleLR" in str(sched_cfg.func))
            
            if is_onecycle:
                # Calculate total steps if not provided
                if self.trainer is not None:
                    try:
                        dataset_size = len(self.trainer.datamodule.train_dataloader())
                    except Exception:
                        dataset_size = 1000
                    total_steps = dataset_size * t_max
                else:
                    total_steps = 1000  # Fallback
                
                if isinstance(sched_cfg, partial):
                    scheduler = sched_cfg(optimizer=optimizer, total_steps=total_steps)
                else:
                    scheduler = hydra.utils.instantiate(
                        sched_cfg, 
                        optimizer=optimizer,
                        total_steps=total_steps
                    )
            else:
                if isinstance(sched_cfg, partial):
                    scheduler = sched_cfg(optimizer=optimizer)
                else:
                    scheduler = hydra.utils.instantiate(sched_cfg, optimizer=optimizer)
            
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step" if "OneCycleLR" in str(type(scheduler)) else "epoch",
                "frequency": 1,
            }
        else:
            warmup_epochs = max(0, int(getattr(self.hparams, "lr_warmup_epochs", 0)))
            warmup_start_factor = float(getattr(self.hparams, "lr_warmup_start_factor", 0.1))

            cosine_t_max = max(1, t_max - warmup_epochs)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_t_max,
                eta_min=self.hparams.scheduler_eta_min,
            )

            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = cosine_scheduler
            
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
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
