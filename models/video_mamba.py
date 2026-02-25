import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from jaxtyping import Float

from models.components.dinov3_wrapper import DinoV3Wrapper
from models.components.kanga_ssm import KangaSSM
from models.components.classification_head import ClassificationHead
from models.components.segmentation_decoder import SegmentationDecoder

class VideoMambaSystem(L.LightningModule):
    def __init__(self, dim_in: int = 768, dim_out: int = 256, num_classes: int = 10, target_size: int = 224, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.feature_extractor = DinoV3Wrapper(freeze=True)
        self.temporal_model = KangaSSM(d_model=dim_in)
        self.clf_head = ClassificationHead(dim_in=dim_in, num_classes=num_classes)
        self.seg_decoder = SegmentationDecoder(dim_in=dim_in, num_classes=num_classes, target_size=target_size)
        
    def forward(self, x: Float[torch.Tensor, "B T C H W"]) -> tuple[Float[torch.Tensor, "B num_classes"], Float[torch.Tensor, "B T num_classes H W"]]:
        # 1. Spatial feature extraction (DinoV3)
        cls_tokens, patch_tokens = self.feature_extractor(x)
        
        # 2. Temporal modeling (KANGA SSM)
        # We contextualize the sequence of cls tokens
        ssm_out = self.temporal_model(cls_tokens)
        
        # 3. Heads
        # Classification uses the temporal context
        logits_clf = self.clf_head(ssm_out)
        
        # Segmentation uses the patch tokens 
        # (in advanced setups, you'd infuse temporal context into patches, here we decode directly to start)
        # Wait, if we use just patch_tokens, it ignores the temporal model. 
        # Let's multiply the patch tokens by the ssm_out weights or just concatenate to infuse temporal context.
        # For simplicity in this dummy, we just add the temporal context broadcasted over patches.
        B, T, D, P = patch_tokens.shape
        ssm_context = ssm_out.unsqueeze(-1) # [B, T, D, 1]
        infused_patches = patch_tokens + ssm_context
        
        logits_seg = self.seg_decoder(infused_patches)
        
        return logits_clf, logits_seg

    def _shared_step(self, batch, batch_idx, prefix="train"):
        frames, labels, masks = batch
        
        # Forward pass
        logits_clf, logits_seg = self(frames)
        
        # Loss calculation
        loss_clf = F.cross_entropy(logits_clf, labels)
        
        # Masks: [B, T, H, W], logits_seg: [B, T, C, H, W]
        # Reshape to 1D for CE loss
        B, T, C, H, W = logits_seg.shape
        loss_seg = F.cross_entropy(logits_seg.view(-1, C), masks.view(-1).long())
        
        loss = loss_clf + loss_seg
        
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_loss_clf", loss_clf)
        self.log(f"{prefix}_loss_seg", loss_seg)
        
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
