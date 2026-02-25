# Video Mamba Project Plan

## Acknowledgement
I am acting as a Senior Machine Learning & Software Engineer. I have reviewed and will strictly adhere to the production-ready coding standards defined in `.cursorrules`.

## Objective
To develop a video understanding pipeline that leverages strong spatial feature extractors and state-of-the-art temporal sequence models. The architecture is primarily inspired by the KANGA project (experiment_5 branch) and integrates DinoV3 for feature extraction.

## Architecture Design
Based on the provided conceptual diagram, the model follows a sequential processing pipeline:

1. **Input Stage**: A sequence of video frames $F_0, F_1, \dots, F_N$.
2. **Spatial Feature Extraction**: 
   - Each frame is passed independently through **DinoV3**.
   - Output: A sequence of spatial embeddings $E_0, E_1, \dots, E_N$ where each $E_i$ represents the dense patch tokens or CLS token of frame $i$. Let's assume tensor shapes: `[B, T, C, H_feat, W_feat]` or `[B, T, C]`.
3. **Temporal Sequence Modeling (SSM/KANGA)**: 
   - The spatial features are fed into a State Space Model (KANGA base).
   - Output: Contextualized temporal embeddings $O_0, O_1, \dots, O_N$ with shape `[B, T, C_out]`.
4. **Multi-Task Heads**:
   - **Task 1: Video Classification**: Pools the SSM outputs to perform a video-level prediction.
   - **Task 2: Per-Frame Segmentation**: Uses the per-frame SSM outputs $O_i$ to generate segmentation masks for each corresponding frame $F_i$.

## Engineering & Implementation Plan

Consistent with the `video_mamba` environment and `.cursorrules`:

### 1. Directory Structure
```text
video_mamba/
├── configs/                 # Hydra configs (datamodule, model, trainer, callbacks)
├── data/                    # Datasets and PyTorch-Lightning DataModules
├── models/
│   ├── components/          # Reusable modules (DinoV3Wrapper, KangaSSM, Heads)
│   └── video_mamba.py       # Main PyTorch-Lightning Module tying components together
├── eval/                    # Standalone evaluation scripts
├── utils/                   # Shared utilities (logging, metric calculations, etc.)
└── train.py                 # Main training script (Hydra entrypoint)
```

### 2. Component Design
- **`DinoV3Wrapper`**: Loads DinoV3, processes frames, and handles shape transformations to extract intermediate features.
- **`KangaSSM`**: The temporal block tracking `experiment_5` from the KANGA repository. Must support sequences of shapes `[B, T, ...]`.
- **`ClassificationHead`**: Maps gathered temporal context `[B, C]` to `[B, num_classes]`.
- **`SegmentationDecoder`**: Maps frame-aligned contexts `[B, T, C, H, W]` to segmentation logits `[B, T, num_classes, H_img, W_img]`.
- **`LightningSystem`**: The core `pytorch_lightning.LightningModule` that orchestrates forward passes, computes losses (CrossEntropy for classification, Dice/CE for segmentation), and logs to Comet.

### 3. Step-by-Step Execution
1. **Environment Setup**: Ensure `video_mamba` WSL environment has PyTorch 2.0+, Hydra, Lightning, and Comet.
2. **Data Pipeline**: Implement DataModules for the video dataset (handling frame extraction, transformations, and segmentation mask loading).
3. **Model Implementation**: 
   - Implement `DinoV3Wrapper` with clear typed tensors (e.g. using `jaxtyping`).
   - Port/Adapt KANGA SSM core.
   - Build dual-headed architecture.
4. **Validation**: Write `pytest` cases to verify tensor shape flow from input `[B, T, C, H, W]` to output losses.
5. **Training Logging**: Integrate `Comet` via PyTorch-Lightning loggers.