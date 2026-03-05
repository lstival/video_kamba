# Ablation Experiments — Interpretation Guide

This document describes the three empirical experiments that validate the claims
made in the paper about the KAN-based decoder and temporal modulation.
It explains what each experiment measures, how to run it, what outputs to
expect, and how to read the results.

---

## Experiment 1 — Activation Entropy (Disentanglement Proof)

### What It Measures

Tests whether the KAN temporal gate produces **sparser, more disentangled
representations** than an equivalent MLP gate.

The temporal gate $\alpha_t$ corresponds to the B-modulation tensor computed by
`FastKANModulator` inside `IntricateKANSSMCore`. Its shape is `[B·T, D]` where
`D=768` is the feature dimension. For each sample, we treat the D-dimensional
vector as a probability distribution and compute its Shannon entropy.

### Formula

$$
\hat{\alpha}_{t,i} = \frac{|\alpha_{t,i}|}{\sum_j |\alpha_{t,j}|},
\qquad
H(\alpha_t) = -\sum_{i=1}^{D} \hat{\alpha}_{t,i} \log(\hat{\alpha}_{t,i})
$$

| Entropy | Meaning |
|---|---|
| **Low** | Activations concentrated on few dimensions → sparse, disentangled |
| **High** | Activations spread uniformly → dense, entangled (typical MLP) |

### Running

```bash
sbatch scripts/slurm_exp1.sh \
    kan_ckpt=lightning_logs/version_kan/checkpoints/best.ckpt \
    mlp_ckpt=lightning_logs/version_mlp/checkpoints/best.ckpt
```

### Outputs

| File | Description |
|---|---|
| `results/exp1/exp1_entropy.pdf` | Overlapping histogram of H values for KAN vs MLP across all DAVIS validation frames |
| `results/exp1/exp1_entropy.csv` | Per-sample entropy values with model label |
| `results/exp1/exp1_summary.txt` | Mean ± std, Welch t-test, Mann-Whitney U statistic |

### How to Interpret

- The **KAN histogram should be left-skewed** (concentrated near low entropy),
  showing that only a sparse subset of channels fires per decision.
- The **MLP histogram should be right-skewed or uniform** (spread toward higher
  entropy), reflecting dense, overlapping neuron activations.
- A statistically significant difference in means (`p < 0.05` in both tests)
  confirms that KAN isolates semantic concepts into distinct channels, obeying
  the **Principle of Simplicity**.
- If KAN entropy is **not** lower: check the L1 sparsity regularisation weight
  or verify the model has fully converged.

---

## Experiment 2 — Spatial Gate Fidelity (IoU_gate, Geometric Proof)

### What It Measures

Tests whether the spatial gate $\mathbf{G}_k$ from `KANSpatialGatingUpBlock`
**geometrically aligns with the ground-truth object mask** — without any
post-hoc supervision on the gate itself.

The gate has shape `[B·T, C_skip, H, W]`. To produce a 2D foreground map,
we take the channel-wise mean, normalise to [0, 1], then threshold at $\tau$.
The cross-attention baseline uses averaged attention weights processed
identically.

### Formula

$$
\text{IoU}_{\text{gate}}
= \frac{|\{\bar{\mathbf{G}}_k > \tau\} \cap M|}{|\{\bar{\mathbf{G}}_k > \tau\} \cup M|}
$$

where $\bar{\mathbf{G}}_k = \frac{1}{C}\sum_c \mathbf{G}_{k,c}$ and
$M \in \{0,1\}^{H \times W}$ is the binary object mask.

The final score is the **arithmetic mean over all three pyramid levels**
(up1 = 14×14, up2 = 28×28, up3 = 56×56).

### Running

```bash
sbatch scripts/slurm_exp2.sh \
    kan_ckpt=lightning_logs/version_kan/checkpoints/best.ckpt \
    xattn_ckpt=lightning_logs/version_xattn/checkpoints/best.ckpt
```

### Outputs

| File | Description |
|---|---|
| `results/exp2/exp2_iou_gate.pdf` | Side-by-side boxplots of per-frame IoU_gate for KAN vs cross-attention |
| `results/exp2/exp2_iou_gate.csv` | Per-frame, per-level IoU values for both models |
| `results/exp2/exp2_summary.txt` | Mean ± std and Mann-Whitney U effect-size test |
| `results/exp2/visualizations/` | Per-sequence gate heatmap overlay on video frame |

### How to Interpret

- **Higher mean IoU_gate for KAN** proves that $\mathbf{G}_k$ learns object
  geometry as a structural necessity, not incidentally.
- **Lower IoU_gate for cross-attention** means attention maps are diffuse and
  do not align with object boundaries — supporting the "blind mixing" claim.
- Qualitative overlays in `visualizations/` should show the gate heatmap
  forming a tight blob around the segmented object. Use these in the paper figure.
- If IoU_gate ≤ 0.05 for all models: check that GT masks and gate tensors
  are being bilinearly / nearest-neighbour resized to the same resolution
  before binarisation.

---

## Experiment 3 — 1D RBF Activation Profile (Qualitative Interpretability Proof)

### What It Measures

Visualises the **learned univariate basis functions** inside `FastKANLayer`,
which is the core of the `FastKANModulator` used in the temporal gate.
Each KAN edge learns a 1D function parameterised as:

$$
f_{o,i}(x)
= \sum_{g=1}^{G} w_{o,i,g} \cdot \exp\!\left(-\frac{(x-\mu_g)^2}{h^2}\right)
+ b_i \cdot \text{SiLU}(x)
$$

where:

| Symbol | Meaning | Shape |
|---|---|---|
| $w_{o,i,g}$ | Learnable `rbf_weight` | `[out, in, G]` |
| $\mu_g$ | Fixed grid centroids (`grid` buffer) in `[-2, 2]` | `[G]` |
| $h = 4/G$ | Fixed bandwidth (0.5 for G=8) | scalar |
| $b_i$ | Residual linear weight (`base_weight`) | `[out, in]` |

### What "Channel" Means

A channel $c$ refers to one input dimension of the temporal gate (one of the
768 DINOv2 feature dimensions). The script **auto-discovers** the top-N
channels with the highest mean absolute activation across the DAVIS validation
set — no manual channel selection needed.

### Running

```bash
sbatch scripts/slurm_exp3.sh \
    kan_ckpt=lightning_logs/version_kan/checkpoints/best.ckpt
```

### Outputs

| File | Description |
|---|---|
| `results/exp3/exp3_rbf_profiles.pdf` | Multi-panel figure: learned KAN curve (solid blue) vs ReLU reference (dashed red) |
| `results/exp3/exp3_top_channels.csv` | All channels ranked by mean \|activation\|, with index and importance score |
| `results/exp3/exp3_weight_heatmap.pdf` | Heatmap of `rbf_weight[:, top_channel, :]` — which grid bins carry the most weight |

### How to Interpret

| Curve shape | Meaning |
|---|---|
| **Peaked / band-pass** | KAN learned a specific filter tuned to a feature range → strong interpretability claim |
| **Multi-modal (two peaks)** | Channel detects two distinct semantic regimes (e.g., occluded vs. fast-moving) |
| **Near-linear / monotone** | Channel behaves like a standard linear layer — valid null finding |

The **ReLU reference** (dashed red) is always piecewise-linear with a single
kink at 0.  Contrast with the KAN curve for the paper argument:

> *"Unlike MLPs which rely on static hyperplanes (ReLU), the KDSM module
> explicitly molds its basis functions to the latent distribution of the
> DINOv2 features."*

---

## Shared Notes

### Checkpoint Naming Convention

| Model | Config | Expected Path |
|---|---|---|
| KAN (full) | `model=default` | `lightning_logs/version_kan/checkpoints/best.ckpt` |
| MLP baseline | `model=mlp_baseline` | `lightning_logs/version_mlp/checkpoints/best.ckpt` |
| Cross-attention | `fusion_mode=cross_attn` | `lightning_logs/version_xattn/checkpoints/best.ckpt` |

### Training the MLP Baseline

```bash
sbatch scripts/slurm_mlp_baseline.sh
```

This uses `configs/model/mlp_baseline.yaml` which sets
`fusion_mode: concat` and `modulator_type: mlp`.

### Reproducibility

All scripts accept a `seed` override:

```bash
python scripts/exp1_activation_entropy.py -m seed=0,1,2 \
    kan_ckpt=... mlp_ckpt=...
```

### Common Failure Modes

| Symptom | Likely Cause | Fix |
|---|---|---|
| Entropy values identical for KAN and MLP | Hook not attached to the right submodule | Print `model.named_modules()` and verify `b_mod_layer_idx` |
| IoU_gate ≈ 0 for all frames | Mask and gate at different resolutions | Ensure GT mask is nearest-neighbour resized to pyramid level resolution |
| RBF curves all flat | `rbf_weight` norms near zero (collapsed) | Check L1 regularisation weight and inspect training loss |
| CUDA OOM during analysis | Batch size too large | Set `batch_size=1` in the script or YAML config |
| `AttributeError: no 'kan' attribute` in Exp 3 | Checkpoint was trained with `modulator_type=mlp` | Use a KAN-trained checkpoint for Exp 3 |
