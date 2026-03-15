# Training Protocol — vision_mamba_tiny_sota_light

## Architecture
- Model: `vision_mamba_tiny_sota_light` (DiagonalKANSSMCore, `ssm_layers=2`)
- ALL checkpoints anteriores (IntricateKANSSMCore) são **incompatíveis**

---

## Sequência de Submissão

### Fase 1 — COCO Pretrain (~15h)
```bash
sbatch scripts/pretrain_coco_sota_light.sh
```
- Do zero, sem checkpoint
- 20 epochs, `limit_train_batches=3000`, `lr=1e-4`
- Output: `checkpoints/best_coco_sota_light.ckpt`
- Critério de sucesso: `val_loss < 0.5` no epoch 5

### Fase 2 — DAVIS Fast Validate (~16h)
```bash
sbatch scripts/ft_davis_fast_validate.sh
```
- Inicia da Fase 1
- DAVIS-only (301 clips), 100 epochs, `limit_train_batches=500` (~12 min/epoch)
- Output: `checkpoints/best_davis_sota_light.ckpt`
- **GATE @ epoch 50**: verificar no Comet (`ft_davis_fast_validate_v1`)
  - `val_J_and_F > 0.55` → ✅ submeter Fase 3
  - `val_J_and_F < 0.55` → ❌ investigar antes de continuar

### Fase 3 — Full YTB+DAV Fine-tune (~71h) [só se gate passou]
```bash
sbatch scripts/ft_coco_ytb_dav_50ep.sh
```
- Inicia da Fase 1 (`best_coco_sota_light.ckpt`)
- YouTube-VOS + DAVIS joint, 50 epochs, `limit_train_batches=3500`
- Comet run: `ft_coco_ytbdav_50ep_sota_light_v3`
- Objetivo final: `val_J_and_F > 0.65`
  (referência: MobileVOS CVPR 2023 ~78% J&F, <5M params)

---

## Resumo de tempo

| Fase | Job | Tempo |
|------|-----|-------|
| 1 | `pretrain_coco_sota_light.sh` | ~15h |
| 2 | `ft_davis_fast_validate.sh` | ~16h |
| **Gate decision** | Comet @ epoch 50 | — |
| 3 | `ft_coco_ytb_dav_50ep.sh` | ~71h |
| **Total (sem Fase 3)** | | **~31h** |
| **Total (com Fase 3)** | | **~102h** |

---

## Critérios de sucesso por fase

| Fase | Epoch | Métrica | Mínimo |
|------|-------|---------|--------|
| COCO pretrain | 5 | `val_loss` | < 0.5 |
| DAVIS validate | 20 | `val_J_and_F` | > 0.25 |
| DAVIS validate | **50** | `val_J_and_F` | **> 0.55 (gate)** |
| YTB+DAV | 20 | `val_J_and_F` | > 0.60 |
| YTB+DAV | 50 | `val_J_and_F` | > 0.65 |
