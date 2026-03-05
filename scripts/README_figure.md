# Professional Qualitative Figure Generator

This script automates the creation of standardized Figures for Video Object Segmentation papers, similar to those seen in CVPR/NeurIPS.

## Usage

Run the script using the project's virtual environment:

```bash
venv/bin/python3 scripts/generate_paper_figure.py \
    --davis_visuals eval_results/top_5_visuals_20260305_092529 \
    --ytb_visuals eval_results/top_5_visuals_20260305_092739 \
    --output docs/qualitative_figure.png
```

## Features

- **Automated Selection**: Automatically picks start, middle, and end frames from result GIFs.
- **Result-Driven**: Can be updated to parse `metrics.json` to find the highest-performing sequences automatically.
- **Scientific Layout**: Includes Frame labels (e.g., `t+0`, `t+8`), column headers, and Ground Truth comparison slots.
- **Professional Formatting**: Uses academic-style spacing and font rendering.

## Requirements

The script uses `Pillow` which is available in `venv`.
