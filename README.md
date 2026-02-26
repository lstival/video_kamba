# Video Mamba: Efficient Video Understanding

A PyTorch Lightning-based framework for video analysis tasks using Mamba-based architectures. This project supports video classification and object segmentation with configurable training pipelines via Hydra.

## 🚀 Features

- **Mamba-based Architectures**: Efficient modern state-space models for video processing.
- **Configurable Training**: Full integration with **Hydra** for easy experimentation.
- **PyTorch Lightning**: Scalable and reproducible training loops.
- **Multiple Tasks**:
  - Video Classification (e.g., HMDB51)
  - Video Object Segmentation (e.g., DAVIS, YouTube-VOS)
- **Auto Batch Size**: Automatic tuning of batch sizes for optimal GPU utilization.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Video_Mamba
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Data Preparation

Use the provided script to download and prepare datasets:

```bash
python download_datasets.py --dataset davis --path ./data/davis
python download_datasets.py --dataset hmdb51 --path ./data/hmdb51
```

## 🚂 Training

Start training using the `train.py` script. Configurations are managed in the `configs/` directory.

### Video Object Segmentation (DAVIS)
```bash
python train.py datamodule=davis
```

### Video Classification (HMDB51)
```bash
python train.py datamodule=hmdb51
```

### Customizing Training
You can override any configuration parameter from the command line:
```bash
python train.py trainer.max_epochs=50 model.lr=1e-4 auto_batch_size=True
```

## 🏗️ Project Structure

- `configs/`: Hydra configuration files for models, trainers, and datasets.
- `models/`: Implementation of Video Mamba and its components.
- `data/`: Dataset storage and loading logic.
- `scripts/`: Utility scripts for data processing and evaluation.
- `train.py`: Main entry point for model training.

## 🧪 Testing

Run smoke tests to verify the installation:
```bash
python smoke_test.py
```

## 📜 License
[MIT License](LICENSE)
