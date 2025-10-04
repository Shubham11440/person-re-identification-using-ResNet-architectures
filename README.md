# CPR ReID Project

Person Re-identification project using PyTorch.

## Environment Setup

This project uses conda for environment management.

### Prerequisites

- Miniconda or Anaconda
- CUDA-capable GPU (for GPU acceleration)

### Installation

1. Create conda environment:

```bash
conda create --name cpr_reid python=3.9 -y
```

2. Activate environment:

```bash
conda activate cpr_reid
```

3. Install PyTorch with CUDA support:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. Install additional dependencies:

```bash
python -m pip install numpy opencv-python
```

## Usage

Activate the conda environment before running any scripts:

```bash
conda activate cpr_reid
```

## Dependencies

- Python 3.9.23
- PyTorch 2.7.1+cu118
- Torchvision 0.22.1+cu118
- NumPy 2.0.2
- OpenCV 4.12.0

## Project Structure

```
Major Project/
├── README.md
├── .gitignore
├── environment.yml    # Conda environment specification
├── configs/          # Configuration files
├── data/            # Dataset files (to be added)
├── notebooks/       # Jupyter notebooks for exploration
├── src/            # Source code
│   ├── datasets.py  # Dataset loading and preprocessing
│   ├── models.py    # Neural network models
│   ├── losses.py    # Loss functions
│   ├── train.py     # Training script
│   ├── evaluate.py  # Evaluation script
│   └── utils.py     # Utility functions
├── scripts/         # Training and evaluation scripts
└── results/         # Training results and outputs
```

## Getting Started

1. **Environment Setup**: Follow the installation steps above
2. **Data Preparation**: Add your person re-identification datasets to the `data/` directory
3. **Configuration**: Modify training parameters in the source files or create config files
4. **Training**: Run training scripts from the `src/` directory
5. **Evaluation**: Use evaluation scripts to assess model performance

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules for datasets, models, losses, and utilities
- **Multiple Loss Functions**: Support for triplet loss, cross-entropy loss, and combined loss functions
- **Flexible Model Architecture**: ResNet-based models with easy extensibility
- **Comprehensive Evaluation**: Ranking metrics including mAP, CMC, and top-k accuracy
- **Training Utilities**: Learning rate scheduling, checkpointing, and progress tracking

## Next Steps

1. Add your person re-identification dataset to the `data/` directory
2. Configure dataset paths and training parameters
3. Start training with `python src/train.py`
4. Monitor training progress and evaluate results
