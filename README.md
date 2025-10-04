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

- PyTorch 2.7.1+cu118
- Torchvision 0.22.1+cu118
- NumPy 2.2.6
- OpenCV 4.12.0

## Project Structure

```
Major Project/
├── README.md
├── .gitignore
├── src/           # Source code
├── data/          # Dataset files
├── models/        # Trained models
├── configs/       # Configuration files
└── scripts/       # Training and evaluation scripts
```