# MSCF-Net: Multi-Scale Texture-Prior and Sparse Cross-Attention Fusion for Automated SPECT/CT Bone Metastasis Segmentation

## Introduction

MSCF-Net is a dual-encoder medical image segmentation network designed for SPECT/CT bone metastasis segmentation. It leverages the complementary information from SPECT (functional/metabolic) and CT (structural/anatomical) modalities through a novel multi-scale cross-attention fusion mechanism.

## Installation

```bash
cd C:\Users\75413\Desktop\MSFC
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- torchvision >= 0.13
- numpy

## Data Preparation

The dataset should be organized as follows:

```
dataset/
├── images/
│   ├── train/          # Training images 
│   └── test/           # Test images
└── masks/
    ├── train/          # Training masks (same filenames as images)
    └── test/           # Test masks
```

### Data Format

- Images: NumPy arrays with shape `(H, W, 2)` where channel 0 is SPECT and channel 1 is CT
- Masks: NumPy arrays with shape `(H, W)` or `(H, W, 1)` with binary values (0 or 1)

### Preprocessing

Same preprocessing is applied to both retrospective dataset and BM-Seg dataset:
- Resize to 256×256
- Normalize SPECT to [0, 1] based on global max
- Normalize CT to [0, 1] based on HU range [-1000, 1000]

## Training

```bash
python train.py
```

### Default Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | Adam | |
| Learning Rate | 1e-4 | |
| Batch Size | 4 | |
| Epochs | 100 | |
| Loss Function | Weighted Dice + Cross-Entropy | Dice weight=0.5, CE weight=0.5 |
| GPU | NVIDIA 4090D | |

### Custom Training

Edit `train.py` to change parameters:

```python
train_model(
    data_dir="dataset",      # Path to dataset
    epochs=100,             # Number of epochs
    batch_size=4,           # Batch size
    lr=0.0001,             # Learning rate
    device="cuda",           # Device (cuda/cpu)
)
```

## Testing

```bash
python test.py
```

### Output Metrics

The test script computes and reports:
- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Measures segmentation accuracy
- **Precision**: Measures positive predictive value
- **Recall**: Measures sensitivity

### Custom Testing

Edit `test.py` to change parameters:

```python
test_model(
    data_dir="dataset",          # Path to dataset
    weight_path="mscf_net.pth", # Path to trained weights
    batch_size=1,                # Batch size (recommend 1 for testing)
    device="cuda",               # Device (cuda/cpu)
)
```

## File Structure

```
C:\Users\75413\Desktop\MSFC\
├── MSCF_Net.py       # Model definition (MSCF_Net, _MSCA, CBAM, MTAB, etc.)
├── train.py           # Training script
├── test.py            # Testing script
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Model Definition

The main model is defined in `MSCF_Net.py`:

```python
class MSCF_Net(nn.Module):
    def __init__(self, in_chans=1, out_chans=1,
                 dims=[48, 96, 192, 384],
                 depths=[2, 2, 2, 2],
                 dropout=0.0):
        # ...
```

### Network Parameters

- `in_chans`: Number of input channels per modality (default: 1)
- `out_chans`: Number of output channels (default: 1 for binary segmentation)
- `dims`: Feature dimensions at each stage
- `depths`: Number of MTABs at each stage
- `dropout`: Dropout rate before final convolution

## Training Details

### Loss Function

The `WeightedDiceCELoss` combines Dice loss and Cross-Entropy loss:

```python
loss = weight_dice * dice_loss + weight_ce * ce_loss
```

Default weights: `weight_dice=0.5`, `weight_ce=0.5`

### Validation

During training, the model is evaluated on the validation set after each epoch. The validation metrics (loss and Dice coefficient) are printed.

### Checkpoint Saving

The model weights are saved as `mscf_net.pth` after training completes.

## Testing Details

### Metrics Computation

- **Dice**: `2 * |A ∩ B| / (|A| + |B|)`
- **IoU**: `|A ∩ B| / |A ∪ B|`
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`

where A is the prediction and B is the ground truth.

### Results

The test script prints per-image metrics and summary statistics (mean ± std).

## Citation

If you use this code or our model in your research, please cite our paper:

```bibtex
@article{XXX,
  title={MSCF-Net: Multi-Scale Texture-Prior and Sparse Cross-Attention Fusion for Automated SPECT/CT Bone Metastasis Segmentation},
  author={},
  journal={},
  year={},
  volume={},
  pages={}
}
```
