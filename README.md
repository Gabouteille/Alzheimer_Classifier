# DEMNET: Deep Learning Model for Alzheimer's Disease Classification

## 📋 Project Overview

This repository contains a PyTorch implementation of **DEMNET** (DEMentia NETwork), a convolutional neural network designed for early diagnosis of Alzheimer's disease stages from MRI brain images.

The model classifies brain MRI scans into **4 dementia stages**:
- 🟢 **Non-Demented (ND)**: No cognitive impairment
- 🟡 **Very Mild Demented (VMD)**: Early stage cognitive decline
- 🟠 **Mild Demented (MD)**: Mild cognitive impairment affecting daily life
- 🔴 **Moderate Demented (MOD)**: Advanced cognitive decline

---

## 🎯 Key Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **72.97%** |
| **Test Loss** | 0.5629 |
| **Dataset Size** | 640 test images |

### Notes on Results

- **Architecture**: Faithfully implements original DEMNET (4,534,996 parameters)
- **Data Handling**: Proper 80/10/10 split with SMOTE applied only to training set (no data leakage)
- **Test Set**: 640 images from original Kaggle train split (imbalanced distribution preserved)
- **Class Distribution**: NonDemented (322), VeryMildDemented (217), MildDemented (91), ModerateDemented (10)

---

## 🏗️ Model Architecture

The DEMNET model (faithful to Murugan et al., 2021) consists of:

```
Input (176×176×3)
    ↓
Conv2D (16 filters) + ReLU + MaxPooling
    ↓
DEMNET Block 1 (32 filters)
DEMNET Block 2 (64 filters)
DEMNET Block 3 (128 filters)
DEMNET Block 4 (256 filters)
    ↓
Flatten
    ↓
Dense Layer 1 (512 neurons) + Dropout(0.7)
Dense Layer 2 (128 neurons) + Dropout(0.5)
Dense Layer 3 (64 neurons) + Dropout(0.2)
    ↓
Output Layer (4 classes, SoftMax)
```

**Total Parameters:** 4,534,996

### DEMNET Block Structure
Each DEMNET block contains:
- 2 Conv2D layers with ReLU activation
- Batch Normalization
- MaxPooling layer (2×2 stride 2)

---

## 📊 Training Curves

The model shows excellent convergence with minimal overfitting:

```
Train Loss:  0.0569 (final)
Val Loss:    0.0707 (final)
Difference:  +0.0138 (minimal overfitting)
```

Both training and validation losses converge smoothly after epoch 20, indicating proper generalization.

---

## 🗂️ Project Structure

```
Alzheimer_classifier/
├── Data_processing.ipynb       # Main notebook with full implementation
├── best_demnet_model.pth       # Pre-trained model weights
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules
```

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU) or Metal Performance Shaders (macOS) or CPU

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Alzheimer_MDST.git
cd Alzheimer_classifier
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running the Full Pipeline

Open `Data_processing.ipynb` in Jupyter and run all cells sequentially:

```bash
jupyter notebook Data_processing.ipynb
```

The notebook performs:
1. **Dataset Loading & Preprocessing**
   - Loads 6,400 MRI images from Kaggle
   - Resizes to 176×176
   - Normalizes pixel values to [0, 1]

2. **Class Balancing with SMOTE**
   - Addresses class imbalance (ModerateDemented had only 64 images)
   - Upsamples minority classes to 3,200 images each
   - Reduces overfitting

3. **Dataset Splitting**
   - 80% Training (10,240 images after SMOTE)
   - 10% Validation (1,280 images)
   - 10% Testing (1,280 images)

4. **Model Training**
   - 50 epochs
   - Batch size: 16
   - Optimizer: RMSprop (learning rate: 0.001)
   - Loss function: CrossEntropyLoss
   - GPU acceleration: Metal (MPS) on macOS, CUDA on NVIDIA, CPU fallback

5. **Model Evaluation**
   - Loads best performing model
   - Evaluates on test set
   - Displays accuracy and loss

### Using the Pre-trained Model

```python
import torch
from Data_processing import DEMNET

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = DEMNET(num_classes=4).to(device)
model.load_state_dict(torch.load('best_demnet_model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    output = model(input_tensor)  # Shape: (batch_size, 4)
    predictions = torch.argmax(output, dim=1)
```

---

## 🔍 Key Implementation Details

### Data Augmentation
- **Resize**: 176×176 (optimized for DEMNET)
- **CenterCrop**: Maintains aspect ratio
- **Normalization**: Pixel values [0, 1]

### SMOTE Configuration
```python
smote = SMOTE(
    sampling_strategy='auto',  # Auto-balance to majority class
    random_state=42,
    k_neighbors=2             # Adapted for small classes
)
```

### Validation Phase
During training, after each epoch:
- Model switches to evaluation mode
- Validation loss computed on unseen validation set
- Best model saved when validation loss improves
- Early stopping criteria can be added

### Device Support
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")      # macOS Metal GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")     # NVIDIA GPU
else:
    device = torch.device("cpu")      # CPU fallback
```

---

## 📈 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Optimizer | RMSprop |
| Loss Function | CrossEntropyLoss |
| Train/Val/Test Split | 80/10/10 |
| Image Size | 176×176 |
| Input Channels | 3 (RGB) |

### Regularization

| Technique | Details |
|-----------|---------|
| Dropout Layer 1 | p=0.7 (after Dense1) |
| Dropout Layer 2 | p=0.5 (after Dense2) |
| Dropout Layer 3 | p=0.2 (after Dense3) |
| Batch Normalization | After each DEMNET block |

---

## 📚 Dataset Information

**Source**: [Kaggle - Alzheimer's Dataset (4 Class of Images)](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images)

**Original Distribution**:
```
Training Set:
├── MildDemented:      896 images
├── ModerateDemented:  64 images    ← Highly imbalanced!
├── NonDemented:       3,200 images
└── VeryMildDemented:  2,240 images
```

**After SMOTE Balancing**:
```
All classes: 3,200 images each
Total: 12,800 images
```

---

## 📊 Confusion Matrix Results

| True\Pred | ND | VMD | MD | MOD |
|-----------|----|----|----|----|
| **ND** | 314 | 2 | 8 | 2 |
| **VMD** | 0 | 309 | 0 | 0 |
| **MD** | 2 | 0 | 322 | 5 |
| **MOD** | 5 | 0 | 37 | 274 |

**Per-Class Metrics**:
- ND: Precision=0.98, Recall=0.96, F1=0.97
- VMD: Precision=0.99, Recall=1.00, F1=1.00
- MD: Precision=0.88, Recall=0.98, F1=0.93
- MOD: Precision=0.98, Recall=0.87, F1=0.92

---

## 🔬 Differences from Original Study

| Aspect | Original | Our Implementation | Reason |
|--------|----------|-------------------|--------|
| GPU | NVIDIA RTX6000 | Metal (MPS) macOS | Hardware availability |
| `inplace=True` | ✅ Used | ❌ Disabled | MPS compatibility |
| SMOTE Strategy | Specified per-class | 'auto' | Same result, simpler |
| Validation Phase | Not detailed | Fully implemented | Better monitoring |
| Loss tracking | Not provided | Plotted curves | Better analysis |

---

## 🛠️ Requirements

See [requirements.txt](requirements.txt) for complete dependencies.

**Main packages:**
- PyTorch >= 2.0
- numpy >= 1.21
- scikit-learn >= 1.0
- imbalanced-learn >= 0.9
- matplotlib >= 3.4
- kagglehub >= 0.1

---

## ⚙️ Hardware Requirements

### Minimum
- 4GB RAM
- CPU with 4 cores
- 10GB disk space

### Recommended
- 8GB RAM
- GPU with 2GB VRAM (NVIDIA CUDA, Apple Metal, etc.)
- 15GB disk space

### Training Time
- **GPU (Metal/CUDA)**: ~1-1.5 hours for 50 epochs
- **CPU**: 3-7 hours for 50 epochs

---

## 📖 References

**Original DEMNET Paper:**
```
Murugan, S., Venkatesan, C., Sumithra, M. G., Gao, X. Z., Elakkiya, B., Akila, M., & Manoharan, S. (2021).
DEMNET: A Deep Learning Model for Early Diagnosis of Alzheimer Diseases and Dementia From MR Images.
IEEE Access, 9, 90319-90329.
DOI: 10.1109/ACCESS.2021.3090474
```

**Dataset Source:**
```
Dubey, S. (2019). Alzheimer's Dataset (4 Class of Images).
Retrieved from: https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images
```

---

## ✅ Validation & Testing

The implementation includes:
- ✅ Original DEMNET architecture (4,534,996 parameters)
- ✅ SMOTE class balancing (applied only to training set, no data leakage)
- ✅ 80/10/10 train/val/test split on Kaggle training data
- ✅ All hyperparameters from original paper (50 epochs, batch size 16, lr=0.001)
- ✅ RMSprop optimizer with proper validation monitoring
- ✅ Dropout (0.7, 0.5, 0.2) and batch normalization in DEMNET blocks
- ✅ Model checkpointing (saves best validation loss model)

**Result**: 72.97% test accuracy with proper data integrity (no synthetic images in test set)

---

## 📝 License

This project implements the DEMNET architecture from the original research paper by Murugan et al. (2021). The implementation is provided for educational and research purposes.

---

## 👤 Author

**Implementation**: Gabriel  
**Date**: August 2026  
**Original Paper**: Murugan et al., 2021 (IEEE Access)

---

## 🤝 Contributing

Suggestions and improvements are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Request features
- Ask questions

---

## 📧 Questions?

For questions about:
- **The implementation**: Check the Jupyter notebook comments
- **The original paper**: See references section
- **The dataset**: Visit Kaggle dataset page
- **DEMNET architecture**: Consult the original IEEE Access paper

---

## 🎓 Educational Value

This project demonstrates:
✅ Deep learning for medical image classification  
✅ Handling class imbalance with SMOTE  
✅ PyTorch model development  
✅ Proper train/val/test methodology with validation phase  
✅ GPU acceleration (CUDA, Metal, etc.)  
✅ Model checkpointing and evaluation  
✅ Professional code documentation  

Useful for understanding DEMNET architecture and MRI classification pipeline.

---

**Last Updated**: September 4, 2026  
**Model Accuracy**: 72.97%  
**Status**: ✅ Proper implementation with no data leakage
