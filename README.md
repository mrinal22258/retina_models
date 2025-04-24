# Predicting Firing Rates of Retinal Ganglion Cells using Deep Learning

This project focuses on predicting the firing rate of retinal ganglion cells in response to natural visual stimuli. We used the `naturalscenes.h5` dataset, which contains natural scene inputs and corresponding ganglion cell activity. Our approach leverages deep learning models tailored to the structure and behavior of biological neural systems.

## Objective

To develop and evaluate deep learning architectures that can predict the firing rates of ganglion cells given natural image stimuli.

---

## 📁 Dataset

Dataset used can be downloaded from: https://purl.stanford.edu/rk663dm5577

- **Source:** `naturalscenes.h5`
- **Description**:  
  Contains:
  - Grayscale natural image patches (shape: `(1, 50, 50)`)
  - Corresponding firing rates of 9 ganglion cells
- **Preprocessing**:
  - Normalized pixel intensities to [0,1]
  - Split into train/validation/test using stratified sampling
  - Performed EDA (see `eda.ipynb` and `eda2.ipynb`) to visualize stimulus-firing patterns and correlation among ganglion cells


---

## Models Implemented

### 1. BNCNN
- A **Batch-Normalized Convolutional Neural Network**
- Captures spatial features in static image patches
- **Architecture**:
  - Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → Linear → Output
  - Optional: Softplus activation for biological realism

### 2. CNNBiLSTM
- Extends spatial learning with **temporal awareness**
- BiLSTM layer helps capture **sequence-based correlations** if image input is treated as temporal patches (optional)
- Useful in mimicking memory-like behavior in biological systems

### 3. LinearStackedBNCNN
- Hybrid model:
  - Feature extractor = frozen BNCNN
  - Final layer = linear regression (or shallow MLP)
- Balances interpretability with representational power

---

## ⚙️ Hyperparameters

```python
hyperparams = {
    'img_shape': (1, 50, 50),
    'chans': [32, 64],           # Number of filters in convolutional layers
    'ksizes': [5, 5, 3],         # Kernel sizes for convolutional layers
    'bias': False,               # Whether to use bias in layers
    'bnorm_d': 2,                # BatchNorm dimension
    'bn_moment': 0.1,            # Momentum for BatchNorm
    'noise': 0.1,                # Gaussian noise for regularization
    'activ_fxn': 'ReLU',         # Activation function
    'n_units': 9,                # Number of ganglion cells (output units)
    'gc_bias': False,            # Bias in final GC layer
    'softplus': False,           # Use softplus in output instead of linear
    'convgc': False,             # Whether to use conv layer for GC layer
    'centers': None              # Optional Gaussian centers for convGC
}
```
---

## 📈 Results Summary

| Model                | Train Corr ↑ | Val Corr ↑ |
|---------------------|--------------|------------|
| **BNCNN**            | 0.5452       | 0.4813     | 
| **LinearStackedBNCNN** | 0.5167    | 0.4820     |
| **CNNBiLSTM**        | 0.5320       | 0.4728     |


---

## Other Resources

PTH files for models and training results are available in this folder: https://drive.google.com/drive/folders/1lIV1FeJuX-xXpeN8xosY_GJCZQCxMWcK
