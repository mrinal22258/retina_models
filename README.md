# Predicting Firing Rates of Retinal Ganglion Cells using Deep Learning

This project focuses on predicting the firing rate of retinal ganglion cells in response to natural visual stimuli. We used the `naturalscenes.h5` dataset, which contains natural scene inputs and corresponding ganglion cell activity. Our approach leverages deep learning models tailored to the structure and behavior of biological neural systems.

## Objective

To develop and evaluate deep learning architectures that can predict the firing rates of ganglion cells given natural image stimuli.

---

## 📁 Dataset

Dataset used can be downloaded from: https://purl.stanford.edu/rk663dm5577

- **Source:** `naturalscenes.h5`
- **Description:** Contains grayscale natural scene patches and corresponding ganglion cell firing rate recordings.
- **Input Shape:** (1, 50, 50) grayscale image patches.

More about the dataset can be learnt from the preprocessing.ipynb. 

---

## Models Implemented

### 1. BNCNN
A batch-normalized convolutional neural network that extracts spatial features from input patches. It applies multiple convolutional layers followed by batch normalization and activation.

### 2. CNNBiLSTM
Combines convolutional layers for feature extraction with a bidirectional LSTM layer to capture temporal dependencies or sequential correlations in the visual stimulus.

### 3. LinearStackedBNCNN
A linear model stacked on top of BNCNN features to provide a simpler but potentially interpretable prediction mechanism.

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

## Other Resources

PTH files for models and training results are available in this folder: https://drive.google.com/drive/folders/1lIV1FeJuX-xXpeN8xosY_GJCZQCxMWcK
