# U-SAMNet: Supplementary Code

**Authors:** Prosenjit Roy, Kijoon Lee, Mohsen Taheri Andani, Toan Truong, Haojun You, Noushin Ghaffari
**Affiliations:** Prairie View A\&M University / Texas A\&M University

\---

## Overview

This folder contains five scripts. They cover the full pipeline from raw images to trained model. The scripts must be run in order.

\---

## Requirements

* Python 3.9
* TensorFlow 2.10
* PyTorch (for GAN training)
* OpenCV
* scikit-learn
* numpy
* CUDA 11.2
* cuDNN 8.1

**Hardware used:** NVIDIA Tesla V100 32GB GPU on Bridges-2 (Pittsburgh Supercomputing Center)

\---

## Scripts

### Step 1 — `data\_split\_augmentation.py`

This script splits the raw dataset into train, val, and test sets.

**What it does:**

* Reads all images from Dataset1 and Dataset2
* Splits 426 original images: 340 train (80%), 43 val (10%), 43 test (10%)
* Splitting is done BEFORE augmentation to prevent data leakage
* Applies rotation augmentation (0°, 90°, 180°, 270°) to the train set only
* Val and test sets keep only the original unrotated images

**Input:** `/content/Dataset1`, `/content/Dataset2`

**Output:**

```
raw\_images/
    train/    ← 1,360 rotated training images
    val/      ← 43 original images
    test/     ← 43 original images
```

**Run:**

```bash
python data\_split\_augmentation.py
```

\---

### Step 2 — `annotation\_pipeline.py`

This script generates binary ground truth masks for all images.

**What it does:**

* Runs each image through a classical computer vision pipeline
* Processes train, val, and test folders separately
* Saves a binary mask for each image with the same filename

**Pipeline steps:**

1. Gaussian blur (kernel 5×5, sigma=0)
2. Laplacian sharpening (centre weight=5, neighbours=-1)
3. CLAHE (clipLimit=2.0, tileGridSize=8×8)
4. Median blur (kernel=5)
5. Sobel edge detection (ksize=3, threshold=20)
6. Hough Circle Transform to find coupons (dp=1.2, minDist=40, param1=60, param2=30, minRadius=25, maxRadius=55)
7. Gaussian adaptive thresholding (blockSize=25, C=3)
8. Morphological erosion (3×3, 1 iter), dilation (3×3, 2 iter), closing (3×3)
9. Contour filter: keep area between 5 and 300 pixels

**Expert verification:** All masks were reviewed by three researchers. Each mask was accepted, rejected, or corrected. Masks with minor errors were rerun with adjusted C values. Badly annotated masks were excluded. After exclusion, 1,147 training masks were retained. All 43 val and 43 test masks were retained.

**Input:** `raw\_images/train/`, `raw\_images/val/`, `raw\_images/test/`

**Output:**

```
raw\_images/
    train/
        images/    ← original images
        masks/     ← binary ground truth masks
    val/
        images/
        masks/
    test/
        images/
        masks/
```

**Run:**

```bash
python annotation\_pipeline.py
```

\---

### Step 3 — `pix2pix\_synthesis\_.py`

This script trains a pix2pix GAN and generates synthetic image-mask pairs.

**What it does:**

* Trains a U-Net generator and PatchGAN discriminator
* Trains ONLY on the train split (val and test are never used)
* Generates 6,000 synthetic image-mask pairs
* Saves synthetic data to a separate folder

**Key settings:**

* Image size: 256×256
* Batch size: 4
* Epochs: 150
* Learning rate: 2e-4
* L1 loss weight: 100

**Input:** `raw\_images/train/images/`, `raw\_images/train/masks/`

**Output:**

```
synthetic\_data/
    images/    ← 6,000 synthetic images
    masks/     ← 6,000 corresponding masks
```

**Run:**

```bash
python pix2pix\_synthesis\_.py
```

\---

### Step 3b — Manual copy (required before Step 4)

After GAN training, copy the synthetic pairs into the train folder manually.

```bash
cp synthetic\_data/images/\* raw\_images/train/images/
cp synthetic\_data/masks/\*  raw\_images/train/masks/
```

After copying, the train folder should contain **7,147 image-mask pairs** (1,147 real + 6,000 synthetic).

\---

### Step 4 — `USAMNet\_implementation.py`

This script trains U-SAMNet and evaluates it on the test set.

**What it does:**

* Builds the U-SAMNet model with UGA block and three output heads
* Trains on the 7,147-image training set (real + synthetic)
* Evaluates on the 43 real test images
* Computes: Accuracy, Precision, Recall, F1, IoU
* Computes MC-dropout uncertainty (T=10 passes)
* Computes bootstrap confidence intervals (n=1,000)
* Computes calibration metrics: Brier Score, ECE, AUROC
* Saves trained model weights

**Key settings:**

* Optimizer: Adam
* LR scheduler: ReduceLROnPlateau (factor=0.5, patience=5, min\_lr=1e-6)
* Batch size: 16
* Epochs: 50
* Input size: 256×256

**Input:** `raw\_images/train/`, `raw\_images/test/`

**Output:** `usamnet.h5`, `usamnet\_inference.h5`

**Run:**

```bash
python USAMNet\_implementation.py
```

\---

### Step 5 — `Baseline\_Models.py`

This script trains and evaluates all nine baseline models.

**Baselines included:**

Standard segmentation:

1. U-Net
2. VGG16-UNet
3. ResNet50-UNet
4. HRNet
5. DeepLabV3+

Uncertainty-aware and attention-based:

6. MC-Dropout U-Net
7. Attention U-Net
8. SE-U-Net
9. CBAM-U-Net

All models are trained with identical settings to U-SAMNet (same optimizer, batch size, epochs, and data). This ensures fair comparison.

**Input:** `raw\_images/train/`, `raw\_images/test/`

**Run:**

```bash
python Baseline\_Models.py
```

\---

## Full Pipeline Summary

```
Step 1: data\_split\_augmentation.py   → splits raw images into train/val/test
Step 2: annotation\_pipeline.py       → generates binary masks for all splits
Step 3: pix2pix\_synthesis\_.py        → trains GAN, generates 6,000 synthetic pairs
Step 3b: manual copy                 → copies synthetic into train folder
Step 4: USAMNet\_implementation.py    → trains U-SAMNet, evaluates, reports metrics
Step 5: Baseline\_Models.py           → trains and evaluates all 9 baselines
```

\---

## Dataset

The E-PBF dataset cannot be shared publicly due to confidentiality restrictions. Data may be available upon request with approval from all collaborating institutions. Contact: noghaffari@pvamu.edu

\---

