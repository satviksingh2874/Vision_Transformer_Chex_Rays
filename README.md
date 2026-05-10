# Vision Transformer (ViT) for Multi-Label Chest X-Ray Classification

A **from-scratch implementation** of a Vision Transformer (ViT) for multi-label classification of chest X-ray images using the **CheXpert dataset**. This project demonstrates how a transformer-based architecture can learn meaningful representations from medical images to predict multiple thoracic conditions simultaneously.

---

## 📌 **Project Overview**
Chest X-ray interpretation is a critical medical imaging task, often requiring the identification of **multiple co-existing abnormalities** in a single image. This project implements a **custom Vision Transformer (ViT)** to address this challenge as a **15-class multi-label classification problem** using the CheXpert benchmark dataset.

### Key Features:
- **End-to-end ViT implementation** (no pre-trained backbones).
- **Multi-label classification** with sigmoid activation for independent disease probability estimation.
- **Handles class imbalance** and uncertain labels (CheXpert-specific).
- **Training workflow** optimized for medical imaging: preprocessing, patch embedding, positional encoding, and transformer encoder blocks.
- **Evaluation metrics**: Per-class AUC-ROC, precision, recall, F1-score, and Hamming loss.

---

## 🛠 **Technologies & Libraries**
- **Python 3.8+**
- **PyTorch** (for model implementation)
- **NumPy, Pandas** (data handling)
- **Matplotlib, Seaborn** (visualization)
- **Scikit-learn** (metrics)

---

## 📂 **Repository Structure**
```bash
.
├── data/                  # CheXpert dataset (or symlink)
│   ├── train/             # Training images and labels
│   ├── valid/             # Validation images and labels
│   └── test/              # Test images and labels
├── models/                # Saved model checkpoints
├── src/
│   ├── config.py          # Hyperparameters and paths
│   ├── dataset.py         # Data loading and preprocessing
│   ├── model.py           # ViT architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── utils.py           # Helper functions
├── results/               # Training logs, ROC curves, etc.
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

---

## 🚀 **Getting Started**

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-username/vit-chexpert.git
cd vit-chexpert
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Dataset Setup**
- Download the **CheXpert dataset** (e.g., from [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/)).
- Organize the dataset into `data/train`, `data/valid`, and `data/test` folders.
- Update paths in `src/config.py`.

### 4. **Training**
```bash
python src/train.py --epochs 33 --batch_size 32 --lr 0.0001
```
- **Hyperparameters** (default):
  - Image size: `(3, 224, 224)`
  - Patch size: `(16, 16)`
  - Embedding dimension: `512`
  - Transformer blocks: `3`
  - Attention heads: `8`
  - MLP hidden dimension: `1024`
  - Optimizer: `Adam`
  - Loss: `BCEWithLogitsLoss`

### 5. **Evaluation**
```bash
python src/evaluate.py --model_path models/optimal_epoch.pth
```
- Generates **ROC curves**, **class-wise metrics**, and **confusion matrices**.

---

## 🔍 **Model Architecture**
### Vision Transformer (ViT) Pipeline:
1. **Patch Embedding**:
   - Input image split into `(16x16)` patches.
   - Flattened and linearly projected into `512`-dim embeddings.
2. **Positional Encoding**:
   - Adds spatial context to patch embeddings.
3. **Transformer Encoder**:
   - Stacked blocks with **multi-head self-attention**, **layer normalization**, and **MLP**.
4. **Classification Head**:
   - Sigmoid activation for **multi-label output** (15 classes).

### Key Components:
- **Multi-Head Attention**: Captures long-range dependencies across patches.
- **Residual Connections**: Stabilizes deep network training.
- **Dropout**: Regularization to prevent overfitting.

---

## 📊 **Results**
| Metric               | Epoch 1 | Epoch 17 (Optimal) | Epoch 33 (Final) |
|----------------------|---------|--------------------|------------------|
| **Validation Loss**  | High    | **Lowest**         | Slightly higher  |
| **AUC-ROC (Macro)**  | ~0.6    | **~0.85**          | ~0.83            |
| **F1-Score (Macro)** | ~0.5    | **~0.75**          | ~0.72            |

- **Optimal Checkpoint**: Epoch 17 (best trade-off between training and validation performance).
- **Overfitting**: Observed in later epochs (validation loss increases).

### ROC Curves:
- **Strong performance** for common pathologies (e.g., "Pneumonia", "Pleural Effusion").
- **Weaker performance** for rare/subtle conditions (e.g., "Pneumothorax").

---

## 📝 **Challenges & Solutions**
| Challenge                          | Solution                                                                 |
|------------------------------------|--------------------------------------------------------------------------|
| **Class Imbalance**                | Used **weighted loss** and **threshold tuning** for rare classes.       |
| **Uncertain Labels**               | Treated "uncertain" as **positive** (CheXpert convention).              |
| **High Computational Cost**        | Optimized batch size (`32`) and mixed-precision training.             |
| **Spatial Context in X-rays**      | **Positional embeddings** + **self-attention** to capture anatomical relationships. |

---

## 🌟 **Key Takeaways**
1. **ViT from Scratch Works**: Achieved **competitive performance** without pre-trained backbones.
2. **Optimal Early Stopping**: Epoch 17 outperformed the final epoch due to overfitting.
3. **Medical Imaging Nuances**: Class imbalance and label uncertainty significantly impact performance.

---

## 🔮 **Future Work**
- **Pretrained ViT**: Fine-tune a pre-trained ViT (e.g., on ImageNet) for better initialization.
- **Attention Visualization**: Use **Grad-CAM** or **attention maps** to interpret model decisions.
- **Data Augmentation**: More aggressive augmentation (e.g., **RandAugment**) for robustness.
- **Class Balancing**: Experiment with **focal loss** or **oversampling**.

---

## 📜 **License**
This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## 🙏 **Acknowledgments**
- **CheXpert Dataset**: [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/)
- **ViT Paper**: [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- **PyTorch**: [Official Documentation](https://pytorch.org/docs/stable/index.html)

---
