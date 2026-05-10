# Vision Transformer (ViT) for Multi-Label Chest X-Ray Classification

A **from-scratch implementation** of a **Vision Transformer (ViT)** for **multi-label classification** of chest X-ray images using the **CheXpert dataset**. This project explores how transformer-based architectures can learn **meaningful representations** from medical images to predict **multiple thoracic conditions** simultaneously.

---

## 📌 **Project Overview**
Chest X-ray interpretation is a **clinically critical** task, often requiring the identification of **multiple co-existing abnormalities** in a single image. This project implements a **custom Vision Transformer (ViT)** to address this as a **15-class multi-label classification problem** using the **CheXpert benchmark dataset**.

### Key Objectives:
- Implement a **Vision Transformer from scratch** (no pre-trained backbones).
- Handle **multi-label classification** with independent disease probability estimation.
- Address **class imbalance** and **uncertain labels** (CheXpert-specific).
- Optimize the **training workflow** for medical imaging: preprocessing, patch embedding, positional encoding, and transformer encoder blocks.

---

## 🛠 **Technologies & Libraries**
- **Python 3.8+**
- **PyTorch** (for model implementation)
- **NumPy, Pandas** (data handling)
- **Matplotlib, Seaborn** (visualization)
- **Scikit-learn** (metrics)

---

## 📂 **Repository Structure**
This repository is organized to provide a **clear and modular** implementation of the ViT for chest X-ray classification.

```bash
.
├── .ipynb_checkpoints/           # Jupyter notebook checkpoints (auto-generated)
├── metrics_VIT_chexstxray/       # Training metrics, graphs, and evaluation results
│   ├── metrics/                  # Performance metrics (e.g., AUC-ROC, F1-score)
│   └── graphs/                   # Training/validation curves and ROC plots
│
├── model_weights/                # Saved model checkpoints (e.g., optimal epoch, final epoch)
│
├── Inference_Code.ipynb           # Notebook for running inference on trained models
├── README.md                     # Project documentation (this file)
├── VIT_CHEX_REPORT.docx          # Detailed project report (theory, methodology, results)
├── ViT_CheXray.ipynb             # Main notebook for model training and evaluation
├── test_df_full.csv              # Test dataset (labels and metadata)
└── train_df_full.csv             # Training dataset (labels and metadata)
