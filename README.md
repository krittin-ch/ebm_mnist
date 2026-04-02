# Energy-Based Model (EBM) with Linear Probing and Vision Transformer Variants

This repository implements **Energy-Based Models (EBMs)** on the **MNIST dataset**, covering:

* Supervised learning
* Self-supervised learning (SSL)
* Linear probing for representation evaluation
* Vision Transformer (ViT) variants
* Feature space visualization using **t-SNE** and **UMAP**

---

## 📌 Overview

Energy-Based Models learn a function ( E(x) ) that assigns:

* **Low energy** → real (data) samples
* **High energy** → invalid / negative samples

Instead of predicting labels directly, the model learns the **structure of the data distribution**.

This repository explores how representations learned by EBMs evolve under:

* Supervised training
* Self-supervised training
* Transformer-based architectures

---

## 📂 Project Structure

```
.
├── data/                  # MNIST dataset
├── sup/                   # Supervised training outputs
├── ssl/                   # Self-supervised (CNN)
├── ssl_lp/                # Linear probe on SSL features
├── ssl_vit/               # Self-supervised ViT
├── ssl_vit_lp/            # Linear probe on ViT features
├── modules.py             # Core model components
├── vit.py                 # Vision Transformer implementation
├── tools.py               # Utilities
├── run_model.py           # Main training script
├── loss_plot.py           # Loss visualization
├── tsne_plot.py           # t-SNE visualization
├── umap_plot.py           # UMAP visualization
├── scripts.sh             # Training scripts
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd <repo-name>

pip install -r requirements.txt
```

---

## 📊 Dataset

The project uses **MNIST**, stored in:

```
data/MNIST/raw/
```

---

## 🚀 Training

### Supervised

```bash
python run_model.py --mode sup
```

---

### Self-Supervised (CNN)

```bash
python run_model.py --mode ssl
```

---

### Self-Supervised (Vision Transformer)

```bash
python run_model.py --mode ssl_vit
```

---

### Linear Probing

```bash
python run_model.py --mode ssl_lp
python run_model.py --mode ssl_vit_lp
```

---

## 🧠 Model Details

### Energy-Based Model

The model learns an energy function:

[
E(x)
]

Training encourages:

* Lower energy for real samples
* Higher energy for negative samples

This creates a structured **energy landscape** over the data.

---

### Linear Probing

After self-supervised training:

* Encoder is **frozen**
* A **linear classifier** is trained on top

This measures how well the learned features separate classes.

---

### Vision Transformer (ViT)

Implemented in `vit.py`:

* Patch embeddings
* Multi-head self-attention
* Transformer encoder layers

Used to compare representation learning against CNN-based EBMs.

---

## 📈 Representation Visualization

The repository tracks how feature representations evolve during training using:

* **t-SNE**
* **UMAP**

Each point represents a data sample, colored by class label.

Over time:

* Clusters become more compact
* Classes become more separable
* Structure emerges in latent space

---

## 🎞️ Visual Results

### t-SNE

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <th>Supervised</th>
    <th>SSL (CNN)</th>
    <th>SSL (ViT)</th>
  </tr>
  <tr>
    <td align="center">
      <img src="sup/tsne_animation.gif" style="width:100%; max-height:250px; object-fit:contain;">
    </td>
    <td align="center">
      <img src="ssl/tsne_animation.gif" style="width:100%; max-height:250px; object-fit:contain;">
    </td>
    <td align="center">
      <img src="ssl_vit/tsne_animation.gif" style="width:100%; max-height:250px; object-fit:contain;">
    </td>
  </tr>
</table>

---

### UMAP

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <th>Supervised</th>
    <th>SSL (CNN)</th>
    <th>SSL (ViT)</th>
  </tr>
  <tr>
    <td align="center">
      <img src="sup/umap_animation.gif" style="width:100%; max-height:250px; object-fit:contain;">
    </td>
    <td align="center">
      <img src="ssl/umap_animation.gif" style="width:100%; max-height:250px; object-fit:contain;">
    </td>
    <td align="center">
      <img src="ssl_vit/umap_animation.gif" style="width:100%; max-height:250px; object-fit:contain;">
    </td>
  </tr>
</table>

---

## 📊 Outputs

Each experiment produces:

* `loss.csv` → training loss
* `weights/` → model checkpoints
* `.gif` → embedding evolution
* `.png` → final visualization snapshot

---

## 🧾 Summary

This project demonstrates:

* Energy-based representation learning
* Self-supervised feature extraction
* Evaluation via linear probing
* Comparison between CNN and Vision Transformer representations
* Visualization of latent space evolution during training
