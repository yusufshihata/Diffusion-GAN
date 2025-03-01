# Diffusion-GAN

A **Diffusion-GAN** implementation for high-quality image generation, combining diffusion models with adversarial training for enhanced synthesis quality.

---

## 🚀 Features
- Implements **Diffusion-GAN** based on recent research.
- Supports **PyTorch** and **CUDA** for efficient training.
- Modular design for **easy customization**.

---

## 📂 Project Structure
```
Diffusion-GAN/
├── models/                # Model architecture
├── datasets/              # Data loading scripts
├── checkpoints/           # Saved models
├── results/               # Generated images
├── scripts/               # Training & evaluation scripts
├── utils/                 # Utility functions
├── README.md              # Project documentation
```

---

## 📦 Installation

First, clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/diffusion-gan.git
cd diffusion-gan
pip install -r requirements.txt
```

Ensure you have **PyTorch** and **CUDA** installed.

---

## 🎯 Training the Model

To train **Diffusion-GAN** from scratch, run:
```bash
python scripts/train.py --dataset path/to/data --epochs 100 --batch_size 64
```

- Modify hyperparameters in `configs/config.yaml`.
- Checkpointing enabled (saved in `checkpoints/`).

---

## 📌 Contributing

Feel free to submit issues, feature requests, or pull requests!

---

## 📜 References
- [Original Diffusion-GAN Paper](https://arxiv.org/abs/2206.02262)
- [GANs and Diffusion Models](https://paperswithcode.com/methods/category/diffusion-models)

---
