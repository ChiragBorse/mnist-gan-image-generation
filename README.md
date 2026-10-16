# 🎨 Deep Convolutional Generative Adversarial Network (DCGAN) on MNIST

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A complete implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** built using **TensorFlow 2** and **Keras** to synthesize realistic handwritten digits from random Gaussian latent noise.

---

## 📌 Problem Statement
**Question:** Implement a Generative Adversarial Network (GAN) architecture using TensorFlow 2. Train the GAN model on a dataset such as MNIST or CIFAR-10 for image generation tasks.

---

## ❓ Project Overview (The 5 Ws)

* **What:** A two-player zero-sum game composed of two competing neural networks:
  * **Generator ($G$):** Maps a 100-dimensional latent noise vector $z \sim \mathcal{N}(0, I)$ into a synthetic $28 \times 28 \times 1$ image.
  * **Discriminator ($D$):** A binary classifier evaluating whether an input image is genuine (from MNIST) or synthetic (from $G$).
* **Why:** Traditional autoencoders generate blurred images due to pixel-wise MSE/BCE averaging. GANs produce sharp, authentic images by using a learned, dynamic discriminator loss rather than a static hand-crafted distance metric.
* **Who:** Students, researchers, and engineers exploring adversarial optimization, generative modeling, and computer vision synthesis.
* **Where:** Runs on Google Colab (free T4 GPU recommended for fast training) or any local Python environment with GPU/CPU support.
* **How:** Optimizes the classic Minimax objective via custom gradient computation using two synchronized `tf.GradientTape` contexts inside a compiled `@tf.function`.

---

## ⚙️ Architecture & Technical Pipeline

```text
 Latent Vector z ~ N(0, 1) [100-d]
               │
               ▼
 ┌──────────────────────────────┐
 │       GENERATOR (G)          │
 │ Dense(7*7*256) -> BatchNorm  │
 │ LeakyReLU -> Reshape(7,7,256)│
 │ Conv2DTranspose 128 (5x5, s1)│
 │ Conv2DTranspose 64  (5x5, s2)│
 │ Conv2DTranspose 1   (5x5, s2)│
 │ Activation: tanh [-1, 1]     │
 └─────────────┬────────────────┘
               │
               ▼
       Synthetic Image
       [28 x 28 x 1]                    Real MNIST Image [28 x 28 x 1]
               │                                      │
               └───────────────┬──────────────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │      DISCRIMINATOR (D)       │
                │ Conv2D 64 (5x5, s2) + Leaky  │
                │ Dropout (0.3)                │
                │ Conv2D 128 (5x5, s2) + Leaky │
                │ Dropout (0.3)                │
                │ Flatten -> Dense(1)          │
                │ Output: Raw Logit            │
                └──────────────┬───────────────┘
                               │
                               ▼
                   Binary Cross-Entropy Loss
                 (with from_logits=True)
