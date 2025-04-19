# Text-to-Speech Model based on RWKV7 Architecture

## Introduction

This repository primarily explores the use of Focal Codec to convert between speech signals and tokens, and employs an RNN model based on RWKV7 to achieve token prediction for speech generation. The main features are as follows:

- **Multi-Stage Training**: During the pre-training phase, the model performs Token Prediction merely on speech tokens, allowing it to extensively learn speech features **from unlabeled speech data**, and then aligns using text tokens.
- **Lightweight Character-Level Text Tokenizer**: Designed and implemented a lightweight character-level text tokenizer, significantly reducing the vocab size, enabling the model to better understand the relationship between text tokens and speech tokens.

## Project Progress

- [x] Conducted experiments based on the public TTS dataset (LibriTTS), achieving remarkable results with 103M parameters.
- [x] Efficiently achieved voice cloning for speech generation by inserting prompt audio tokens after text instructions.
- [ ] Train guide, Inference guide and trained weights is coming soon.

## Demonstration

Here shows some generated samples with prompt speech provided:



https://github.com/user-attachments/assets/7a6ba041-3cb2-45b0-9cdf-a5bb161e4e2d


