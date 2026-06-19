# CNN vs SmolVLM for Automatic Modulation Classification

This project compares a Multi-Task CNN (ResNet-based) and a fine-tuned SmolVLM model for Automatic Modulation Classification (AMC) under realistic RF impairments.

## Features

- Automatic Modulation Classification
- RF Impairment Simulation
- AWGN Noise
- IQ Imbalance
- Phase Noise
- ResNet Multi-Task Learning
- SmolVLM Fine-Tuning using LoRA

## Technologies

- Python
- PyTorch
- HuggingFace Transformers
- LoRA
- NumPy
- Matplotlib

## Results

Baseline CNN achieved 78% classification accuracy while SmolVLM demonstrated improved robustness on unseen SNR conditions.

## Repository Structure

```text
main.py
modules/
requirements.txt
cnn_confusion_matrix.png
ΣΥΓΚΡΙΣΗ_CNN_Smol_vlm.pdf
