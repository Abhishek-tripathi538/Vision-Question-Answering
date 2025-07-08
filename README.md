# Fine-tuning BLIP2 with LoRA and Quantization

## Introduction

This project demonstrates how to fine-tune the BLIP2 (Bootstrapped Language-Image Pretraining) model for image captioning tasks using Parameter-Efficient Fine-Tuning (PEFT) methods, specifically LoRA (Low-Rank Adaptation), along with model quantization for efficient training.

### What is BLIP2?

**BLIP2** is a state-of-the-art vision-language model developed to bridge the gap between visual and textual understanding. It is designed for tasks such as image captioning, visual question answering, and image-text retrieval. BLIP2 leverages a two-stage approach: first, it uses a vision encoder to extract image features, and then a large language model (LLM) to generate or interpret text based on those features. This architecture allows BLIP2 to achieve strong performance on a wide range of multimodal tasks.

### What are PEFT Methods and LoRA?

**Parameter-Efficient Fine-Tuning (PEFT)** methods are techniques that enable the adaptation of large pre-trained models to new tasks using only a small number of additional parameters. This is especially useful when working with very large models, as it reduces the computational and memory requirements for fine-tuning.

**LoRA (Low-Rank Adaptation)** is a popular PEFT method. Instead of updating all the weights of a large model, LoRA injects small trainable matrices (of low rank) into certain layers (like attention projections). During fine-tuning, only these additional parameters are updated, while the original model weights remain frozen. This approach drastically reduces the number of trainable parameters and makes fine-tuning feasible even on limited hardware.

### What is Quantization?

**Quantization** is a model compression technique that reduces the precision of the numbers used to represent model parameters (e.g., from 32-bit floating point to 4-bit integers). This leads to significant reductions in memory usage and can speed up both training and inference. In this project, 4-bit quantization is applied using the `bitsandbytes` library, enabling efficient training of large models like BLIP2 on consumer GPUs.

---

## Code Flow

1. **Configuration**
   - Sets device (CPU/GPU), training folder, epochs, batch size, and learning rate.

2. **Model and Processor Loading**
   - Loads the BLIP2 model and processor using HuggingFace Transformers.
   - Applies 4-bit quantization using `BitsAndBytesConfig` for efficient memory usage.

3. **Applying LoRA**
   - Configures LoRA with specified parameters (rank, alpha, dropout, target modules).
   - Wraps the BLIP2 model with LoRA using PEFT, enabling parameter-efficient fine-tuning.

4. **Dataset Preparation**
   - Loads image-caption pairs from a specified folder structure.
   - Each caption is paired with all images in its corresponding subfolder.

5. **Dataset and DataLoader**
   - Defines a custom `ImageCaptioningDataset` for loading and processing images and captions.
   - Uses a custom `collate_fn` to batch and tokenize data for training.

6. **Training Loop**
   - Sets up the optimizer and runs the training loop for the specified number of epochs.
   - For each batch, computes the loss, performs backpropagation, and updates model parameters.
   - Prints loss statistics during training.

7. **Saving the Model**
   - After training, saves the fine-tuned model and processor to the `./blip2_lora_finetuned` directory.

## Folder Structure

- `train/`
  - `caption/` - Contains `.txt` files with captions.
  - `Image/` - Contains subfolders for each caption, each with corresponding images.
- `blip2_lora_finetuned/` - Output directory for the fine-tuned model and processor.

## Requirements

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- PEFT
- bitsandbytes
- tqdm
- Pillow

## Usage

1. Prepare your dataset in the expected folder structure.
2. Run the fine-tuning script:
   ```sh
   python finetune.py
   ```
3. The fine-tuned model will be saved in `./blip2_lora_finetuned`.

---

For more details on PEFT, LoRA, quantization, and BLIP2, refer to the official documentation:
- [PEFT](https://github.com/huggingface/peft)
-