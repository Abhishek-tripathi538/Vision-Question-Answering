import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_FOLDER = "./train"
EPOCHS = 5
BATCH_SIZE = 2
LR = 5e-5

# ======================
# LOAD BLIP2 + LoRA
# ======================
print("📦 Loading BLIP2 model and processor...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "ybelkada/blip2-opt-2.7b-fp16-sharded",
    device_map="auto",
    quantization_config=quant_config
)

print("✅ Loaded BLIP2.")
print("🎛️ Applying LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ======================
# LOAD DATASET
# ======================
def load_dataset_from_folder(train_dir):
    captions_root = os.path.join(train_dir, "caption")
    images_root = os.path.join(train_dir, "Image")
    image_paths, texts = [], []

    for fname in os.listdir(captions_root):
        if not fname.endswith(".txt"):
            continue
        id_name = fname.replace(".txt", "")
        caption_path = os.path.join(captions_root, fname)
        image_folder = os.path.join(images_root, id_name)

        if not os.path.exists(image_folder):
            continue

        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        for img_file in os.listdir(image_folder):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(image_folder, img_file)
                image_paths.append(img_path)
                texts.append(caption)

    return list(zip(image_paths, texts))

print("📁 Loading dataset from folder...")
examples = load_dataset_from_folder(TRAIN_FOLDER)
print(f"✅ Dataset loaded with {len(examples)} examples.")

# ======================
# DATASET + DATALOADER
# ======================
class ImageCaptioningDataset(Dataset):
    def __init__(self, examples, processor):
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        img_path, text = self.examples[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "text": text}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    text_inputs = processor.tokenizer([item["text"] for item in batch], padding=True, return_tensors="pt", truncation=True)
    input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone()
    }

train_dataset = ImageCaptioningDataset(examples, processor)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ======================
# TRAINING LOOP
# ======================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

print("🚀 Starting fine-tuning...\n")

for epoch in range(EPOCHS):
    print(f"🔁 Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0

    for i, batch in enumerate(tqdm(train_dataloader)):
        input_ids = batch["input_ids"].to(DEVICE)
        pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"Batch {i+1} | Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"✅ Epoch {epoch+1} completed | Avg Loss: {total_loss / len(train_dataloader):.4f}")

# ======================
# SAVE MODEL
# ======================
print("💾 Saving model to ./blip2_lora_finetuned")
model.save_pretrained("./blip2_lora_finetuned")
processor.save_pretrained("./blip2_lora_finetuned")
