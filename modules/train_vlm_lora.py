import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from PIL import Image
import os
import gc

# 1. Αναγκαστική επιλογή της NVIDIA RTX 3050 ως κύρια συσκευή
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_id = "HuggingFaceTB/SmolVLM-Instruct"

print("Προετοιμασία Processor...")
processor = AutoProcessor.from_pretrained(model_id)
# Μικρή ανάλυση για ασφάλεια στη VRAM
processor.image_processor.size = {"height": 128, "width": 128}

print("Φόρτωση Μοντέλου στην GPU...")
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map={"": "cuda:0"}, # Ρητή ανάθεση ολόκληρου του μοντέλου στην NVIDIA
    trust_remote_code=True
)

# 2. Ρύθμιση LoRA
config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# 3. Προετοιμασία Δεδομένων με χειροκίνητο device assignment
def collate_fn(batch):
    images = []
    texts = []
    for item in batch:
        img_path = os.path.join("constellation_dataset/images", item["image"])
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        prompt = f"User: <image>\n{item['conversations'][0]['content']} Assistant: {item['conversations'][1]['content']}"
        images.append([image])
        texts.append(prompt)
    
    if not images: 
        return None

    # Επεξεργασία των inputs
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    # Στέλνουμε ρητά κάθε tensor στην κάρτα γραφικών
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# 4. Φόρτωση Dataset
print("Φόρτωση Dataset...")
dataset = load_dataset("json", data_files="vlm_dataset.jsonl", split="train")

# 5. Training Arguments (Ρυθμίσεις κατά του "παγώματος")
training_args = TrainingArguments(
    output_dir="./vlm_constellations_results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1, 
    learning_rate=2e-4,
    logging_steps=1,
    max_steps=100,
    fp16=True,                     
    dataloader_num_workers=0,      # Απενεργοποίηση multi-processing (κρίσιμο για Python 3.14 στα Windows)
    dataloader_pin_memory=False,   # Αποφυγή εμπλοκής με το σύστημα μνήμης
    gradient_checkpointing=True,   
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False
)

# Καθαρισμός προηγούμενης μνήμης
gc.collect()
torch.cuda.empty_cache()

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn
)

print("\n--- ΕΚΚΙΝΗΣΗ ΕΚΠΑΙΔΕΥΣΗΣ ---")
trainer.train()

# 7. Αποθήκευση
model.save_pretrained("./vlm_lora_final")
print("\nΤΕΛΟΣ! Το μοντέλο αποθηκεύτηκε στο: ./vlm_lora_final")