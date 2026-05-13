"""
MedSimplify — Fine-tuning Script (Unsloth + QLoRA)
Run on Kaggle T4/P100 or Lightning AI.
Fine-tunes Gemma 4 9B for document simplification.
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# === CONFIG ===
MODEL_NAME = "google/gemma-4-9b"  # Adjust if model ID is different on HF
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
OUTPUT_DIR = "medsimplify-gemma4-9b"
DATA_PATH = "data/processed/train.jsonl"

# === LOAD MODEL ===
print("Loading model with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    dtype=None,  # auto-detect
)

# === ADD LORA ADAPTERS ===
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# === FORMAT DATA ===
PROMPT_TEMPLATE = """Below is a complex document. Rewrite it in Easy Read format for someone with a cognitive disability.

### Rules:
- Short sentences (maximum 10 words each)
- Simple words only (no jargon)
- Emoji icons for visual anchoring
- Bullet points for actions
- Extract all required actions into a checklist

### Complex Document:
{input}

### Easy Read Version:
{output}"""


def formatting_func(examples):
    texts = []
    for instruction, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        text = PROMPT_TEMPLATE.format(input=inp, output=out) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}


# === LOAD DATASET ===
print("Loading training data...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(formatting_func, batched=True)

print(f"Training on {len(dataset)} examples")

# === TRAINING ===
print("Starting fine-tuning...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_steps=100,
        save_total_limit=2,
    ),
)

trainer_stats = trainer.train()
print(f"\nTraining complete! Loss: {trainer_stats.training_loss:.4f}")

# === SAVE ===
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# === SAVE AS GGUF FOR OLLAMA ===
print("Exporting to GGUF for Ollama...")
model.save_pretrained_gguf(
    f"{OUTPUT_DIR}-gguf",
    tokenizer,
    quantization_method="q4_k_m"
)

print(f"\nDone! Model saved to {OUTPUT_DIR} and {OUTPUT_DIR}-gguf")
print("Next: upload to HuggingFace Hub or use locally with Ollama")
