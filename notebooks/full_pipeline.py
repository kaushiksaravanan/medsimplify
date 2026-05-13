# MedSimplify - Complete Training Pipeline
# Fine-tune Gemma for Document Accessibility
# Run on Kaggle with GPU P100 enabled
import subprocess, sys, os, json, random

# 0. Install Dependencies
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "trl", "peft", "accelerate", "bitsandbytes"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "textstat", "huggingface_hub"], check=False)

# 1. Data Preparation
print("\n" + "="*60 + "\nSTEP 1: Data Preparation\n" + "="*60)
from datasets import load_dataset
os.makedirs("data", exist_ok=True)
all_pairs = []

print("Loading Wiki Auto...")
try:
    ds = load_dataset("GEM/wiki_auto_asset_turk", "train", split="train[:4000]", trust_remote_code=True)
    for item in ds:
        if "source" in item and "target" in item and len(item.get("source","")) > 50:
            all_pairs.append({"instruction": "Simplify this text for someone with a reading difficulty. Use short sentences (max 10 words). Replace jargon.", "input": item["source"], "output": item["target"]})
    print(f"  -> {len(all_pairs)} Wiki Auto pairs")
except Exception as e:
    print(f"  Wiki Auto failed: {e}")

print("Loading ASSET...")
try:
    ds = load_dataset("asset", "simplification", split="test", trust_remote_code=True)
    count = 0
    for item in ds:
        if "original" in item and "simplifications" in item:
            for simp in item["simplifications"][:2]:
                all_pairs.append({"instruction": "Rewrite in Easy Read format. Simple words. Max 10 words per sentence.", "input": item["original"], "output": simp})
                count += 1
                if count >= 2000: break
        if count >= 2000: break
    print(f"  -> {count} ASSET pairs")
except Exception as e:
    print(f"  ASSET failed: {e}")

# Synthetic document pairs (highest quality)
SYNTH = [
    {"i": "Transform this medical document into Easy Read format. Use: short sentences (max 10 words), markers [URGENT] [WARNING] [INFO] [OK] [MEDICINE] [DATE] [CALL] [CLOCK], bullet points, action checklist.",
     "inp": "Dear Patient, Following your recent appointment on 14/03/2026, I am writing to inform you that your blood test results indicate elevated levels of glycated haemoglobin (HbA1c) at 58 mmol/mol, which is above the recommended threshold of 48 mmol/mol for non-diabetic individuals. This finding is consistent with a diagnosis of Type 2 Diabetes Mellitus. I would recommend commencing Metformin 500mg twice daily with meals, and scheduling a follow-up appointment in 3 months for repeat HbA1c monitoring.",
     "out": "[INFO] Your doctor is writing about your blood test.\n\n[URGENT] Important result:\nYour blood sugar is too high.\nThis means you have Type 2 Diabetes.\n\n[MEDICINE] What you need to do:\n* Take a new medicine called Metformin\n* Take 1 tablet in the morning with food\n* Take 1 tablet in the evening with food\n\n[DATE] Appointments to make:\n* See your doctor again in 3 months\n* Get your eyes checked\n* Get your feet checked"},
    {"i": "Transform this government letter into Easy Read format.",
     "inp": "NOTICE OF DETERMINATION: Having considered your application, we have determined that you are not eligible for Universal Credit payments. Your combined household income of 2847 per month exceeds the threshold of 1248 per month. You have the right to request a Mandatory Reconsideration within one calendar month.",
     "out": "[INFO] This is about your Universal Credit.\n\n[WARNING] Decision:\nYou cannot get Universal Credit right now.\n\n[INFO] Why:\nYou earn too much money.\n* You earn: 2847 a month\n* The limit is: 1248 a month\n\n[INFO] If you disagree:\n* Ask them to look at it again\n* Do this within 1 month\n\n[CLOCK] Deadline: 1 month from today."},
    {"i": "Transform this prescription into Easy Read format.",
     "inp": "PRESCRIPTION: Amoxicillin 500mg Capsules. Take ONE capsule THREE times a day at regular intervals. Complete the full course even if you feel better. Do not take if allergic to penicillin. If difficulty breathing or swelling, seek immediate medical attention.",
     "out": "[MEDICINE] Your Medicine: Amoxicillin\n\n[INFO] How to take it:\n* Take 1 tablet, 3 times a day\n* Morning, afternoon, and evening\n* Take ALL tablets even if you feel better\n\n[WARNING] Do NOT take if:\n* You are allergic to penicillin\n\n[URGENT] Call 999 if:\n* You cannot breathe\n* Your face or throat swells up"},
    {"i": "Transform this eviction notice into Easy Read format.",
     "inp": "SECTION 21 NOTICE: You must leave by 01/08/2026. If you do not leave, court proceedings may be issued. You are advised to seek legal advice. Citizens Advice or Shelter can provide free assistance.",
     "out": "[URGENT] About your home\n\nYour landlord wants you to leave.\n\n[DATE] You must leave by: August 1, 2026\n\n[WARNING] If you do not leave:\n* They can take you to court\n\n[CALL] Get FREE help:\n* Citizens Advice\n* Shelter\n\n[OK] You might NOT have to leave.\nGet advice now."},
]

for s in SYNTH:
    all_pairs.append({"instruction": s["i"], "input": s["inp"], "output": s["out"]})
    all_pairs.append({"instruction": "Convert to Easy Read. Max 10 words per sentence. Use markers. Action checklist.", "input": s["inp"], "output": s["out"]})
print(f"  -> {len(SYNTH)*2} synthetic pairs")
print(f"\nTOTAL: {len(all_pairs)} training pairs")

random.seed(42)
random.shuffle(all_pairs)
eval_size = min(200, len(all_pairs)//10)
train_data = all_pairs[:-eval_size] if eval_size else all_pairs
eval_data = all_pairs[-eval_size:] if eval_size else []

with open("data/train.jsonl", "w") as f:
    for p in train_data: f.write(json.dumps(p) + "\n")
with open("data/eval.jsonl", "w") as f:
    for p in eval_data: f.write(json.dumps(p) + "\n")
print(f"Saved: {len(train_data)} train, {len(eval_data)} eval")

# 2. Fine-tuning
print("\n" + "="*60 + "\nSTEP 2: Fine-tuning\n" + "="*60)
from unsloth import FastLanguageModel
import torch

MODEL_NAME = "google/gemma-3-4b-it"
MAX_SEQ_LENGTH = 2048

print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True, dtype=None)
print("Model loaded!")

model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", use_gradient_checkpointing="unsloth", random_state=42)

TMPL = "<start_of_turn>user\n{instruction}\n\nDocument:\n{input}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
def fmt(ex):
    return {"text": [TMPL.format(instruction=i, input=inp, output=o) for i, inp, o in zip(ex["instruction"], ex["input"], ex["output"])]}

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
dataset = dataset.map(fmt, batched=True, remove_columns=dataset.column_names)
print(f"Training on {len(dataset)} examples")

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text", max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(output_dir="outputs", per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=10, num_train_epochs=3, learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25, optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
        seed=42, save_strategy="epoch", save_total_limit=2))

print("Training...")
stats = trainer.train()
print(f"Done! Loss: {stats.training_loss:.4f}")

# 3. Evaluation
print("\n" + "="*60 + "\nSTEP 3: Evaluation\n" + "="*60)
import textstat
TEST = [
    "Dear Patient, your blood test results indicate elevated HbA1c at 58 mmol/mol consistent with Type 2 Diabetes. Commence Metformin 500mg twice daily.",
    "NOTICE: Your tenancy will not be renewed. Vacate by 01/09/2026 or court proceedings will be issued.",
    "Your child needs immunisation at school on 22/04/2026. Return consent form by 15/04/2026.",
]

FastLanguageModel.for_inference(model)
for i, doc in enumerate(TEST):
    prompt = "<start_of_turn>user\nTransform into Easy Read format. Short sentences, markers, action checklist.\n\nDocument:\n" + doc + "<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, temperature=0.3, do_sample=True)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nDoc {i+1}: Grade {textstat.flesch_kincaid_grade(doc):.1f} -> {textstat.flesch_kincaid_grade(resp):.1f}")
    print(resp[:300])

# 4. Save and Push
print("\n" + "="*60 + "\nSTEP 4: Save and Push\n" + "="*60)
model.save_pretrained("medsimplify-gemma4")
tokenizer.save_pretrained("medsimplify-gemma4")
print("Model saved locally!")

try:
    model.save_pretrained_gguf("medsimplify-gguf", tokenizer, quantization_method="q4_k_m")
    print("GGUF exported!")
except Exception as e:
    print(f"GGUF export: {e}")

try:
    model.push_to_hub("kaushikss/medsimplify-gemma4", token="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")
    tokenizer.push_to_hub("kaushikss/medsimplify-gemma4", token="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")
    print("Pushed to HuggingFace Hub!")
except Exception as e:
    print(f"Hub push: {e}")

print("\n" + "="*60 + "\nPIPELINE COMPLETE!\n" + "="*60)
