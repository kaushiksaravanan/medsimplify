"""
MedSimplify — Data Preparation Script
Downloads and formats training data for Gemma 4 fine-tuning.
Sources: Simple Wikipedia aligned pairs, Easy Read documents, MedlinePlus summaries.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_wiki_simple_pairs(max_pairs=5000):
    """Load Simple Wikipedia aligned with English Wikipedia for complex→simple pairs."""
    print("Loading Wiki Simple/Complex aligned pairs...")

    # wiki_lingua has aligned simple/complex summaries
    # Alternative: GEM/wiki_auto_asset_turk for sentence-level simplification
    ds = load_dataset("wiki_auto", "auto_acl", split="train", trust_remote_code=True)

    pairs = []
    for i, item in enumerate(ds):
        if i >= max_pairs:
            break
        if len(item.get("normal_sentence", "")) > 50 and len(item.get("simple_sentence", "")) > 20:
            pairs.append({
                "instruction": "Simplify this text for someone with a reading difficulty. Use short sentences (max 10 words). Replace jargon with simple words. Add a bullet-point summary of key actions.",
                "input": item["normal_sentence"],
                "output": item["simple_sentence"]
            })

    print(f"  → {len(pairs)} Wiki pairs collected")
    return pairs


def prepare_asset_turk_pairs(max_pairs=3000):
    """ASSET/TurkCorpus — human-written simplifications of English Wikipedia sentences."""
    print("Loading ASSET simplification dataset...")

    try:
        ds = load_dataset("asset", "ratings", split="simplification_1", trust_remote_code=True)
        pairs = []
        for i, item in enumerate(ds):
            if i >= max_pairs:
                break
            if "original" in item and "simplification" in item:
                pairs.append({
                    "instruction": "Rewrite this sentence in Easy Read format. Use simple words. One idea per sentence. Maximum 10 words per sentence.",
                    "input": item["original"],
                    "output": item["simplification"]
                })
        print(f"  → {len(pairs)} ASSET pairs collected")
        return pairs
    except Exception as e:
        print(f"  → ASSET failed: {e}, trying alternative...")
        return []


def prepare_medical_simplification(max_pairs=2000):
    """Medical text simplification pairs from COCHRANE/MedlinePlus aligned data."""
    print("Loading medical simplification pairs...")

    try:
        # MSD (Medical Simplification Dataset) or similar
        ds = load_dataset("mrjunos/MedQuAD-simplified", split="train", trust_remote_code=True)
        pairs = []
        for i, item in enumerate(ds):
            if i >= max_pairs:
                break
            pairs.append({
                "instruction": "Simplify this medical text for a patient with low health literacy. Explain all medical terms. Use short sentences. List any actions the patient needs to take.",
                "input": item.get("complex", item.get("question", "")),
                "output": item.get("simple", item.get("answer", ""))
            })
        print(f"  → {len(pairs)} medical pairs collected")
        return pairs
    except Exception as e:
        print(f"  → Medical dataset failed: {e}")
        return []


def prepare_synthetic_document_pairs():
    """
    Create synthetic document simplification examples from common templates.
    These represent the exact use case: medical letters, gov forms, prescriptions.
    """
    print("Creating synthetic document pairs...")

    templates = [
        {
            "input": "Dear Patient, Following your recent appointment on 14/03/2026, I am writing to inform you that your blood test results indicate elevated levels of glycated haemoglobin (HbA1c) at 58 mmol/mol, which is above the recommended threshold of 48 mmol/mol for non-diabetic individuals. This finding is consistent with a diagnosis of Type 2 Diabetes Mellitus. I would recommend commencing Metformin 500mg twice daily with meals, and scheduling a follow-up appointment in 3 months for repeat HbA1c monitoring. Please also arrange a retinal screening appointment and annual foot check. If you experience any gastrointestinal side effects, please contact the surgery.",
            "output": "Your doctor is writing about your blood test.\n\n🔴 Important result:\nYour blood sugar is too high.\nThis means you have Type 2 Diabetes.\n\n💊 What you need to do:\n• Take a new medicine called Metformin\n• Take 1 tablet in the morning with food\n• Take 1 tablet in the evening with food\n\n📅 Appointments to make:\n• See your doctor again in 3 months\n• Get your eyes checked\n• Get your feet checked\n\n⚠️ If the medicine makes your stomach hurt:\nCall your doctor's office."
        },
        {
            "input": "NOTICE OF DETERMINATION: Re: Your application for Universal Credit dated 02/02/2026. Having considered your application and supporting documentation, we have determined that you are not eligible for Universal Credit payments at this time. The reason for this decision is that your combined household income of £2,847 per month exceeds the applicable threshold for your household composition (single claimant, no dependents) of £1,248 per month. You have the right to request a Mandatory Reconsideration of this decision within one calendar month of the date of this letter. To do so, please write to the address above stating your reasons for disagreement.",
            "output": "This is about your Universal Credit application.\n\n❌ Decision:\nYou cannot get Universal Credit right now.\n\n📋 Why:\nYou earn too much money.\n• You earn: £2,847 a month\n• The limit is: £1,248 a month\n\n✋ If you disagree:\nYou can ask them to look at it again.\n• You must do this within 1 month\n• Write a letter saying why you disagree\n• Send it to the address on the top of their letter\n\n⏰ Deadline: 1 month from when you got this letter."
        },
        {
            "input": "PRESCRIPTION INFORMATION: Amoxicillin 500mg Capsules. Take ONE capsule THREE times a day at regular intervals. Complete the full course even if you feel better. Do not take if you are allergic to penicillin. Side effects may include: diarrhoea, nausea, skin rash. If you develop a severe allergic reaction (difficulty breathing, swelling of face/throat), seek immediate medical attention. Do not consume alcohol while taking this medication. Store below 25°C. Keep out of reach of children.",
            "output": "💊 Your Medicine: Amoxicillin\n\n📋 How to take it:\n• Take 1 tablet, 3 times a day\n• Morning, afternoon, and evening\n• Take ALL the tablets even if you feel better\n\n🚫 Do NOT take if:\n• You are allergic to penicillin\n\n⚠️ You might feel:\n• Sick to your stomach\n• Need to go to the toilet more\n• Get a skin rash\n\n🚨 Call 999 immediately if:\n• You cannot breathe properly\n• Your face or throat swells up\n\n🍺 No alcohol while taking this medicine\n\n📦 Keep in a cool place. Keep away from children."
        }
    ]

    pairs = []
    for t in templates:
        pairs.append({
            "instruction": "Transform this document into Easy Read format for someone with a cognitive disability. Use: short sentences (max 10 words each), emoji icons for visual anchoring, bullet points for actions, color-coded urgency (🔴 urgent, ⚠️ warning, 📋 information, ✅ good news). Extract all required actions into a clear checklist.",
            "input": t["input"],
            "output": t["output"]
        })

    print(f"  → {len(pairs)} synthetic document pairs created")
    return pairs


def save_dataset(pairs, filename="train.jsonl"):
    """Save pairs in JSONL format for Unsloth fine-tuning."""
    output_path = OUTPUT_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(pairs)} training examples to {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("MedSimplify — Training Data Preparation")
    print("=" * 60)

    all_pairs = []

    # Source 1: Wiki aligned pairs
    wiki_pairs = prepare_wiki_simple_pairs(max_pairs=4000)
    all_pairs.extend(wiki_pairs)

    # Source 2: ASSET/TurkCorpus
    asset_pairs = prepare_asset_turk_pairs(max_pairs=2000)
    all_pairs.extend(asset_pairs)

    # Source 3: Medical simplification
    medical_pairs = prepare_medical_simplification(max_pairs=2000)
    all_pairs.extend(medical_pairs)

    # Source 4: Synthetic document pairs (our exact use case)
    synthetic_pairs = prepare_synthetic_document_pairs()
    all_pairs.extend(synthetic_pairs)

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(all_pairs)} training pairs")
    print(f"{'=' * 60}")

    # Save
    save_dataset(all_pairs, "train.jsonl")

    # Also save a small eval set (last 200 pairs)
    if len(all_pairs) > 200:
        eval_pairs = all_pairs[-200:]
        train_pairs = all_pairs[:-200]
        save_dataset(train_pairs, "train.jsonl")
        save_dataset(eval_pairs, "eval.jsonl")
        print(f"Split: {len(train_pairs)} train, {len(eval_pairs)} eval")


if __name__ == "__main__":
    main()
