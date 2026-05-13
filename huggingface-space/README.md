---
title: MedSimplify
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
python_version: "3.10"
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - gemma
  - accessibility
  - document-simplification
  - easy-read
  - cognitive-disability
short_description: "Gemma 4 transforms documents into Easy Read format"
---

# MedSimplify 📄

**Making medical and government documents accessible for everyone.**

Fine-tuned Gemma 4 transforms complex jargon into Easy Read format — locally, privately, in 20+ languages.

## What it does

Upload or paste any complex document (medical letter, government form, legal notice, prescription) and MedSimplify transforms it into Easy Read format:
- Short sentences (max 10 words)
- Simple words only
- Emoji icons for visual anchoring
- Action checklists extracted
- Available in 20+ languages

## Built for

People with cognitive disabilities, brain injuries, low literacy, or non-native speakers who need to understand important documents about their health, housing, and rights.

## Tech

- **Model**: Gemma 4, fine-tuned with Unsloth (QLoRA)
- **Training**: 8,000+ Complex→Simple document pairs
- **Privacy**: Runs locally via Ollama — documents never leave your device
- **Built for**: [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
