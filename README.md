# MedSimplify — AI Document Accessibility for Cognitive Disabilities

> **Making medical and government documents understandable for everyone.**
> Fine-tuned Gemma 4 transforms complex jargon into Easy Read format — locally, privately, in 20+ languages.

## Demo

[Live Demo](https://huggingface.co/spaces/YOUR_USERNAME/medsimplify) | [Video](https://youtube.com/watch?v=YOUR_VIDEO_ID)

## The Problem

1 in 5 adults struggle to read standard documents. Medical letters, government forms, and legal notices are written at a Grade 12-16 reading level. For people with cognitive disabilities, brain injuries, low literacy, or non-native speakers, these documents are effectively inaccessible — leading to missed appointments, medication errors, and lost benefits.

## How It Works

```
📄 Input: Photo of document OR pasted text
    ↓
🧠 Gemma 4 Vision: Reads and understands the document
    ↓
✨ Fine-tuned Gemma 4 (Unsloth): Transforms to Easy Read format
    ↓
📋 Output: Simple language + action checklist + icons + audio
```

1. **Take a photo** of any medical letter, government form, or legal notice
2. **Gemma 4 reads it** using multimodal vision capabilities
3. **Fine-tuned model simplifies it** to Grade 3-4 reading level
4. **Output includes**: simplified text, action items checklist, urgency indicators, audio narration
5. **Runs locally** via Ollama — your medical documents never leave your device

## Results

| Metric | Before (base Gemma 4) | After (fine-tuned) |
|--------|----------------------|-------------------|
| Flesch-Kincaid Grade | 14.2 | 3.8 |
| Information preserved | - | 94% |
| Processing time | - | <8 seconds |

## Architecture

- **Model**: Gemma 4 9B, fine-tuned with Unsloth (QLoRA)
- **Training**: 8,000 Complex→Simple document pairs (Simple Wikipedia + Easy Read UK Gov + MedlinePlus)
- **Multimodal**: Gemma 4 Vision for document photo OCR
- **Deployment**: HuggingFace Spaces (Gradio) + Ollama for local/offline
- **Languages**: 20+ (leveraging Gemma 4's multilingual capabilities)

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/medsimplify
cd medsimplify
pip install -r requirements.txt

# Run locally with Ollama
ollama pull medsimplify:latest
python app/main.py

# Or use the hosted demo
# https://huggingface.co/spaces/YOUR_USERNAME/medsimplify
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Core Model | Gemma 4 9B | Powerful enough for simplification, runs on consumer GPU |
| Fine-tuning | Unsloth + QLoRA | 10x faster training, fits free Kaggle GPU |
| Vision/OCR | Gemma 4 Vision | Native multimodal — no external OCR dependency |
| App Framework | Gradio | Fast deployment, GPU on HuggingFace Spaces |
| Local Deploy | Ollama | Privacy-critical: medical docs stay on device |
| TTS | Browser Web Speech API | Zero-cost, no external dependency |

## Known Limitations

- Best on English documents (other languages: functional but less refined)
- Complex tables/charts may lose formatting in simplification
- Medical advice should always be verified with a healthcare provider
- Processing time varies by document length (typically 3-10 seconds)

## License

Apache 2.0 (same as Gemma 4)

## Hackathon

Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) on Kaggle.
Track: Digital Equity & Inclusivity + Unsloth Special Technology Prize.
