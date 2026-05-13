# DECISIONS.md — Architectural Decision Log

| Timestamp | Decision | Chose | Over | Because |
|-----------|----------|-------|------|---------|
| 2026-05-13 00:00 | Model size | Gemma 4 9B | 26B/31B | Runs on free Kaggle T4 GPU, fast enough for live demo (<10s), quality sufficient for simplification task |
| 2026-05-13 00:00 | Fine-tuning method | Unsloth QLoRA (4-bit) | Full fine-tune / LoRA without Unsloth | 10x training speed, fits 16GB VRAM, qualifies for Unsloth $10K prize track |
| 2026-05-13 00:00 | Training data | Simple Wikipedia + Easy Read UK Gov + MedlinePlus | Custom-collected medical only | Available immediately, diverse document types, high-quality simplification pairs |
| 2026-05-13 00:00 | Deployment | HuggingFace Spaces (Gradio) | Vercel/Railway/Streamlit Cloud | Free GPU inference, public URL, no login required, instant deploy from git |
| 2026-05-13 00:00 | Local runtime | Ollama | llama.cpp direct / vLLM | Simplest UX for end users, qualifies for Ollama prize, one-command install |
| 2026-05-13 00:00 | OCR approach | Gemma 4 Vision (native multimodal) | Tesseract/PaddleOCR | Shows deep Gemma 4 usage, no external dependency, understands layout not just text |
| 2026-05-13 00:00 | UI framework | Gradio | Streamlit / Next.js | Fastest to prototype, native HuggingFace integration, GPU on Spaces |
| 2026-05-13 00:00 | Target reading level | Flesch-Kincaid Grade 3-4 | Grade 6-8 | Accessible to people with cognitive disabilities (Easy Read standard), not just "simpler" |
| 2026-05-13 00:00 | Prize strategy | Main + Digital Equity + Unsloth | Single track | Rules confirm Main + Special Tech stackable. Digital Equity is our natural fit. |
