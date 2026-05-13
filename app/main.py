"""
MedSimplify — Gradio Web Application
Document accessibility through Gemma 4 simplification.
Deploy to HuggingFace Spaces for free GPU inference.
"""

import gradio as gr
import textstat
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === MODEL LOADING ===
MODEL_PATH = "YOUR_HF_USERNAME/medsimplify-gemma4-9b"  # Update after upload

print("Loading MedSimplify model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded!")

# === CORE SIMPLIFICATION ===
SYSTEM_PROMPT = """You are MedSimplify, an AI that transforms complex documents into Easy Read format for people with cognitive disabilities.

Rules:
- Short sentences (maximum 10 words each)
- Simple words only — explain any jargon
- Use emoji icons for visual anchoring: 🔴 urgent, ⚠️ warning, 📋 info, ✅ good, 💊 medicine, 📅 appointment, 📞 call
- Bullet points for all actions
- Extract ALL required actions into a clear checklist at the end
- One idea per line
- Color code by type: urgent actions first, then information, then background

Transform the following document:"""


def simplify_document(text, language="English"):
    """Take complex text and return Easy Read version."""
    if not text or len(text.strip()) < 10:
        return "Please paste a document to simplify.", "", ""

    prompt = f"{SYSTEM_PROMPT}\n\n{text}\n\nEasy Read Version ({language}):"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Calculate readability scores
    original_grade = textstat.flesch_kincaid_grade(text)
    simplified_grade = textstat.flesch_kincaid_grade(response) if response else 0

    score_text = f"📊 Readability: Grade {original_grade:.1f} → Grade {simplified_grade:.1f}"

    return response, score_text, f"Original: {len(text.split())} words | Simplified: {len(response.split())} words"


# === EXAMPLE DOCUMENTS ===
EXAMPLES = [
    ["Dear Patient, Following your recent appointment on 14/03/2026, I am writing to inform you that your blood test results indicate elevated levels of glycated haemoglobin (HbA1c) at 58 mmol/mol, which is above the recommended threshold of 48 mmol/mol for non-diabetic individuals. This finding is consistent with a diagnosis of Type 2 Diabetes Mellitus. I would recommend commencing Metformin 500mg twice daily with meals, and scheduling a follow-up appointment in 3 months for repeat HbA1c monitoring.", "English"],
    ["NOTICE OF DETERMINATION: Having considered your application and supporting documentation, we have determined that you are not eligible for Universal Credit payments at this time. The reason for this decision is that your combined household income exceeds the applicable threshold for your household composition. You have the right to request a Mandatory Reconsideration within one calendar month.", "English"],
    ["PRESCRIPTION: Amoxicillin 500mg Capsules. Take ONE capsule THREE times a day at regular intervals. Complete the full course even if you feel better. Do not take if you are allergic to penicillin. Side effects may include: diarrhoea, nausea, skin rash. If you develop difficulty breathing or swelling of face/throat, seek immediate medical attention.", "English"],
]

# === GRADIO INTERFACE ===
with gr.Blocks(
    title="MedSimplify — Document Accessibility",
    theme=gr.themes.Soft(),
    css="""
    .output-text { font-size: 18px !important; line-height: 2 !important; }
    .header { text-align: center; margin-bottom: 20px; }
    """
) as demo:
    gr.HTML("""
    <div class="header">
        <h1>📄 MedSimplify</h1>
        <p><b>Making documents accessible for everyone.</b></p>
        <p>Paste any medical letter, government form, or legal notice.<br>
        Gemma 4 transforms it into Easy Read format — simple words, short sentences, clear actions.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="📄 Paste your document here",
                placeholder="Paste a medical letter, government form, prescription, or any complex document...",
                lines=10,
            )
            language = gr.Dropdown(
                choices=["English", "Spanish", "Hindi", "Arabic", "French", "Portuguese",
                         "German", "Chinese", "Japanese", "Korean"],
                value="English",
                label="🌍 Output language"
            )
            simplify_btn = gr.Button("✨ Simplify", variant="primary", size="lg")

        with gr.Column():
            output_text = gr.Textbox(
                label="📋 Easy Read Version",
                lines=12,
                elem_classes=["output-text"]
            )
            score_display = gr.Textbox(label="Readability Score", interactive=False)
            word_count = gr.Textbox(label="Word Count", interactive=False)

    simplify_btn.click(
        fn=simplify_document,
        inputs=[input_text, language],
        outputs=[output_text, score_display, word_count]
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text, language],
        label="📝 Try these examples:"
    )

    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; color: #666;">
        <p>🔒 <b>Privacy:</b> This app can run locally via Ollama — your documents never leave your device.</p>
        <p>Built with Gemma 4 (fine-tuned with Unsloth) for the Gemma 4 Good Hackathon.</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True)
