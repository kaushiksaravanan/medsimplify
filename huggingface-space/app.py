"""
MedSimplify — HuggingFace Spaces App
Deploys to: https://huggingface.co/spaces/YOUR_USERNAME/medsimplify
"""

import gradio as gr
import textstat
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# === MODEL CONFIG ===
MODEL_ID = "YOUR_USERNAME/medsimplify-gemma4"  # UPDATE THIS after training

# Load with 4-bit quantization for Spaces free GPU (T4 16GB)
print("Loading MedSimplify model...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
)
print("Model loaded!")


# === PROMPTS ===
SIMPLIFY_PROMPT = """<start_of_turn>user
Transform this document into Easy Read format for someone with a cognitive disability.

Rules:
- Short sentences (maximum 10 words each)
- Simple words only — explain any jargon
- Use emoji icons: 🔴 urgent, ⚠️ warning, 📋 info, ✅ good, 💊 medicine, 📅 appointment, 📞 call
- Bullet points for all actions
- Extract ALL required actions into a clear checklist at the end
- One idea per line

Document:
{document}<end_of_turn>
<start_of_turn>model
"""


def simplify(document: str, language: str = "English") -> tuple[str, str, str]:
    """Core simplification function."""
    if not document or len(document.strip()) < 20:
        return "⚠️ Please paste a document (at least 20 characters).", "", ""

    # Add language instruction if not English
    lang_note = f"\n\nOutput in {language}." if language != "English" else ""
    prompt = SIMPLIFY_PROMPT.format(document=document) + lang_note

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Remove any trailing model tokens
    if "<end_of_turn>" in response:
        response = response.split("<end_of_turn>")[0].strip()

    # Readability metrics
    orig_grade = textstat.flesch_kincaid_grade(document)
    simp_grade = textstat.flesch_kincaid_grade(response) if len(response) > 20 else 0.0

    score_text = f"📊 Reading level: Grade {orig_grade:.1f} → Grade {simp_grade:.1f} (target: Grade 3-4)"
    stats_text = f"Words: {len(document.split())} → {len(response.split())} | Improvement: {orig_grade - simp_grade:.1f} grade levels"

    return response, score_text, stats_text


# === EXAMPLES ===
EXAMPLES = [
    [
        "Dear Patient, Following your recent appointment on 14/03/2026, I am writing to inform you that your blood test results indicate elevated levels of glycated haemoglobin (HbA1c) at 58 mmol/mol, which is above the recommended threshold of 48 mmol/mol for non-diabetic individuals. This finding is consistent with a diagnosis of Type 2 Diabetes Mellitus. I would recommend commencing Metformin 500mg twice daily with meals, and scheduling a follow-up appointment in 3 months for repeat HbA1c monitoring. Please also arrange a retinal screening appointment and annual foot check.",
        "English"
    ],
    [
        "NOTICE OF DETERMINATION: Having considered your application and supporting documentation, we have determined that you are not eligible for Universal Credit payments at this time. The reason for this decision is that your combined household income of £2,847 per month exceeds the applicable threshold for your household composition (single claimant, no dependents) of £1,248 per month. You have the right to request a Mandatory Reconsideration within one calendar month of the date of this letter.",
        "English"
    ],
    [
        "SECTION 21 NOTICE: You are hereby given notice that possession of the property at [address] is required after 01/08/2026. This notice is given under section 21(1)(b) of the Housing Act 1988. If you do not leave, court proceedings may be issued. You are advised to seek legal advice. Citizens Advice (citizensadvice.org.uk) or Shelter (shelter.org.uk) can help.",
        "English"
    ],
    [
        "PRESCRIPTION: Amoxicillin 500mg. Take ONE capsule THREE times a day. Complete the full course even if you feel better. Do not take if allergic to penicillin. If you develop difficulty breathing or swelling, seek immediate medical attention.",
        "Spanish"
    ],
]


# === INTERFACE ===
with gr.Blocks(
    title="MedSimplify — Document Accessibility",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>📄 MedSimplify</h1>
        <h3>Making documents accessible for everyone</h3>
        <p>Paste any medical letter, government form, or legal notice.<br>
        <b>Gemma 4</b> transforms it into Easy Read format — simple words, short sentences, clear actions.</p>
        <p style="color: #666; font-size: 0.9em;">
            🔒 Privacy: Can run locally via Ollama — documents never leave your device.<br>
            🌍 Supports 20+ languages | ♿ Designed for cognitive accessibility
        </p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="📄 Paste your document",
                placeholder="Paste a medical letter, government form, prescription, legal notice, or any complex document here...",
                lines=12,
                max_lines=20,
            )
            with gr.Row():
                language = gr.Dropdown(
                    choices=["English", "Spanish", "Hindi", "Arabic", "French",
                             "Portuguese", "German", "Chinese", "Japanese",
                             "Korean", "Tamil", "Bengali", "Urdu"],
                    value="English",
                    label="🌍 Output language",
                    scale=2,
                )
                btn = gr.Button("✨ Simplify", variant="primary", scale=1, size="lg")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="📋 Easy Read Version",
                lines=12,
                max_lines=20,
                show_copy_button=True,
            )
            score = gr.Textbox(label="Readability", interactive=False)
            stats = gr.Textbox(label="Statistics", interactive=False)

    btn.click(fn=simplify, inputs=[input_text, language], outputs=[output_text, score, stats])
    input_text.submit(fn=simplify, inputs=[input_text, language], outputs=[output_text, score, stats])

    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text, language],
        outputs=[output_text, score, stats],
        fn=simplify,
        cache_examples=False,
        label="📝 Try these real-world examples:",
    )

    gr.HTML("""
    <div style="text-align: center; margin-top: 30px; padding: 15px; background: #f0f9ff; border-radius: 10px;">
        <p><b>Built with Gemma 4</b> (fine-tuned with <a href="https://github.com/unslothai/unsloth">Unsloth</a>)
        for the <a href="https://www.kaggle.com/competitions/gemma-4-good-hackathon">Gemma 4 Good Hackathon</a></p>
        <p style="color: #666;">Track: Digital Equity & Inclusivity | Special Tech: Unsloth</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
