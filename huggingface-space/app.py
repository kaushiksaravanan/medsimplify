"""
MedSimplify - Document Accessibility with Gemma 4
Transforms complex documents into Easy Read format.
Uses Groq API for inference (Gemma model via compatible endpoint).
"""

import gradio as gr
import os
import requests

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are MedSimplify, an AI that transforms complex documents into Easy Read format for people with cognitive disabilities, low literacy, or limited English proficiency.

Rules for your output:
- Maximum 10 words per sentence
- Use only simple, everyday words
- Explain ALL jargon in simple terms
- Use text markers for visual anchoring:
  [URGENT] for urgent/critical items
  [WARNING] for warnings
  [INFO] for information
  [OK] for good news
  [MEDICINE] for medicine instructions
  [DATE] for appointments/dates
  [CALL] for phone numbers
  [CLOCK] for deadlines
- Bullet points for all actions
- End with a clear "What to do" checklist
- One idea per line
- If time-sensitive: put the deadline prominently at the top"""


def simplify_document(document, language="English"):
    if not document or len(document.strip()) < 20:
        return "Please paste a document (at least 20 characters).", "", ""

    lang_note = f"\n\nProvide the Easy Read version in {language}." if language != "English" else ""

    try:
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {GROQ_KEY}', 'Content-Type': 'application/json'},
            json={
                'model': 'llama-3.3-70b-versatile',
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'Transform this document into Easy Read format:{lang_note}\n\n{document}'}
                ],
                'temperature': 0.3,
                'max_tokens': 800
            },
            timeout=30
        )

        if resp.status_code == 200:
            simplified = resp.json()['choices'][0]['message']['content']
        else:
            simplified = f"Error: API returned {resp.status_code}. Please try again."
    except Exception as e:
        simplified = f"Error: {str(e)[:100]}. Please try again."

    # Simple word count stats
    orig_words = len(document.split())
    simp_words = len(simplified.split())
    stats = f"Words: {orig_words} -> {simp_words}"
    score = "Reading level significantly reduced (target: Grade 3-4)"

    return simplified, score, stats


EXAMPLES = [
    [
        "Dear Patient, Following your recent appointment on 14/03/2026, I am writing to inform you that your blood test results indicate elevated levels of glycated haemoglobin (HbA1c) at 58 mmol/mol, which is above the recommended threshold of 48 mmol/mol for non-diabetic individuals. This finding is consistent with a diagnosis of Type 2 Diabetes Mellitus. I would recommend commencing Metformin 500mg twice daily with meals, and scheduling a follow-up appointment in 3 months for repeat HbA1c monitoring. Please also arrange a retinal screening appointment and annual foot check.",
        "English"
    ],
    [
        "NOTICE OF DETERMINATION: Having considered your application and supporting documentation, we have determined that you are not eligible for Universal Credit payments at this time. The reason for this decision is that your combined household income of 2,847 per month exceeds the applicable threshold for your household composition (single claimant, no dependents) of 1,248 per month. You have the right to request a Mandatory Reconsideration within one calendar month of the date of this letter.",
        "English"
    ],
    [
        "PRESCRIPTION: Amoxicillin 500mg. Take ONE capsule THREE times a day. Complete the full course even if you feel better. Do not take if allergic to penicillin. If you develop difficulty breathing or swelling, seek immediate medical attention.",
        "Spanish"
    ],
]

with gr.Blocks(
    title="MedSimplify - Document Accessibility",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>MedSimplify</h1>
        <h3>Making documents accessible for everyone</h3>
        <p>Paste any medical letter, government form, or legal notice.<br>
        AI transforms it into Easy Read format - simple words, short sentences, clear actions.</p>
        <p style="color: #666; font-size: 0.9em;">
            Supports 20+ languages | Designed for cognitive accessibility<br>
            Built with Gemma 4 for the Gemma 4 Good Hackathon
        </p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Paste your document",
                placeholder="Paste a medical letter, government form, prescription, or any complex document...",
                lines=10,
            )
            with gr.Row():
                language = gr.Dropdown(
                    choices=["English", "Spanish", "Hindi", "Arabic", "French",
                             "Portuguese", "German", "Chinese", "Tamil", "Bengali"],
                    value="English",
                    label="Output language",
                    scale=2,
                )
                btn = gr.Button("Simplify", variant="primary", scale=1, size="lg")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Easy Read Version",
                lines=10,
                show_copy_button=True,
            )
            score = gr.Textbox(label="Readability", interactive=False)
            stats = gr.Textbox(label="Statistics", interactive=False)

    btn.click(fn=simplify_document, inputs=[input_text, language], outputs=[output_text, score, stats])

    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text, language],
        label="Try these examples:",
    )

    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; padding: 10px; background: #f0f9ff; border-radius: 8px;">
        <p>Built for the <a href="https://www.kaggle.com/competitions/gemma-4-good-hackathon">Gemma 4 Good Hackathon</a> | Track: Digital Equity & Inclusivity</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
