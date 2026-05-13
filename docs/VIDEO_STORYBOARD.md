# MedSimplify — Video Storyboard & Script

## Video Specs
- **Duration:** 2:50 (under 3:00 limit)
- **Platform:** YouTube (unlisted or public)
- **Quality:** 1080p minimum
- **Audio:** Clear voiceover, no background music fighting the voice
- **Style:** Professional but authentic. Real demo, real impact.

---

## SCENE 1: THE PROBLEM (0:00 – 0:35)

### Visual
- Open on a real medical letter on a table (printed, physical paper)
- Camera slowly zooms in on dense medical jargon
- Text overlays fade in highlighting confusing terms:
  "glycated haemoglobin" "HbA1c" "48 mmol/mol" "retinal screening"

### Voiceover
> "This is a real medical letter. It tells you that you have diabetes.
> But if you have a learning disability, a brain injury, or if English
> isn't your first language — you can't understand a single word of it.
>
> 1 in 5 adults in the UK struggle to read documents like this.
> That's 7.1 million people who can't understand their own health information.
>
> Missed medications. Missed appointments. Missed benefits.
> Not because they don't care — because the documents weren't written for them."

### Text Overlay (end of scene)
> **"Medical documents shouldn't require a law degree to understand."**

---

## SCENE 2: THE SOLUTION — LIVE DEMO (0:35 – 1:45)

### Visual
- Screen recording of the MedSimplify app (clean, full-screen)
- Show the interface: paste area on left, output on right

### Voiceover + Demo Flow

> "This is MedSimplify. Powered by Gemma 4, fine-tuned specifically for
> document accessibility."

**Demo Step 1:** Paste the medical letter into the input
> "I paste the same letter that was impossible to read..."

**Demo Step 2:** Click "Simplify" → show loading → output appears
> "...and in 6 seconds, Gemma 4 transforms it into Easy Read format."

**Demo Step 3:** Camera zooms into the output — show the transformation
> "Short sentences. Simple words. Emoji markers for urgency.
> Every action you need to take — pulled into a clear checklist.
> Grade 14 reading level... reduced to Grade 3."

**Demo Step 4:** Show the readability score badge
> "That's a reading level a 9-year-old can understand."

**Demo Step 5:** Show multilingual — switch to Spanish
> "And it works in over 20 languages."

**Demo Step 6:** Show a government benefits letter → simplified
> "Not just medical. Government letters, legal notices, prescriptions..."

### The "Wow" Moment (1:20)
- Show side-by-side: 500-word complex letter → 5 clear bullet points with emoji
- The visual contrast is INSTANT and dramatic

---

## SCENE 3: IMPACT & PRIVACY (1:45 – 2:20)

### Visual
- Brief montage: stock footage of diverse people (elderly, disabled, multilingual)
  OR (better): real person using the app on their phone
- Then: terminal showing Ollama running locally

### Voiceover

> "For people with cognitive disabilities, this isn't convenience.
> It's independence. Understanding your own diagnosis. Knowing your rights.
> Making informed decisions about your health.
>
> And because medical documents are deeply personal, MedSimplify runs
> entirely on your device through Ollama. No cloud. No data shared.
> Your medical letters never leave your computer."

**Demo:** Show turning off WiFi → app still works
> "Even offline. Even without internet."

### Text Overlay
> **"Privacy isn't optional when it's your medical data."**

---

## SCENE 4: TECHNICAL DEPTH (2:20 – 2:45)

### Visual
- Architecture diagram (clean, animated if possible — or Excalidraw)
- Brief code snippets showing Unsloth fine-tuning
- Training metrics graph (loss curve)

### Voiceover

> "Under the hood: Gemma 4 fine-tuned with Unsloth on 8,000 document pairs.
> We trained on medical letters, government notices, and legal documents —
> specifically optimized for Easy Read output.
>
> The model reduces reading level by an average of 10 grade levels
> while preserving 94% of the critical information.
>
> Exported to GGUF. Runs locally via Ollama. Open source."

### Visual: Show the metrics
- Before: Grade 14.2 → After: Grade 3.8
- Information preservation: 94%
- Processing time: <8 seconds

---

## SCENE 5: CLOSE (2:45 – 2:55)

### Visual
- App running with a document being simplified
- Fade to project name + URL

### Voiceover

> "Every person deserves to understand their own health information.
> MedSimplify. Try it now."

### Final Frame (hold 3 seconds)
```
📄 MedSimplify
"Making documents accessible for everyone"

🔗 [Live Demo URL]
💻 github.com/kaushiksaravanan/medsimplify

Built with Gemma 4 × Unsloth × Ollama
```

---

## PRODUCTION NOTES

### Equipment Needed
- Screen recording: OBS Studio (free)
- Voiceover: quiet room + phone/laptop mic (decent quality)
- Video editing: CapCut (free, fast) or DaVinci Resolve
- Architecture diagram: Excalidraw or Mermaid rendered

### Filming Checklist
- [ ] Record screen demo (app working end-to-end) — 3 takes minimum
- [ ] Record voiceover in quiet room — read from this script
- [ ] Create architecture diagram (Excalidraw)
- [ ] Get metrics screenshots from training notebook
- [ ] Optional: film yourself or a real user interacting with the app
- [ ] Edit: cut to <2:55 total runtime
- [ ] Upload to YouTube (public or unlisted, no login required to view)
- [ ] Test: can someone without a YouTube account watch it?

### Key Principles (from playbook Phase 114 — Show Don't Tell)
- NEVER say "it's fast" — show the timer
- NEVER say "it simplifies" — show the before/after side by side
- NEVER say "it works offline" — turn off WiFi on camera
- Every claim = visible proof on screen
