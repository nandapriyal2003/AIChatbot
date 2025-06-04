# Mental Health Chatbot

This project is a mental health support chatbot built using the DialoGPT-small model from Hugging Face and deployed with Gradio.

---

## Features

- Conversational support using a pretrained or fine-tuned DialoGPT model
- Journaling mode that logs the conversation locally
- Automatic detection of stress-related keywords and delivery of breathing/grounding techniques
- Access to mental health resources and support hotlines

---

## How to Use

1. Type a message describing how you're feeling.
2. Optionally enable "Journaling Mode" to save your conversation entries.
3. If you express stress or anxiety, the bot will guide you through a simple grounding technique.
4. Click "View Journal" to read saved entries.
5. Click "Resources" to access mental health support links.

---

## Model Info

- Base model: `microsoft/DialoGPT-small`
- The model can be fine-tuned for mental health tone or used out of the box
- Runs with PyTorch backend; GPU not required

---

## Setup Instructions (for local use)

```bash
pip install -r requirements.txt
python ui.py
