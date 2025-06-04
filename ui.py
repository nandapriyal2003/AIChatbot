import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# Load model and tokenizer
model_path = "microsoft/DialoGPT-small"  # Or your fine-tuned model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

chat_history_ids = None  # Track conversation context


# --- 📔 Journaling ---
def save_to_journal(user_msg, bot_reply):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("journal.txt", "a") as f:
        f.write(f"[{now}] USER: {user_msg}\n[{now}] BOT: {bot_reply}\n\n")


def read_journal():
    try:
        with open("journal.txt", "r") as f:
            content = f.read()
            return content if content.strip() else "📝 Your journal is empty."
    except FileNotFoundError:
        return "No journal found yet."


# --- 🌬️ Stress detection and grounding techniques ---
def is_stressed(text):
    keywords = ["stressed", "anxious", "overwhelmed", "panic", "can't breathe", "scared", "nervous", "worried"]
    return any(word in text.lower() for word in keywords)


def get_grounding_technique():
    return (
        "🧘 Let's try this together:\n"
        "- Breathe in for 4 seconds\n"
        "- Hold for 4 seconds\n"
        "- Breathe out for 4 seconds\n"
        "Repeat this 3 times. You're doing great. 🌱"
    )


# --- 🔗 Support resources ---
def get_resources():
    return (
        "### 🆘 Mental Health Resources\n"
        "- 📞 [U.S. 988 Suicide & Crisis Lifeline](https://988lifeline.org/)\n"
        "- 🌍 [Befrienders Worldwide – International Support](https://www.befrienders.org/)\n"
        "- 🧠 [Mental Health America](https://mhanational.org/)\n"
        "- 💬 If you're in immediate danger, call emergency services in your area.\n"
    )


# --- Main chatbot response logic ---
def respond(user_input, history, journaling_enabled):
    global chat_history_ids

    # 🌬️ If user seems stressed, skip model and respond supportively
    if is_stressed(user_input):
        calming_message = get_grounding_technique()
        if journaling_enabled:
            save_to_journal(user_input, calming_message)
            calming_message += "\n📝 Your entry has been saved to your journal."
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": calming_message})
        return history, ""

    # Otherwise: model-based generation
    new_input = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
    bot_input_ids = torch.cat([chat_history_ids, new_input], dim=-1) if chat_history_ids is not None else new_input

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=30,
        top_p=0.85,
        temperature=0.6
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    if journaling_enabled:
        save_to_journal(user_input, response)
        response += "\n📝 Your entry has been saved to your journal."

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history, ""


# --- Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Mental Health Support Chatbot")

    chatbot = gr.Chatbot(label="Chat", type="messages")
    msg = gr.Textbox(placeholder="How are you feeling today?", label="You")
    journal_toggle = gr.Checkbox(label="📝 Enable Journaling Mode", value=False)

    send = gr.Button("Send")
    clear = gr.Button("Clear Chat")
    view_btn = gr.Button("📔 View Journal")
    resources_btn = gr.Button("🔗 Resources")

    journal_output = gr.Textbox(label="📔 Your Journal", lines=10, interactive=False)
    resources_output = gr.Markdown(visible=False)

    # Event bindings
    send.click(respond, [msg, chatbot, journal_toggle], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg])
    view_btn.click(read_journal, None, journal_output)
    resources_btn.click(get_resources, None, resources_output).then(
        lambda: gr.update(visible=True), None, [resources_output]
    )
    gr.Markdown(
        """
        ---
        **Disclaimer:**  
        This chatbot is a supportive tool and is *not* a substitute for professional mental health care. If you are in crisis or require urgent help, please contact a licensed mental health provider or emergency services.
        """,
        elem_id="disclaimer"
    )

demo.launch()
