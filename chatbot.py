import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_path = "./model/dialoGPT-mental-health/checkpoint-500"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Enable GPU/MPS if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Start chatting
print("🧠 Mental Health Chatbot (type 'quit' to exit)")
chat_history_ids = None

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    new_input = tokenizer(user_input + tokenizer.eos_token, return_tensors='pt', padding=True)
    new_input_ids = new_input['input_ids']
    attention_mask = new_input['attention_mask']

    # Concatenate new input with previous history
    bot_input_ids = torch.cat([chat_history_ids, new_input], dim=-1) if chat_history_ids is not None else new_input

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        attention_mask=attention_mask 
    )

    # Decode and print last response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {response}")
