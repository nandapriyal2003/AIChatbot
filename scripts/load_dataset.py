from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# Load the dataset
dataset = load_dataset("empathetic_dialogues")

# Save training split
df = pd.DataFrame(dataset["train"])
df[["context", "utterance"]].to_csv("data/empathetic_dialogues.csv", index=False)

#print("Saved to data/empathetic_dialogues.csv")