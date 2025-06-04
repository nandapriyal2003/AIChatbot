import pandas as pd
import json

df = pd.read_csv("data/empathetic_dialogues.csv")

formatted = [
    {
        "input": f"<|startoftext|>{row['context']}",
        "output": f"{row['utterance']}<|endoftext|>"
    }
    for _, row in df.iterrows()
]

with open("data/formatted_data.json", "w") as f:
    json.dump(formatted, f, indent=2)

#print("Saved to data/formatted_data.json")
