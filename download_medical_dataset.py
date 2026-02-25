from datasets import load_dataset
import pandas as pd
import os

print("Downloading PubMed RCT 20k...")

ds = load_dataset("armanc/pubmed-rct20k")

train_df = pd.DataFrame(ds["train"])

train_df = train_df.rename(columns={
    "text": "case_text",
    "label": "case_outcome"
})[["case_text", "case_outcome"]]

os.makedirs("data/raw", exist_ok=True)

train_df.to_csv("data/raw/legal_text_classification.csv", index=False)

print("Dataset saved successfully!")
print(train_df["case_outcome"].value_counts())