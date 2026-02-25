import torch
import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.module1 import module1_selection
from src.module2 import train_one_epoch
from src.module4 import evaluate_model, active_learning_update

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 5e-5
EPOCHS = 2

print("Starting Active Learning Pipeline...")

# ---------------- LOAD DATA ----------------
import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

CSV_PATH = os.path.join("data", "raw", "legal_text_classification.csv")

df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
df = df[["case_text", "case_outcome"]].dropna()

df = df.rename(columns={"case_text": "text", "case_outcome": "label"})
df["label"] = df["label"].astype(str).str.strip().str.lower()

df["label"] = df["label"].astype("category")
label_names = list(df["label"].cat.categories)
df["label"] = df["label"].cat.codes

NUM_LABELS = len(label_names)
print("Number of classes:", NUM_LABELS)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# ---------------- INITIAL SPLIT ----------------
all_indices = np.arange(len(train_dataset))
np.random.shuffle(all_indices)

initial_labeled_size = 200

labeled_indices = all_indices[:initial_labeled_size]
unlabeled_indices = all_indices[initial_labeled_size:]

# ---------------- LOAD MODEL ----------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
num_labels=NUM_LABELS
)
model.to(DEVICE)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# ---------------- TOKENIZE ----------------
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ---------------- INITIAL TRAINING ----------------
print("Training initial model...")

labeled_dataset = tokenized_train.select(labeled_indices)
train_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
    print(f"Epoch {epoch+1}, Loss: {loss}")

# ---------------- BASELINE EVALUATION ----------------
test_loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE)
baseline_acc, baseline_prec, baseline_rec, baseline_f1 = evaluate_model(
    model, test_loader, DEVICE
)

print("\nBaseline Metrics:")
print("Accuracy:", baseline_acc)
print("Precision:", baseline_prec)
print("Recall:", baseline_rec)
print("F1 Score:", baseline_f1)
print("Baseline Accuracy:", baseline_acc)

# ---------------- MODULE 2: UNCERTAINTY RANKING ----------------
print("Computing uncertainty...")

from src.module2 import compute_uncertainty_scores

candidate_indices = compute_uncertainty_scores(
    model,
    tokenized_train.select(unlabeled_indices),
    DEVICE
)

# ---------------- MODULE 1: CLIP + DIVERSITY ----------------
print("Applying Module 1 selection...")

selected_indices = module1_selection(
    model,
    tokenized_train,
    unlabeled_indices,
    DEVICE,
    final_k=100
)

# ---------------- UPDATE POOLS ----------------
labeled_indices, unlabeled_indices = active_learning_update(
    labeled_indices,
    unlabeled_indices,
    selected_indices
)

# ---------------- RETRAIN ----------------
print("Retraining with new labeled data...")

updated_labeled_dataset = tokenized_train.select(labeled_indices)
train_loader = DataLoader(updated_labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
    print(f"Retrain Epoch {epoch+1}, Loss: {loss}")

# ---------------- FINAL EVALUATION ----------------
updated_acc, updated_prec, updated_rec, updated_f1 = evaluate_model(
    model, test_loader, DEVICE
)

print("\nUpdated Metrics:")
print("Accuracy:", updated_acc)
print("Precision:", updated_prec)
print("Recall:", updated_rec)
print("F1 Score:", updated_f1)

print("\nAccuracy Improvement:", updated_acc - baseline_acc)

print("Updated Accuracy:", updated_acc)
print("Improvement:", updated_acc - baseline_acc)

print("Pipeline Completed Successfully.")