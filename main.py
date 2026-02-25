# main.py (clean final)
import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.module1 import module1_selection
from src.module2 import train_one_epoch, compute_uncertainty_scores
from src.module4 import evaluate_model, active_learning_update


# ---------------- CONFIG ----------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = os.path.join("data", "raw", "legal_text_classification.csv")

BATCH_SIZE = 8                 # CPU safe; if GPU use 16
LR = 5e-5
EPOCHS = 3
MAX_LENGTH = 128               # BIG SPEED BOOST (was 256)

INITIAL_LABELED_SIZE = 1000
FINAL_K = 200                 # reduce to 30–50 for faster demo

# IMPORTANT SPEED CONTROL (BIGGEST FIX)
MAX_UNLABELED_POOL = 1500       # only score/select from first 800 unlabeled


# ---------------- REPRODUCIBILITY ----------------
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Starting Active Learning Pipeline...")

# ---------------- LOAD CSV ----------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
df = df[["case_text", "case_outcome"]].dropna()

df = df.rename(columns={"case_text": "text", "case_outcome": "label"})
df["label"] = df["label"].astype(str).str.strip().str.lower()

df["label"] = df["label"].astype("category")
label_names = list(df["label"].cat.categories)
df["label"] = df["label"].cat.codes

NUM_LABELS = len(label_names)
print("Number of classes:", NUM_LABELS)

# OPTIONAL SPEED: for demo, sample only 3000 rows
# df = df.sample(n=3000, random_state=SEED).reset_index(drop=True)

# ---------------- TRAIN / TEST SPLIT ----------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["label"]
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# ---------------- INITIAL LABELED/UNLABELED SPLIT ----------------
all_indices = np.arange(len(train_dataset))
np.random.shuffle(all_indices)

if INITIAL_LABELED_SIZE >= len(all_indices):
    raise ValueError("INITIAL_LABELED_SIZE is too big for your training dataset.")

labeled_indices = all_indices[:INITIAL_LABELED_SIZE]
unlabeled_indices = all_indices[INITIAL_LABELED_SIZE:]

print(f"Initial labeled: {len(labeled_indices)} | unlabeled: {len(unlabeled_indices)}")

# ---------------- TOKENIZER + MODEL ----------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=NUM_LABELS
).to(DEVICE)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

# tokenize once
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ---------------- INITIAL TRAINING ----------------
print("\nTraining initial model...")
labeled_dataset = tokenized_train.select(labeled_indices)
train_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss}")

# ---------------- BASELINE EVALUATION ----------------
test_loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE, shuffle=False)

baseline_acc, baseline_prec, baseline_rec, baseline_f1 = evaluate_model(
    model, test_loader, DEVICE
)

print("\nBaseline Metrics:")
print("Accuracy:", baseline_acc)
print("Precision:", baseline_prec)
print("Recall:", baseline_rec)
print("F1 Score:", baseline_f1)

# ---------------- ACTIVE LEARNING (FAST VERSION) ----------------
print("\nComputing uncertainty (limited pool)...")

# Limit unlabeled pool (BIG FIX)
unlabeled_pool = unlabeled_indices[:MAX_UNLABELED_POOL]
unlabeled_pool_dataset = tokenized_train.select(unlabeled_pool)

# Score uncertainty on limited pool
uncertainty_scores = compute_uncertainty_scores(
    model,
    unlabeled_pool_dataset,
    DEVICE,
    BATCH_SIZE
)

print("Uncertainty scored:", len(uncertainty_scores))

print("\nSelecting new samples (Module 1) on limited pool...")

selected_indices = module1_selection(
    model=model,
    tokenized_unlabeled_dataset=tokenized_train,
    candidate_indices=unlabeled_pool,    # IMPORTANT: pass pool only
    device=DEVICE,
    final_k=FINAL_K,
)

selected_indices = np.array(selected_indices, dtype=int)
print("Selected samples:", len(selected_indices))

# ---------------- UPDATE POOLS ----------------
labeled_indices, unlabeled_indices = active_learning_update(
    labeled_indices=labeled_indices,
    unlabeled_indices=unlabeled_indices,
    selected_indices=selected_indices
)

print("New labeled size:", len(labeled_indices))
print("Remaining unlabeled:", len(unlabeled_indices))

# ---------------- RETRAIN ----------------
print("\nRetraining with expanded labeled set...")

updated_labeled_dataset = tokenized_train.select(labeled_indices)
updated_train_loader = DataLoader(updated_labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, updated_train_loader, optimizer, DEVICE)
    print(f"Retrain Epoch {epoch+1}/{EPOCHS}, Loss: {loss}")

# ---------------- UPDATED EVALUATION ----------------
updated_acc, updated_prec, updated_rec, updated_f1 = evaluate_model(
    model, test_loader, DEVICE
)

print("\nUpdated Metrics:")
print("Accuracy:", updated_acc)
print("Precision:", updated_prec)
print("Recall:", updated_rec)
print("F1 Score:", updated_f1)

print("\nAccuracy Improvement:", updated_acc - baseline_acc)

# ---------------- SAVE FINAL MODEL ----------------
os.makedirs("models/updated_model", exist_ok=True)
model.save_pretrained("models/updated_model")
tokenizer.save_pretrained("models/updated_model")

print("\nPipeline Completed Successfully.")
