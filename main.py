# =========================
# main.py (Fast + Academic Version)
# =========================
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
from src.module2 import train_one_epoch
from src.module4 import evaluate_model, active_learning_update


# ---------------- CONFIG ----------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = os.path.join("data", "raw", "legal_text_classification.csv")

BATCH_SIZE = 8
LR = 5e-5
EPOCHS = 2
MAX_LENGTH = 128

# Better for showing improvement than 1000
INITIAL_LABELED_SIZE = 500

FINAL_K = 200
MAX_UNLABELED_POOL = 1000
NUM_ITERATIONS = 2

CLIP_PERCENTILE = 90
SIMILARITY_THRESHOLD = 0.95

# ---------------- REPRODUCIBILITY ----------------
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Starting Active Learning Pipeline...")
print("Device:", DEVICE)

# ---------------- LOAD CSV ----------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
df = df[["case_text", "case_outcome"]].dropna()

df = df.rename(columns={"case_text": "text", "case_outcome": "label"})
df["label"] = df["label"].astype(str).str.strip().str.lower()

# ---------------- TOP-4 CLASSES FILTER ----------------
top_labels = df["label"].value_counts().head(4).index.tolist()
print("Top-4 labels kept:", top_labels)

df = df[df["label"].isin(top_labels)].reset_index(drop=True)

label_to_id = {lab: i for i, lab in enumerate(top_labels)}
df["label"] = df["label"].map(label_to_id).astype(int)

NUM_LABELS = 4
print("Number of classes:", NUM_LABELS)
print("Class counts:\n", df["label"].value_counts().sort_index())

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
    INITIAL_LABELED_SIZE = max(50, int(0.1 * len(all_indices)))
    print("Adjusted INITIAL_LABELED_SIZE to:", INITIAL_LABELED_SIZE)

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

# Tokenize once
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ---------------- INITIAL TRAINING ----------------
print("\nTraining initial model...")
train_loader = DataLoader(tokenized_train.select(labeled_indices), batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LR)
for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

# ---------------- BASELINE EVALUATION ----------------
test_loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE, shuffle=False)
baseline_acc, baseline_prec, baseline_rec, baseline_f1 = evaluate_model(model, test_loader, DEVICE)

print("\nBaseline Metrics:")
print(f"Accuracy: {baseline_acc:.4f}")
print(f"Weighted Precision: {baseline_prec:.4f}")
print(f"Weighted Recall: {baseline_rec:.4f}")
print(f"Weighted F1: {baseline_f1:.4f}")

# ---------------- ACTIVE LEARNING LOOP ----------------
for it in range(1, NUM_ITERATIONS + 1):
    print("\n======================================")
    print(f" Active Learning Iteration {it}/{NUM_ITERATIONS}")
    print("======================================")

    if len(unlabeled_indices) == 0:
        print("No unlabeled samples left. Stopping.")
        break

    unlabeled_pool = unlabeled_indices[: min(MAX_UNLABELED_POOL, len(unlabeled_indices))]

    # Select samples using Module 1 (clipping + density + diversity)
    selected_indices = module1_selection(
        model=model,
        tokenized_unlabeled_dataset=tokenized_train,
        candidate_indices=unlabeled_pool,
        device=DEVICE,
        clip_percentile=CLIP_PERCENTILE,
        final_k=min(FINAL_K, len(unlabeled_pool)),
        similarity_threshold=SIMILARITY_THRESHOLD
    )

    selected_indices = np.array(selected_indices, dtype=int)

    # Safety: always ensure we add K samples (fill randomly if needed)
    target_k = min(FINAL_K, len(unlabeled_pool))
    if len(selected_indices) < target_k:
        remaining = np.setdiff1d(unlabeled_pool, selected_indices)
        if len(remaining) > 0:
            extra = remaining[: (target_k - len(selected_indices))]
            selected_indices = np.concatenate([selected_indices, extra])
    selected_indices = selected_indices[:target_k]

    print("Selected samples:", len(selected_indices))

    # Update pools
    labeled_indices, unlabeled_indices = active_learning_update(
        labeled_indices=labeled_indices,
        unlabeled_indices=unlabeled_indices,
        selected_indices=selected_indices
    )

    print("New labeled size:", len(labeled_indices))
    print("Remaining unlabeled:", len(unlabeled_indices))

    # Retrain
    updated_loader = DataLoader(tokenized_train.select(labeled_indices), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, updated_loader, optimizer, DEVICE)
        print(f"Iter {it} - Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

    # Evaluate after iteration
    acc, prec, rec, f1 = evaluate_model(model, test_loader, DEVICE)
    print(f"\nIteration {it} Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted Precision: {prec:.4f}")
    print(f"Weighted Recall: {rec:.4f}")
    print(f"Weighted F1: {f1:.4f}")

print("\nPipeline Completed Successfully.")

# ---------------- SAVE MODEL ----------------
os.makedirs("models/updated_model", exist_ok=True)
model.save_pretrained("models/updated_model")
tokenizer.save_pretrained("models/updated_model")
print("Saved model to: models/updated_model")