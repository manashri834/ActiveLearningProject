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

# ✅ Medical dataset CSV (clear name)
CSV_PATH = os.path.join("data", "raw", "pubmed_rct20k.csv")

BATCH_SIZE = 8
LR = 5e-5
EPOCHS = 3              # slightly higher for stability
MAX_LENGTH = 128

INITIAL_LABELED_SIZE = 500
FINAL_K = 100           # less aggressive = more stable
MAX_UNLABELED_POOL = 1000
NUM_ITERATIONS = 2

CLIP_PERCENTILE = 90
SIMILARITY_THRESHOLD = 0.95


# ---------------- HELPERS ----------------
def pct(x: float) -> float:
    return x * 100.0


def ensure_medical_csv(csv_path: str):
    """
    If csv_path does not exist, download PubMed RCT 20k (medical) and save it in the format:
      case_text, case_outcome
    """
    if os.path.exists(csv_path):
        return

    print("Medical CSV not found. Downloading PubMed RCT 20k dataset...")
    from datasets import load_dataset  # imported here to avoid forcing dependency if already exists

    ds = load_dataset("armanc/pubmed-rct20k")
    train_df = pd.DataFrame(ds["train"])

    # Convert into your expected format
    train_df = train_df.rename(columns={"text": "case_text", "label": "case_outcome"})[
        ["case_text", "case_outcome"]
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    train_df.to_csv(csv_path, index=False)

    print(f"✅ Medical dataset saved to: {csv_path}")
    print(train_df["case_outcome"].value_counts())


# ---------------- REPRODUCIBILITY ----------------
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Starting Active Learning Pipeline...")
print("Device:", DEVICE)

# ---------------- LOAD / CREATE MEDICAL CSV ----------------
ensure_medical_csv(CSV_PATH)

# ---------------- LOAD CSV ----------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
df = df[["case_text", "case_outcome"]].dropna()

df = df.rename(columns={"case_text": "text", "case_outcome": "label"})
df["label"] = df["label"].astype(str).str.strip().str.lower()

# ---------------- TOP-4 CLASSES FILTER ----------------
top_labels = df["label"].value_counts().head(4).index.tolist()
df = df[df["label"].isin(top_labels)].reset_index(drop=True)

label_to_id = {lab: i for i, lab in enumerate(top_labels)}
df["label"] = df["label"].map(label_to_id).astype(int)

NUM_LABELS = len(top_labels)

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

labeled_indices = all_indices[:INITIAL_LABELED_SIZE]
unlabeled_indices = all_indices[INITIAL_LABELED_SIZE:]

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

test_loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- INITIAL TRAINING ----------------
train_loader = DataLoader(
    tokenized_train.select(labeled_indices),
    batch_size=BATCH_SIZE,
    shuffle=True
)

optimizer = AdamW(model.parameters(), lr=LR)
for _ in range(EPOCHS):
    _ = train_one_epoch(model, train_loader, optimizer, DEVICE)

# ---------------- BASELINE (BEFORE AL) ----------------
baseline_acc, _, _, _ = evaluate_model(model, test_loader, DEVICE)
baseline_accuracy = baseline_acc

# ---------------- ACTIVE LEARNING LOOP ----------------
final_accuracy = baseline_accuracy  # fallback

for it in range(1, NUM_ITERATIONS + 1):
    if len(unlabeled_indices) == 0:
        break

    unlabeled_pool = unlabeled_indices[: min(MAX_UNLABELED_POOL, len(unlabeled_indices))]

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

    # Ensure we always add K samples
    target_k = min(FINAL_K, len(unlabeled_pool))
    if len(selected_indices) < target_k:
        remaining = np.setdiff1d(unlabeled_pool, selected_indices)
        if len(remaining) > 0:
            extra = remaining[: (target_k - len(selected_indices))]
            selected_indices = np.concatenate([selected_indices, extra])
    selected_indices = selected_indices[:target_k]

    labeled_indices, unlabeled_indices = active_learning_update(
        labeled_indices=labeled_indices,
        unlabeled_indices=unlabeled_indices,
        selected_indices=selected_indices
    )

    updated_loader = DataLoader(
        tokenized_train.select(labeled_indices),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = AdamW(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        _ = train_one_epoch(model, updated_loader, optimizer, DEVICE)

    acc, _, _, _ = evaluate_model(model, test_loader, DEVICE)
    final_accuracy = acc

# ---------------- FINAL SIMPLE DEMO OUTPUT ----------------
before = pct(baseline_accuracy)
after = pct(final_accuracy)
improve_points = after - before
improve_relative = (improve_points / before) * 100 if before > 0 else 0

sign_pts = "+" if improve_points >= 0 else ""
sign_rel = "+" if improve_relative >= 0 else ""

print("\n✅ Before Active Learning (500 labeled samples):")
print(f"Accuracy = {before:.2f}%\n")

print("✅ After Active Learning (900 labeled samples):")
print(f"Accuracy = {after:.2f}%\n")

print("📈 Improvement:")
print(f"{sign_pts}{improve_points:.2f} percentage points (≈ {sign_rel}{improve_relative:.0f}% vs before)")

# ---------------- SAVE MODEL ----------------
os.makedirs("models/updated_model", exist_ok=True)
model.save_pretrained("models/updated_model")
tokenizer.save_pretrained("models/updated_model")