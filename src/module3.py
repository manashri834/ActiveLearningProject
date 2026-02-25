import numpy as np
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

# ---------------- CONFIG ----------------
BATCH_SIZE = 32
FINAL_K = 200
SIMILARITY_THRESHOLD = 0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD SPLITS ----------------
labeled_indices = np.load("data/processed/labeled_indices.npy")
unlabeled_indices = np.load("data/processed/unlabeled_indices.npy")
clipped_indices = np.load("data/processed/clipped_indices.npy")

# ---------------- LOAD DATASET ----------------
dataset = load_dataset("ag_news")
train_dataset = dataset["train"].select(range(2000))

# Recreate unlabeled subset
unlabeled_subset = train_dataset.select(unlabeled_indices)
clipped_dataset = unlabeled_subset.select(clipped_indices)

# ---------------- LOAD TOKENIZER ----------------
tokenizer = DistilBertTokenizer.from_pretrained("models/initial_model")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_clipped = clipped_dataset.map(tokenize_function, batched=True)
tokenized_clipped.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ---------------- LOAD TRAINED MODEL ----------------
model = DistilBertForSequenceClassification.from_pretrained(
    "models/initial_model"
)

model.to(DEVICE)
model.eval()

# ---------------- GET EMBEDDINGS ----------------
loader = DataLoader(tokenized_clipped, batch_size=BATCH_SIZE)

embeddings = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embed = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embed.cpu().numpy())

embeddings = np.vstack(embeddings)

print("Embeddings shape:", embeddings.shape)

# ---------------- DENSITY ESTIMATION ----------------
similarity_matrix = cosine_similarity(embeddings)
density_scores = similarity_matrix.mean(axis=1)

dense_ranked = np.argsort(-density_scores)

# ---------------- DIVERSITY FILTER ----------------
selected = []

for idx in dense_ranked:
    if len(selected) == 0:
        selected.append(idx)
        continue

    sims = similarity_matrix[idx, selected]

    if np.max(sims) < SIMILARITY_THRESHOLD:
        selected.append(idx)

    if len(selected) >= FINAL_K:
        break

final_selected_indices = clipped_indices[selected]

np.save("data/processed/final_selected_indices.npy", final_selected_indices)

print("Final selected samples:", len(final_selected_indices))
print("Module 3 completed successfully.")