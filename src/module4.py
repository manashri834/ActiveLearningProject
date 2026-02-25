import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return acc, prec, rec, f1


# ---------------------------------------------------
# ACTIVE LEARNING POOL UPDATE
# ---------------------------------------------------

def active_learning_update(labeled_indices, unlabeled_indices, selected_indices):

    new_labeled = np.concatenate([labeled_indices, selected_indices])
    new_unlabeled = np.setdiff1d(unlabeled_indices, selected_indices)

    return new_labeled, new_unlabeled