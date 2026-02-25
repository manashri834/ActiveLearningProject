import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------
def evaluate_model(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return accuracy, precision, recall, f1


# ---------------------------------------------------
# ACTIVE LEARNING POOL UPDATE
# ---------------------------------------------------
def active_learning_update(labeled_indices, unlabeled_indices, selected_indices):

    new_labeled = np.concatenate([labeled_indices, selected_indices])
    new_unlabeled = np.setdiff1d(unlabeled_indices, selected_indices)

    print("New labeled size:", len(new_labeled))

    return new_labeled, new_unlabeled