import torch
import numpy as np
from torch.utils.data import DataLoader


# ---------------------------------------------------
# TRAIN MODEL FOR ONE EPOCH
# ---------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ---------------------------------------------------
# COMPUTE ENTROPY-BASED UNCERTAINTY SCORES
# ---------------------------------------------------
import numpy as np
import torch
from torch.utils.data import DataLoader

def compute_uncertainty_scores(model, tokenized_unlabeled_dataset, device, batch_size=8):
    """
    Returns uncertainty score per sample using entropy.
    Output: np.array shape (N,)
    """
    model.eval()
    loader = DataLoader(tokenized_unlabeled_dataset, batch_size=batch_size, shuffle=False)

    scores = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)

            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            scores.extend(entropy.cpu().numpy())

    return np.array(scores)