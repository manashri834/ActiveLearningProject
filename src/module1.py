import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity


def extract_embeddings(model, dataset_subset, device, batch_size=32):
    model.eval()
    loader = DataLoader(dataset_subset, batch_size=batch_size)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            cls_embed = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embed.cpu().numpy())

    return np.vstack(embeddings)


def compute_uncertainty(model, dataset_subset, device, batch_size=32):
    model.eval()
    loader = DataLoader(dataset_subset, batch_size=batch_size)

    uncertainties = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)

            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            uncertainties.extend(entropy.cpu().numpy())

    return np.array(uncertainties)


def module1_selection(
    model,
    tokenized_unlabeled_dataset,
    candidate_indices,
    device,
    clip_percentile=95,
    final_k=100,
    similarity_threshold=0.95,
    batch_size=32
):
    """
    Module 1:
    1) compute uncertainty (entropy)
    2) clip extreme uncertainty (keep <= percentile)
    3) compute embeddings
    4) density ranking (mean cosine similarity)
    5) diversity filter (reject if too similar to selected)
    Always returns exactly final_k (or fewer only if pool is smaller).
    """

    candidate_indices = np.array(candidate_indices, dtype=int)

    # Safety: if pool is smaller than final_k
    if len(candidate_indices) <= final_k:
        return candidate_indices

    # 1) Uncertainty on candidate pool
    candidate_dataset = tokenized_unlabeled_dataset.select(candidate_indices)
    uncertainties = compute_uncertainty(model, candidate_dataset, device, batch_size=batch_size)

    # 2) Clipping
    clip_threshold = np.percentile(uncertainties, clip_percentile)
    filtered_mask = uncertainties <= clip_threshold
    filtered_indices = candidate_indices[filtered_mask]

    # If clipping removes everything, fall back to original pool
    if len(filtered_indices) == 0:
        filtered_indices = candidate_indices

    # If still too small, just return what we have
    if len(filtered_indices) <= final_k:
        return filtered_indices

    # 3) Embeddings
    filtered_dataset = tokenized_unlabeled_dataset.select(filtered_indices)
    embeddings = extract_embeddings(model, filtered_dataset, device, batch_size=batch_size)

    # 4) Density ranking
    similarity_matrix = cosine_similarity(embeddings)
    density_scores = similarity_matrix.mean(axis=1)
    ranked = np.argsort(-density_scores)

    # 5) Diversity filter
    selected = []
    for idx in ranked:
        if len(selected) == 0:
            selected.append(idx)
            continue

        sims = similarity_matrix[idx, selected]
        if np.max(sims) < similarity_threshold:
            selected.append(idx)

        if len(selected) >= final_k:
            break

    # ✅ Force-fill if diversity rejected too many
    if len(selected) < final_k:
        remaining = [i for i in ranked if i not in selected]
        selected.extend(remaining[: (final_k - len(selected))])

    selected = selected[:final_k]
    return filtered_indices[selected]