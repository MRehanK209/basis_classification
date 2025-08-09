import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score
import json
from tqdm import tqdm
import re
from typing import List, Dict, Optional, Tuple

def load_label_map(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_labels_to_binary(bk_list, label_map):
    label_count = len(label_map)
    binary_labels = []
    for entry in bk_list:
        label_vec = [0] * label_count
        labels = str(entry).split('|') if pd.notna(entry) else []
        for label in labels:
            if label in label_map:
                label_vec[label_map[label]] = 1
        binary_labels.append(label_vec)
    return binary_labels

def _build_sentences_from_fields(
    dataset: pd.DataFrame,
    text_fields: Optional[List[str]],
    lowercase: bool,
    remove_special_chars: bool,
    raw_text_max_length: Optional[int]
) -> List[str]:
    if not text_fields:
        text_fields = ["Title", "Summary", "Keywords", "LOC_Keywords", "RVK"]

    # Only keep fields that actually exist in the dataframe
    available_fields = [f for f in text_fields if f in dataset.columns]

    if available_fields:
        # Build one string per row from the available fields
        def build_row(row):
            parts = []
            for f in available_fields:
                val = row.get(f)
                val = "" if pd.isna(val) else str(val)
                parts.append(f"{f}: {val}")
            return " \n".join(parts)

        sentences = dataset.apply(build_row, axis=1).astype(str).tolist()
    else:
        # Fallback: combine known fields if custom ones are missing
        sentences = (
            "Title: " + dataset.get("Title", pd.Series('', index=dataset.index)).fillna('') + "\n" +
            "Summary: " + dataset.get("Summary", pd.Series('', index=dataset.index)).fillna('') + "\n" +
            "Keywords: " + dataset.get("Keywords", pd.Series('', index=dataset.index)).fillna('') + "\n" +
            "LOC_Keywords: " + dataset.get("LOC_Keywords", pd.Series('', index=dataset.index)).fillna('') + "\n" +
            "RVK: " + dataset.get("RVK", pd.Series('', index=dataset.index)).fillna('')
        ).astype(str).tolist()

    def clean_text(t: str) -> str:
        if lowercase:
            t = t.lower()
        if remove_special_chars:
            t = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß\s\:\,\.\-\_\|\;\(\)\/]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
        if raw_text_max_length is not None:
            t = t[:raw_text_max_length]
        return t

    return [clean_text(t) for t in sentences]

def transform_data(
    dataset: pd.DataFrame,
    label_map: Dict[str, int],
    max_length: int = 768,
    batch_size: int = 32,
    model_name: str = "facebook/bart-large",
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = None,
    text_fields: Optional[List[str]] = None,
    lowercase: bool = False,
    remove_special_chars: bool = False,
    raw_text_max_length: Optional[int] = None,
    shuffle: bool = True,
):
    sentences = _build_sentences_from_fields(
        dataset,
        text_fields=text_fields,
        lowercase=lowercase,
        remove_special_chars=remove_special_chars,
        raw_text_max_length=raw_text_max_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(
        sentences,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    if 'BK' in dataset.columns:
        binary_labels = torch.tensor(convert_labels_to_binary(dataset["BK"].tolist(), label_map))
        dataset_t = TensorDataset(input_ids, attention_mask, binary_labels)
        kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    else:
        dataset_t = TensorDataset(input_ids, attention_mask)
        kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset_t, **kwargs)

def evaluate_model(
    model,
    data_loader,
    device,
    criterion,
    desc: str = "Eval",
    prediction_threshold: float = 0.5,
    return_predictions: bool = False,
    metrics_list: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], float, Optional[np.ndarray], Optional[np.ndarray]]:
    all_pred, all_labels = [], []
    model.eval()
    val_total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels.to(device, non_blocking=True).float())
            val_total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted_labels = (probs > prediction_threshold).int().cpu()
            all_pred.append(predicted_labels)
            all_labels.append(labels.cpu())

    avg_val_loss = val_total_loss / max(1, len(data_loader))
    all_predictions = torch.cat(all_pred, dim=0).numpy()
    all_true_labels = torch.cat(all_labels, dim=0).numpy()

    # Metrics
    metrics = {}
    # Always available
    metrics["subset_accuracy"] = float(np.mean(np.all(all_true_labels == all_predictions, axis=1)))
    metrics["mcc"] = float(matthews_corrcoef(all_true_labels.ravel(), all_predictions.ravel()))
    # Micro
    metrics["precision_micro"] = float(precision_score(all_true_labels, all_predictions, average='micro', zero_division=0))
    metrics["recall_micro"] = float(recall_score(all_true_labels, all_predictions, average='micro', zero_division=0))
    metrics["f1_micro"] = float(f1_score(all_true_labels, all_predictions, average='micro', zero_division=0))
    # Macro
    metrics["precision_macro"] = float(precision_score(all_true_labels, all_predictions, average='macro', zero_division=0))
    metrics["recall_macro"] = float(recall_score(all_true_labels, all_predictions, average='macro', zero_division=0))
    metrics["f1_macro"] = float(f1_score(all_true_labels, all_predictions, average='macro', zero_division=0))
    # Overall accuracy over all positions (legacy)
    metrics["accuracy"] = float((all_true_labels == all_predictions).mean())

    if metrics_list:
        # Only keep requested ones
        metrics = {k: v for k, v in metrics.items() if k in metrics_list}

    return (metrics, avg_val_loss, (all_true_labels if return_predictions else None), (all_predictions if return_predictions else None))

def load_and_split_data(data_path, sample_size=100000, train_ratio=0.8, seed=42):
    dataset = pd.read_csv(data_path).sample(sample_size)
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(np.arange(dataset.shape[0]))
    data_shuffled = dataset.iloc[shuffled_indices]
    train_size = int(train_ratio * len(data_shuffled))
    train_df = data_shuffled.iloc[:train_size]
    dev_df = data_shuffled.iloc[train_size:]
    return train_df, dev_df

def seed_everything(seed=11711, deterministic: bool = True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic