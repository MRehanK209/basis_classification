import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
    label_column: str = "BK",   # NEW
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

    if label_column in dataset.columns:  # changed from 'BK'
        binary_labels = torch.tensor(convert_labels_to_binary(dataset[label_column].tolist(), label_map))
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
    child_to_parent: Optional[List[Tuple[int, int]]] = None,   # NEW
    enforce_hierarchy: bool = False,                            # NEW
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

    # Enforce hierarchy at inference (if parent index exists in the same label space)
    if enforce_hierarchy and child_to_parent:
        for c_idx, p_idx in child_to_parent:
            all_predictions[:, c_idx] = np.logical_and(all_predictions[:, c_idx], all_predictions[:, p_idx])

    # Metrics (same as before) ...
    metrics = {}
    metrics["subset_accuracy"] = float(np.mean(np.all(all_true_labels == all_predictions, axis=1)))
    metrics["mcc"] = float(matthews_corrcoef(all_true_labels.ravel(), all_predictions.ravel()))
    metrics["precision_micro"] = float(precision_score(all_true_labels, all_predictions, average='micro', zero_division=0))
    metrics["recall_micro"] = float(recall_score(all_true_labels, all_predictions, average='micro', zero_division=0))
    metrics["f1_micro"] = float(f1_score(all_true_labels, all_predictions, average='micro', zero_division=0))
    metrics["precision_macro"] = float(precision_score(all_true_labels, all_predictions, average='macro', zero_division=0))
    metrics["recall_macro"] = float(recall_score(all_true_labels, all_predictions, average='macro', zero_division=0))
    metrics["f1_macro"] = float(f1_score(all_true_labels, all_predictions, average='macro', zero_division=0))
    metrics["accuracy"] = float((all_true_labels == all_predictions).mean())

    if metrics_list:
        metrics = {k: v for k, v in metrics.items() if k in metrics_list}

    return (metrics, avg_val_loss, (all_true_labels if return_predictions else None), (all_predictions if return_predictions else None))

def evaluate_hierarchical_model(
    model,
    data_loader,
    device,
    criterion,
    desc: str = "Eval",
    prediction_threshold: float = 0.5,
    return_predictions: bool = False,
    metrics_list: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Evaluate hierarchical model on both parent and child predictions
    """
    all_parent_pred, all_parent_labels = [], []
    all_child_pred, all_child_labels = [], []
    model.eval()
    val_total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            input_ids, attention_mask, parent_labels, child_labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            parent_labels = parent_labels.to(device, non_blocking=True)
            child_labels = child_labels.to(device, non_blocking=True)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, mode="joint")
            
            # Compute loss
            loss_dict = model.compute_hierarchical_loss(
                outputs['parent_logits'], 
                outputs['child_logits'],
                parent_labels, 
                child_labels, 
                criterion
            )
            val_total_loss += loss_dict['total_loss'].item()
            
            # Parent predictions
            parent_probs = torch.sigmoid(outputs['parent_logits'])
            parent_predicted_labels = (parent_probs > prediction_threshold).int().cpu()
            all_parent_pred.append(parent_predicted_labels)
            all_parent_labels.append(parent_labels.cpu())
            
            # Child predictions
            child_probs = torch.sigmoid(outputs['child_logits'])
            child_predicted_labels = (child_probs > prediction_threshold).int().cpu()
            all_child_pred.append(child_predicted_labels)
            all_child_labels.append(child_labels.cpu())

    avg_val_loss = val_total_loss / max(1, len(data_loader))
    
    # Concatenate all predictions
    all_parent_predictions = torch.cat(all_parent_pred, dim=0).numpy()
    all_parent_true_labels = torch.cat(all_parent_labels, dim=0).numpy()
    all_child_predictions = torch.cat(all_child_pred, dim=0).numpy()
    all_child_true_labels = torch.cat(all_child_labels, dim=0).numpy()

    # Compute metrics for both levels
    parent_metrics = compute_metrics(all_parent_true_labels, all_parent_predictions, prefix="parent_")
    child_metrics = compute_metrics(all_child_true_labels, all_child_predictions, prefix="child_")
    
    # Combine metrics
    metrics = {**parent_metrics, **child_metrics}
    
    # Overall metrics (focus on child performance as main task)
    overall_metrics = compute_metrics(all_child_true_labels, all_child_predictions, prefix="")
    metrics.update(overall_metrics)

    if return_predictions:
        return metrics, avg_val_loss, all_child_true_labels, all_child_predictions
    else:
        return metrics, avg_val_loss

def compute_metrics(true_labels, predictions, prefix=""):
    """Compute evaluation metrics with optional prefix"""
    from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score
    
    metrics = {}
    metrics[f"{prefix}subset_accuracy"] = float(np.mean(np.all(true_labels == predictions, axis=1)))
    metrics[f"{prefix}mcc"] = float(matthews_corrcoef(true_labels.ravel(), predictions.ravel()))
    metrics[f"{prefix}precision_micro"] = float(precision_score(true_labels, predictions, average='micro', zero_division=0))
    metrics[f"{prefix}recall_micro"] = float(recall_score(true_labels, predictions, average='micro', zero_division=0))
    metrics[f"{prefix}f1_micro"] = float(f1_score(true_labels, predictions, average='micro', zero_division=0))
    metrics[f"{prefix}precision_macro"] = float(precision_score(true_labels, predictions, average='macro', zero_division=0))
    metrics[f"{prefix}recall_macro"] = float(recall_score(true_labels, predictions, average='macro', zero_division=0))
    metrics[f"{prefix}f1_macro"] = float(f1_score(true_labels, predictions, average='macro', zero_division=0))
    
    return metrics

    
def transform_hierarchical_data(
    dataset: pd.DataFrame,
    parent_label_map: Dict[str, int],
    child_label_map: Dict[str, int], 
    max_length: int = 768,
    batch_size: int = 32,
    model_name: str = "facebook/bart-large",
    text_fields: Optional[List[str]] = None,
    lowercase: bool = False,
    remove_special_chars: bool = False,
    raw_text_max_length: Optional[int] = None,
    shuffle: bool = True,
):
    """Transform data for hierarchical training with both parent and child labels"""
    
    # Build sentences (same as existing transform_data)
    sentences = _build_sentences_from_fields(
        dataset, text_fields, lowercase, remove_special_chars, raw_text_max_length
    )
    
    # Convert both parent (BK_TOP) and child (BK) labels to binary
    parent_labels = convert_labels_to_binary(dataset['BK_TOP'].tolist(), parent_label_map)
    child_labels = convert_labels_to_binary(dataset['BK'].tolist(), child_label_map)
    
    # Create dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_obj = HierarchicalBKDataset(sentences, parent_labels, child_labels, tokenizer, max_length)
    
    # Create dataloader
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': 0,
        'pin_memory': False
    }
    
    return DataLoader(dataset_obj, **kwargs)

class HierarchicalBKDataset(Dataset):
    """Dataset class for hierarchical BK classification with parent and child labels"""
    
    def __init__(self, sentences, parent_labels, child_labels, tokenizer, max_length):
        self.sentences = sentences
        self.parent_labels = parent_labels  # Binary vectors for parent labels
        self.child_labels = child_labels    # Binary vectors for child labels  
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        parent_label = self.parent_labels[idx]
        child_label = self.child_labels[idx]
        
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return (
            encoding['input_ids'].flatten(),
            encoding['attention_mask'].flatten(), 
            torch.tensor(parent_label, dtype=torch.long),
            torch.tensor(child_label, dtype=torch.long)
        )

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

def build_hierarchy_mask(parent_label_map, child_label_map, parent_rule="before_dot"):
    """
    Build hierarchical constraint mask based on label structure
    
    Args:
        parent_label_map: Dict mapping parent labels to indices
        child_label_map: Dict mapping child labels to indices  
        parent_rule: Rule for extracting parent from child ("before_dot" | "first_two_digits")
    
    Returns:
        Dict mapping parent_idx -> list of child_idx
    """
    parent_to_child = {}
    
    # Initialize empty lists for all parents
    for parent_label, parent_idx in parent_label_map.items():
        parent_to_child[parent_idx] = []
    
    # Map children to parents based on rule
    for child_label, child_idx in child_label_map.items():
        if parent_rule == "before_dot":
            parent_label = child_label.split('.')[0] if '.' in child_label else child_label
        elif parent_rule == "first_two_digits":
            parent_label = child_label[:2] if len(child_label) >= 2 else child_label
        else:
            continue
            
        if parent_label in parent_label_map:
            parent_idx = parent_label_map[parent_label]
            parent_to_child[parent_idx].append(child_idx)
    
    return parent_to_child

def evaluate_hierarchical_model_improved(
    model,
    data_loader,
    device,
    criterion,
    desc: str = "Eval",
    prediction_thresholds: list = [0.25],
    return_predictions: bool = False,
    metrics_list: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Improved evaluation with multiple thresholds and better metrics
    """
    all_parent_pred, all_parent_labels = [], []
    all_child_pred, all_child_labels = [], []
    model.eval()
    val_total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            input_ids, attention_mask, parent_labels, child_labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            parent_labels = parent_labels.to(device, non_blocking=True)
            child_labels = child_labels.to(device, non_blocking=True)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, mode="joint")
            
            # Compute loss
            loss_dict = model.compute_hierarchical_loss(
                outputs['parent_logits'], 
                outputs['child_logits'],
                parent_labels, 
                child_labels, 
                criterion
            )
            val_total_loss += loss_dict['total_loss'].item()
            
            # Store probabilities for multi-threshold evaluation
            parent_probs = torch.sigmoid(outputs['parent_logits']).cpu()
            child_probs = torch.sigmoid(outputs['child_logits']).cpu()
            all_parent_pred.append(parent_probs)
            all_child_pred.append(child_probs)
            all_parent_labels.append(parent_labels.cpu())
            all_child_labels.append(child_labels.cpu())

    avg_val_loss = val_total_loss / max(1, len(data_loader))
    
    # Concatenate all predictions and labels
    all_parent_probs = torch.cat(all_parent_pred, dim=0).numpy()
    all_parent_labels = torch.cat(all_parent_labels, dim=0).numpy()
    all_child_probs = torch.cat(all_child_pred, dim=0).numpy()
    all_child_labels = torch.cat(all_child_labels, dim=0).numpy()

    # Evaluate at multiple thresholds
    results = {}
    best_child_f1 = 0.0
    best_threshold = prediction_thresholds[0]
    
    for threshold in prediction_thresholds:
        # Convert probabilities to predictions
        parent_predictions = (all_parent_probs > threshold).astype(int)
        child_predictions = (all_child_probs > threshold).astype(int)
        
        # Compute metrics for both levels
        parent_metrics = compute_metrics(all_parent_labels, parent_predictions, prefix=f"parent_t{threshold}_")
        child_metrics = compute_metrics(all_child_labels, child_predictions, prefix=f"child_t{threshold}_")
        
        # Track best threshold based on child F1
        child_f1_key = f"child_t{threshold}_f1_macro"
        if child_f1_key in child_metrics and child_metrics[child_f1_key] > best_child_f1:
            best_child_f1 = child_metrics[child_f1_key]
            best_threshold = threshold
        
        # Add to results
        results.update(parent_metrics)
        results.update(child_metrics)
    
    # Set default metrics to best threshold
    best_parent_pred = (all_parent_probs > best_threshold).astype(int)
    best_child_pred = (all_child_probs > best_threshold).astype(int)
    
    default_parent_metrics = compute_metrics(all_parent_labels, best_parent_pred, prefix="parent_")
    default_child_metrics = compute_metrics(all_child_labels, best_child_pred, prefix="child_")
    overall_metrics = compute_metrics(all_child_labels, best_child_pred, prefix="")
    
    results.update(default_parent_metrics)
    results.update(default_child_metrics)
    results.update(overall_metrics)
    results['best_threshold'] = best_threshold

    if return_predictions:
        return results, avg_val_loss, all_child_labels, best_child_pred
    else:
        return results, avg_val_loss

# Replace the existing evaluate_hierarchical_model function
evaluate_hierarchical_model = evaluate_hierarchical_model_improved