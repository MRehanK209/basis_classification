import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score
import json
import os
from tqdm import tqdm

def load_label_map(label_map_path):
    """Load label mapping from JSON file"""
    with open(label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_labels_to_binary(bk_list, label_map):
    """Convert BK codes to multi-hot binary labels"""
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

def transform_data(dataset, label_map, max_length=768, batch_size=32, model_name="facebook/bart-large", num_workers=0, pin_memory=False):
    """Transform dataset into DataLoader"""
    sentences = (
        "Title: " + dataset["Title"].fillna('') + "\n" +
        "Summary: " + dataset["Summary"].fillna('') + "\n" +
        "Keywords: " + dataset["Keywords"].fillna('') + "\n" +
        "LOC_Keywords: " + dataset["LOC_Keywords"].fillna('') + "\n" +
        "RVK: " + dataset["RVK"].fillna('')
    ).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    if 'BK' in dataset.columns:
        binary_labels = torch.tensor(convert_labels_to_binary(dataset["BK"].tolist(), label_map))
        dataset = TensorDataset(input_ids, attention_mask, binary_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)
    else:
        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=pin_memory)
    
    return dataloader

def accuracy_binary(predicted_labels_np, true_labels_np):
    # Global/micro metrics across all labels and samples
    y_true = true_labels_np.ravel()
    y_pred = predicted_labels_np.ravel()

    # Micro accuracy over all label positions (not per label)
    overall_accuracy = (true_labels_np == predicted_labels_np).mean()

    # Global MCC / precision / recall / F1 (binary on flattened arrays)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)

    # Subset accuracy (exact match per sample) remains last
    subset_accuracy = np.mean(np.all(true_labels_np == predicted_labels_np, axis=1))

    return overall_accuracy, mcc, precision, recall, f1, subset_accuracy

def evaluate_model(model, test_data, device, desc="Eval"):
    """Evaluate model on a dataloader and return global metrics"""
    all_pred, all_labels = [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_data, desc=desc, leave=False):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            predicted_labels = (probs > 0.5).int().cpu()
            all_pred.append(predicted_labels)
            all_labels.append(labels.cpu())

    all_predictions = torch.cat(all_pred, dim=0).numpy()
    all_true_labels = torch.cat(all_labels, dim=0).numpy()

    return accuracy_binary(all_predictions, all_true_labels)

def load_and_split_data(data_path, sample_size=100000, train_ratio=0.8, seed=42):
    """Load and split data into train/dev sets"""
    dataset = pd.read_csv(data_path).sample(sample_size)
    
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(np.arange(dataset.shape[0]))
    data_shuffled = dataset.iloc[shuffled_indices]
    train_size = int(train_ratio * len(data_shuffled))
    
    train_df = data_shuffled.iloc[:train_size]
    dev_df = data_shuffled.iloc[train_size:]
    
    return train_df, dev_df

def seed_everything(seed=11711):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True