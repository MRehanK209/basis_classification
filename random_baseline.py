import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import get_model
from utils import load_label_map, transform_data, seed_everything
from config import get_config
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score
from utils import load_and_split_data, evaluate_model

# Add wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def evaluate_random_baseline_on_test(config, test_csv_path, wandb_enabled=False):
    """Evaluate randomly initialized model on test set batch by batch with progress"""
    print("=== RANDOM BASELINE EVALUATION ON TEST SET (WITH PROGRESS) ===")
    
    # Set seed for reproducibility
    seed_everything(config.seed)
    
    # Load label map
    label_map = load_label_map(config.label_map_path)
    config.num_labels = len(label_map)
    
    # Load test set directly
    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded test set: {len(test_df)} samples")
    
    # Transform test data to DataLoader
    test_data = transform_data(test_df, label_map, 
                              max_length=config.max_length, 
                              batch_size=config.batch_size,
                              model_name=config.model_name)
    
    # Initialize random model
    model = get_model(config.model_type, num_labels=config.num_labels)
    device = torch.device("cuda" if config.use_gpu else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model: {config.model_type}")
    print(f"Number of labels: {config.num_labels}")
    print(f"Test samples: {len(test_df)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Total batches: {len(test_data)}")
    
    # Collect all predictions and true labels
    all_predictions = []
    all_true_labels = []
    
    # Process batch by batch with progress bar (only console progress, no wandb logging)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_data, desc="Evaluating Random Baseline")):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Get random predictions (no training, just forward pass)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Convert logits to probabilities and then to binary predictions
            probs = torch.sigmoid(outputs)
            predicted_labels = (probs > 0.5).int()
            
            # Store predictions and labels
            all_predictions.append(predicted_labels.cpu())
            all_true_labels.append(labels)
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_labels = torch.cat(all_true_labels, dim=0)
    
    # Convert to numpy for metric calculation
    y_true = all_true_labels.numpy()
    y_pred = all_predictions.numpy()
    
    print(f"\nCalculating metrics on {len(y_true)} samples...")
    
    # Calculate all metrics according to your config
    metrics = {}
    
    # 1. Subset Accuracy (Exact Match)
    subset_accuracy = (y_true == y_pred).all(axis=1).mean()
    metrics['test_subset_accuracy'] = float(subset_accuracy)
    
    # 2. Matthews Correlation Coefficient (Global)
    mcc = matthews_corrcoef(y_true.ravel(), y_pred.ravel())
    metrics['test_mcc'] = float(mcc)
    
    # 3. Micro-averaged metrics
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    metrics['test_precision_micro'] = float(precision_micro)
    metrics['test_recall_micro'] = float(recall_micro)
    metrics['test_f1_micro'] = float(f1_micro)
    
    # 4. Macro-averaged metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['test_precision_macro'] = float(precision_macro)
    metrics['test_recall_macro'] = float(recall_macro)
    metrics['test_f1_macro'] = float(f1_macro)
    
    # Add metadata
    metrics['test_samples'] = len(test_df)
    metrics['num_labels'] = config.num_labels
    metrics['batch_size'] = config.batch_size
    metrics['total_batches'] = len(test_data)
    
    # Log ONLY final overall results to wandb (no batch progress)
    if wandb_enabled and WANDB_AVAILABLE:
        # Only log the final metrics, no progress tracking
        wandb_logs = {f'baseline_random/{key}': value for key, value in metrics.items()}
        wandb.log(wandb_logs)
        print("Final results logged to wandb")
    
    print("\n" + "="*50)
    print("RANDOM BASELINE TEST RESULTS")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float) and 'test_' in key:
            print(f"{key}: {value:.4f}")
        elif not isinstance(value, float):
            print(f"{key}: {value}")
    print("="*50)
    
    return metrics

# Keep original function for backward compatibility
def evaluate_random_baseline(config):
    """Original function - now deprecated, use evaluate_random_baseline_on_test_with_progress"""
    print("Warning: Using deprecated function. Consider using evaluate_random_baseline_on_test_with_progress")
    
    # Set seed for reproducibility
    seed_everything(config.seed)
    
    # Load data
    label_map = load_label_map(config.label_map_path)
    config.num_labels = len(label_map)
    
    train_df, dev_df = load_and_split_data(
        config.data_path, 
        sample_size=config.sample_size,
        train_ratio=config.train_ratio,
        seed=config.seed
    )
    
    # Transform data
    dev_data = transform_data(dev_df, label_map, 
                             max_length=config.max_length, 
                             batch_size=config.batch_size,
                             model_name=config.model_name)
    
    # Initialize random model
    model = get_model(config.model_type, num_labels=config.num_labels)
    device = torch.device("cuda" if config.use_gpu else "cpu")
    model = model.to(device)
    
    print(f"Model: {config.model_type}")
    print(f"Number of labels: {config.num_labels}")
    print(f"Evaluation samples: {len(dev_df)}")
    
    # Evaluate without training
    metrics = evaluate_model(model, dev_data, device)
    val_accuracy, val_mcc, val_precision, val_recall, val_f1, val_subset_accuracy = metrics
    
    print("\n=== RANDOM BASELINE RESULTS ===")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"MCC: {val_mcc:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1: {val_f1:.4f}")
    print(f"Subset Accuracy: {val_subset_accuracy:.4f}")
    
    return {
        'accuracy': val_accuracy,
        'mcc': val_mcc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'subset_accuracy': val_subset_accuracy
    }

if __name__ == "__main__":
    config = get_config()
    baseline_metrics = evaluate_random_baseline(config)