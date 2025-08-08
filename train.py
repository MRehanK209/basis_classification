import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os

from model import get_model
from utils import load_label_map, load_and_split_data, transform_data, evaluate_model, seed_everything
from config import get_config

def train_model(model, train_data, dev_data, device, config):
    """Training loop with mixed precision"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scaler = GradScaler() if config.use_mixed_precision else None
    
    history = []
    os.makedirs(config.save_dir, exist_ok=True)
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        all_pred = []
        all_labels = []

        for batch in tqdm(train_data, desc=f"Training Epoch {epoch + 1}"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()

            if config.use_mixed_precision:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted_labels = (probs > 0.5).int().cpu().numpy()
            all_pred.extend(predicted_labels)
            all_labels.extend(labels.int().cpu().numpy())

        # Calculate metrics
        avg_train_loss = total_loss / len(train_data)
        train_metrics = evaluate_model(model, train_data, device)
        val_metrics = evaluate_model(model, dev_data, device)

        # Save checkpoint
        checkpoint_path = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

        # Print metrics
        print(f"\nEpoch {epoch+1}/{config.epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train Metrics: Acc={train_metrics[0]:.4f}, F1={train_metrics[4]:.4f}, MCC={train_metrics[1]:.4f}, Precision={train_metrics[2]:.4f}, Recall={train_metrics[3]:.4f}, Subset={train_metrics[5]:.4f}")
        print(f"Val Metrics: Acc={val_metrics[0]:.4f}, F1={val_metrics[4]:.4f}, MCC={val_metrics[1]:.4f}, Precision={val_metrics[2]:.4f}, Recall={val_metrics[3]:.4f}, Subset={val_metrics[5]:.4f}")
        
        history.append((avg_train_loss, train_metrics, val_metrics))

    return model, history

def main():
    config = get_config()
    seed_everything(config.seed)
    
    # Load data and labels
    label_map = load_label_map(config.label_map_path)
    config.num_labels = len(label_map)
    
    train_df, dev_df = load_and_split_data(
        config.data_path, 
        sample_size=config.sample_size,
        train_ratio=config.train_ratio,
        seed=config.seed
    )
    
    # Transform data
    train_data = transform_data(train_df, label_map, 
                               max_length=config.max_length, 
                               batch_size=config.batch_size,
                               model_name=config.model_name)
    dev_data = transform_data(dev_df, label_map, 
                             max_length=config.max_length, 
                             batch_size=config.batch_size,
                             model_name=config.model_name)
    
    # Initialize model
    model = get_model(config.model_type, num_labels=config.num_labels)
    device = torch.device("cuda" if config.use_gpu else "cpu")
    model = model.to(device)
    
    print(f"Training {config.model_type} with {config.num_labels} labels")
    print(f"Training samples: {len(train_df)}, Validation samples: {len(dev_df)}")
    
    # Train model
    model, history = train_model(model, train_data, dev_data, device, config)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
