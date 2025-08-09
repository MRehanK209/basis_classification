import math
import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
import numpy as np

from utils import load_label_map, transform_data, evaluate_model

def _build_scheduler(config, optimizer, total_steps: int):
    sched_type = getattr(config, "lr_scheduler_type", None) or getattr(config, "scheduler", None)
    warmup_steps = getattr(config, "warmup_steps", 0)
    warmup_ratio = getattr(config, "warmup_ratio", None)
    min_lr = getattr(config, "min_lr", 0.0)
    base_lr = optimizer.param_groups[0]["lr"]
    lr_min_factor = min_lr / base_lr if base_lr > 0 else 0.0

    if warmup_ratio is not None and warmup_steps == 0:
        warmup_steps = int(max(0, warmup_ratio * total_steps))

    if not sched_type or sched_type == "none":
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    def lr_lambda(current_step: int):
        if warmup_steps and current_step < warmup_steps:
            return max(lr_min_factor, float(current_step) / max(1, warmup_steps))
        # progress after warmup in [0,1]
        denom = max(1, total_steps - warmup_steps)
        progress = float(current_step - warmup_steps) / denom
        progress = min(max(progress, 0.0), 1.0)
        if sched_type == "cosine":
            # cosine from 1.0 down to lr_min_factor
            return max(lr_min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif sched_type == "linear":
            return max(lr_min_factor, 1.0 - progress * (1.0 - lr_min_factor))
        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def train_model(model, train_data, dev_data, test_data, device, config):
    """
    Training loop with: weight_decay, grad accumulation, scheduler/warmup, early stopping,
    checkpointing, and configurable prediction threshold + metrics.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=getattr(config, "weight_decay", 0.0))
    scaler = GradScaler() if getattr(config, "use_mixed_precision", False) else None

    grad_accum = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    steps_per_epoch = (len(train_data) + grad_accum - 1) // grad_accum
    total_steps = steps_per_epoch * int(config.epochs)

    # Scheduler (supports cosine/linear, warmup_steps or warmup_ratio, min_lr)
    scheduler = _build_scheduler(config, optimizer, total_steps)

    history = []
    os.makedirs(config.save_dir, exist_ok=True)

    best_metric = None
    best_epoch = -1
    no_improve = 0
    monitor_metric = getattr(config, "monitor_metric", "f1_macro")
    save_best_only = bool(getattr(config, "save_best_only", False))
    checkpoint_frequency = int(getattr(config, "checkpoint_frequency", 1))
    log_progress = bool(getattr(config, "log_training_progress", True))

    global_step = 0
    for epoch in range(int(config.epochs)):
        model.train()
        total_loss = 0.0
        if log_progress:
            print(f"Epoch {epoch + 1}/{config.epochs}")

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_data, desc=f"Training Epoch {epoch + 1}")):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            if getattr(config, "use_mixed_precision", False):
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels.float())
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels.float())
                loss.backward()

            total_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        avg_train_loss = total_loss / max(1, len(train_data))

        # Validation
        val_metrics, avg_val_loss, _, _ = evaluate_model(
            model,
            dev_data,
            device,
            criterion,
            desc="Evaluation on Validation Set",
            prediction_threshold=getattr(config, "prediction_threshold", 0.5),
            return_predictions=False,
            metrics_list=getattr(config, "evaluation_metrics", None),
        )

        # Track history
        history.append((avg_train_loss, avg_val_loss, val_metrics))

        # Monitor
        current = val_metrics.get(monitor_metric, None)
        improved = current is not None and (best_metric is None or current > (best_metric + float(getattr(config, "min_delta", 0.0))))

        # Checkpointing
        should_save_periodic = ((epoch + 1) % checkpoint_frequency == 0)
        should_save_best = improved

        if not save_best_only and should_save_periodic:
            ckpt_path = os.path.join(config.save_dir, f"epoch_{epoch+1}.pt")
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, ckpt_path)
        if should_save_best:
            best_metric = current
            best_epoch = epoch + 1
            ckpt_path = os.path.join(config.save_dir,f"best_model_{best_epoch}.pt")
            torch.save({"epoch": best_epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_metric": best_metric, "monitor_metric": monitor_metric}, ckpt_path)
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if bool(getattr(config, "early_stopping_enabled", False)):
            patience = int(getattr(config, "early_stopping_patience", 5))
            if no_improve >= patience:
                if log_progress:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch}, best {monitor_metric}: {best_metric}")
                break

        if log_progress:
            # Compact progress print
            mm = val_metrics.get(monitor_metric, None)
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | {monitor_metric}: {mm if mm is not None else 'n/a'}")

    # Final test evaluation
    test_results = {}
    test_true = test_pred = None
    if test_data is not None:
        test_results, avg_test_loss, test_true, test_pred = evaluate_model(
            model,
            test_data,
            device,
            criterion,
            desc="Evaluation on Test Set",
            prediction_threshold=getattr(config, "prediction_threshold", 0.5),
            return_predictions=bool(getattr(config, "save_predictions", False)),
            metrics_list=getattr(config, "evaluation_metrics", None),
        )
        test_results["loss"] = float(avg_test_loss)

        # Optionally save predictions
        if getattr(config, "save_predictions", False):
            # Load label map for decoding indices to labels
            label_map = load_label_map(config.label_map_path)
            inv_labels = {idx: label for label, idx in label_map.items()}
            out_path = os.path.join(config.save_dir, "test_predictions.csv")
            with open(out_path, "w", encoding="utf-8") as f:
                # header
                f.write("index,true_labels,predicted_labels\n")
                for i in range(test_pred.shape[0]):
                    true_idxs = np.where(test_true[i] == 1)[0].tolist()
                    pred_idxs = np.where(test_pred[i] == 1)[0].tolist()
                    true_labels = "|".join(inv_labels[idx] for idx in true_idxs if idx in inv_labels)
                    pred_labels = "|".join(inv_labels[idx] for idx in pred_idxs if idx in inv_labels)
                    f.write(f"{i},{true_labels},{pred_labels}\n")

    # Always save last checkpoint
    last_path = os.path.join(config.save_dir, "last.pt")
    torch.save({"model_state": model.state_dict()}, last_path)

    return model, history, test_results