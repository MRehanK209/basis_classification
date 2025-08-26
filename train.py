import math
import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
import numpy as np

from utils import load_label_map, transform_data, evaluate_model, evaluate_hierarchical_model

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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=getattr(config, "weight_decay", 0.0))
    scaler = GradScaler() if getattr(config, "use_mixed_precision", False) else None

    grad_accum = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    steps_per_epoch = (len(train_data) + grad_accum - 1) // grad_accum
    total_steps = steps_per_epoch * int(config.epochs)
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

    # Hierarchy config
    child_to_parent = getattr(config, "child_to_parent", None)
    hier_lambda = float(getattr(config, "hierarchy_lambda", 0.0))

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
                    # hierarchy penalty: encourage child_logit <= parent_logit
                    if hier_lambda > 0.0 and child_to_parent:
                        c_idx = torch.tensor([c for c, _ in child_to_parent], device=outputs.device, dtype=torch.long)
                        p_idx = torch.tensor([p for _, p in child_to_parent], device=outputs.device, dtype=torch.long)
                        penalty = torch.relu(outputs[:, c_idx] - outputs[:, p_idx]).mean()
                        loss = loss + hier_lambda * penalty
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels.float())
                if hier_lambda > 0.0 and child_to_parent:
                    c_idx = torch.tensor([c for c, _ in child_to_parent], device=outputs.device, dtype=torch.long)
                    p_idx = torch.tensor([p for _, p in child_to_parent], device=outputs.device, dtype=torch.long)
                    penalty = torch.relu(outputs[:, c_idx] - outputs[:, p_idx]).mean()
                    loss = loss + hier_lambda * penalty
                loss.backward()

            total_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        avg_train_loss = total_loss / max(1, len(train_data))

        # Validation (pass constraints to eval)
        val_metrics, avg_val_loss, _, _ = evaluate_model(
            model, dev_data, device, criterion,
            desc="Evaluation on Validation Set",
            prediction_threshold=getattr(config, "prediction_threshold", 0.5),
            return_predictions=False,
            metrics_list=getattr(config, "evaluation_metrics", None),
            child_to_parent=getattr(config, "child_to_parent", None),
            enforce_hierarchy=bool(getattr(config, "enforce_hierarchy", False)),
        )

        history.append((avg_train_loss, avg_val_loss, val_metrics))

        current = val_metrics.get(monitor_metric, None)
        improved = current is not None and (best_metric is None or current > (best_metric + float(getattr(config, "min_delta", 0.0))))

        should_save_periodic = ((epoch + 1) % checkpoint_frequency == 0)
        should_save_best = improved

        if not save_best_only and should_save_periodic:
            ckpt_path = os.path.join(config.save_dir, f"epoch_{epoch+1}.pt")
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, ckpt_path)
        if should_save_best:
            best_metric = current; best_epoch = epoch + 1
            ckpt_path = os.path.join(config.save_dir, f"best_model_{best_epoch}.pt")
            torch.save({"epoch": best_epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_metric": best_metric, "monitor_metric": monitor_metric}, ckpt_path)
            no_improve = 0
        else:
            no_improve += 1

        if bool(getattr(config, "early_stopping_enabled", False)):
            patience = int(getattr(config, "early_stopping_patience", 5))
            if no_improve >= patience:
                if log_progress:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch}, best {monitor_metric}: {best_metric}")
                break

        if log_progress:
            mm = val_metrics.get(monitor_metric, None)
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | {monitor_metric}: {mm if mm is not None else 'n/a'}")

    # Test
    test_results = {}
    if test_data is not None:
        test_results, avg_test_loss, test_true, test_pred = evaluate_model(
            model, test_data, device, criterion,
            desc="Evaluation on Test Set",
            prediction_threshold=getattr(config, "prediction_threshold", 0.5),
            return_predictions=bool(getattr(config, "save_predictions", False)),
            metrics_list=getattr(config, "evaluation_metrics", None),
            child_to_parent=getattr(config, "child_to_parent", None),
            enforce_hierarchy=bool(getattr(config, "enforce_hierarchy", False)),
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


def train_hierarchical_model(model, train_data, dev_data, test_data, device, config):
    """
    Hierarchical-aware training with MCC optimization
    """
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' for sample-wise loss
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=getattr(config, "weight_decay", 0.0))
    scaler = GradScaler() if getattr(config, "use_mixed_precision", False) else None

    grad_accum = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    steps_per_epoch = (len(train_data) + grad_accum - 1) // grad_accum
    total_steps = steps_per_epoch * int(config.epochs)
    
    # Build scheduler (reuse existing function)
    from train import _build_scheduler
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

    # Hierarchical training settings - DEFINE hier_config FIRST
    hier_config = getattr(config, 'hierarchy_config', {})
    sequential_training = bool(getattr(config, "sequential_training", False))
    parent_epochs = int(getattr(config, "parent_epochs", 5))
    teacher_forcing = bool(getattr(config, "teacher_forcing", True))
    freeze_parent = bool(getattr(config, "freeze_parent", True))
    
    # Dynamic loss weighting based on hierarchy performance - NOW hier_config is defined
    parent_weight_initial = getattr(config, 'parent_weight', hier_config.get('parent_weight', 0.2))
    parent_weight_schedule = torch.linspace(parent_weight_initial, 0.1, int(config.epochs))
    
    # Best MCC tracking for early stopping
    best_mcc = -1.0
    patience_counter = 0
    mcc_history = []
    
    # Hierarchy config
    child_to_parent = getattr(config, "child_to_parent", None)
    hier_lambda = float(getattr(config, "hierarchy_lambda", 0.0))

    global_step = 0
    
    # NEW: Set up hierarchy mask if model supports it
    if hasattr(model, 'set_hierarchy_mask'):
        from utils import build_hierarchy_mask
        parent_rule = getattr(config, 'parent_rule', hier_config.get('parent_rule', 'before_dot'))
        
        # Get label maps from config
        parent_label_map = getattr(config, 'parent_label_map', {})
        child_label_map = getattr(config, 'child_label_map', {})
        
        if parent_label_map and child_label_map:
            parent_to_child = build_hierarchy_mask(
                parent_label_map, 
                child_label_map, 
                parent_rule
            )
            model.set_hierarchy_mask(parent_to_child)
            print(f"Set hierarchy mask with {len(parent_to_child)} parent-child relationships")
    
    # Phase 1: Parent-only training (if sequential)
    if sequential_training:
        if log_progress:
            print(f"Phase 1: Training parent classifier for {parent_epochs} epochs")
        
        for epoch in range(parent_epochs):
            model.train()
            total_loss = 0.0
            
            for step, batch in enumerate(tqdm(train_data, desc=f"Parent Training Epoch {epoch + 1}")):
                input_ids, attention_mask, parent_labels, child_labels = [b.to(device) for b in batch]
                
                optimizer.zero_grad()
                
                if getattr(config, "use_mixed_precision", False):
                    with autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, mode="parent_only")
                        loss = criterion(outputs['parent_logits'], parent_labels.float()).mean()
                    scaler.scale(loss).backward()
                    if (step + 1) % grad_accum == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        global_step += 1
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, mode="parent_only")
                    loss = criterion(outputs['parent_logits'], parent_labels.float()).mean()
                    loss.backward()
                    if (step + 1) % grad_accum == 0:
                        optimizer.step()
                        scheduler.step()
                        global_step += 1
                
                total_loss += loss.item()
                
                if (step + 1) % grad_accum == 0:
                    optimizer.zero_grad(set_to_none=True)
            
            avg_loss = total_loss / max(1, len(train_data))
            if log_progress:
                print(f"Parent Epoch {epoch + 1}/{parent_epochs} - Loss: {avg_loss:.4f}")
        
        # Freeze parent parameters if specified
        if freeze_parent:
            for param in model.parent_classifier.parameters():
                param.requires_grad = False
            if log_progress:
                print("Frozen parent classifier parameters")

    # Phase 2: Joint/Child training
    training_epochs = int(config.epochs) - (parent_epochs if sequential_training else 0)
    
    for epoch in range(training_epochs):
        # Dynamic parent weight (decrease over time to focus on child)
        if epoch < len(parent_weight_schedule):
            current_parent_weight = parent_weight_schedule[epoch].item()
            if hasattr(model, 'parent_weight'):
                model.parent_weight = current_parent_weight
        
        model.train()
        total_loss = 0.0
        parent_loss_sum = 0.0
        child_loss_sum = 0.0
        
        if log_progress:
            phase = "Joint" if not sequential_training else "Child"
            print(f"{phase} Training Epoch {epoch + 1}/{training_epochs}")

        for step, batch in enumerate(tqdm(train_data, desc=f"Training Epoch {epoch + 1}")):
            input_ids, attention_mask, parent_labels, child_labels = [b.to(device) for b in batch]

            if getattr(config, "use_mixed_precision", False):
                with autocast():
                    # Forward pass
                    parent_targets = parent_labels if teacher_forcing else None
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        mode="joint",
                        parent_targets=parent_targets
                    )
                    
                    # Compute hierarchical loss
                    loss_dict = model.compute_hierarchical_loss(
                        outputs['parent_logits'], 
                        outputs['child_logits'],
                        parent_labels, 
                        child_labels, 
                        criterion
                    )
                    
                    loss = loss_dict['total_loss']
                    if sequential_training and freeze_parent:
                        # Only child loss if parent is frozen
                        loss = loss_dict['child_loss']
                        
                scaler.scale(loss).backward()
            else:
                parent_targets = parent_labels if teacher_forcing else None
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    mode="joint",
                    parent_targets=parent_targets
                )
                
                loss_dict = model.compute_hierarchical_loss(
                    outputs['parent_logits'], 
                    outputs['child_logits'],
                    parent_labels, 
                    child_labels, 
                    criterion
                )
                
                loss = loss_dict['total_loss']
                if sequential_training and freeze_parent:
                    loss = loss_dict['child_loss']
                    
                loss.backward()

            total_loss += loss.item()
            parent_loss_sum += loss_dict['parent_loss'].item()
            child_loss_sum += loss_dict['child_loss'].item()

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
        avg_parent_loss = parent_loss_sum / max(1, len(train_data))
        avg_child_loss = child_loss_sum / max(1, len(train_data))

        # Validation
        from utils import evaluate_hierarchical_model_improved
        eval_thresholds = getattr(config, "evaluation_thresholds", [0.25])
        val_metrics, avg_val_loss = evaluate_hierarchical_model_improved(
            model, dev_data, device, criterion,
            desc="Hierarchical Validation",
            prediction_thresholds=eval_thresholds,
            return_predictions=False,
            metrics_list=getattr(config, "evaluation_metrics", None),
        )

        history.append((avg_train_loss, avg_val_loss, val_metrics, avg_parent_loss, avg_child_loss))

        # MCC-based early stopping and checkpointing
        current_mcc = val_metrics.get('mcc', 0.0)
        mcc_history.append(current_mcc)
        current = val_metrics.get(monitor_metric, None)
        improved = current is not None and (best_metric is None or current > (best_metric + float(getattr(config, "min_delta", 0.0))))

        if improved:
            best_metric = current
            best_epoch = epoch + 1
            ckpt_path = os.path.join(config.save_dir, f"best_model_{best_epoch}.pt")
            torch.save({
                "epoch": best_epoch, 
                "model_state": model.state_dict(), 
                "optimizer_state": optimizer.state_dict(), 
                "best_metric": best_metric, 
                "monitor_metric": monitor_metric
            }, ckpt_path)
            no_improve = 0
            
            # Also save best MCC model
            if current_mcc > best_mcc:
                best_mcc = current_mcc
                best_mcc_path = os.path.join(config.save_dir, f"best_mcc_model.pt")
                torch.save({
                    "epoch": epoch + 1, 
                    "model_state": model.state_dict(), 
                    "best_mcc": best_mcc,
                    "val_metrics": val_metrics
                }, best_mcc_path)
                print(f"New best MCC: {best_mcc:.4f} at epoch {epoch+1}")
        else:
            no_improve += 1

        # Early stopping
        if bool(getattr(config, "early_stopping_enabled", False)):
            patience = int(getattr(config, "early_stopping_patience", 5))
            if no_improve >= patience:
                if log_progress:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

        parent_f1 = val_metrics.get('parent_f1_macro', 0.0)
        child_f1 = val_metrics.get('child_f1_macro', 0.0)
        best_thresh = val_metrics.get('best_threshold', 0.25)
        current_mcc = val_metrics.get('mcc', 0.0)

        print(f"Train Loss: {avg_train_loss:.4f} (Parent: {avg_parent_loss:.4f}, Child: {avg_child_loss:.4f}) | Val Loss: {avg_val_loss:.4f}")
        print(f"Parent F1: {parent_f1:.4f}, Child F1: {child_f1:.4f}, MCC: {current_mcc:.4f}, Best Threshold: {best_thresh:.2f}")
        print(f"{monitor_metric}: {current if current is not None else 'n/a'}")

    # Test evaluation
    test_results = {}
    if test_data is not None:
        test_thresholds = getattr(config, "evaluation_thresholds", [0.25])
        test_results, avg_test_loss = evaluate_hierarchical_model_improved(
            model, test_data, device, criterion,
            desc="Hierarchical Test",
            prediction_thresholds=test_thresholds,
            return_predictions=False,
            metrics_list=getattr(config, "evaluation_metrics", None),
        )
        test_results["loss"] = float(avg_test_loss)

    # Save final checkpoint
    last_path = os.path.join(config.save_dir, "last.pt")
    torch.save({"model_state": model.state_dict()}, last_path)

    return model, history, test_results