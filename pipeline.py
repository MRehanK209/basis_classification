#!/usr/bin/env python3
"""
Modular BK Classification Pipeline
Uses existing modules: data_preprocessing, random_baseline, train, model, utils
Supports configurable execution: preprocessing, baseline, training, or all
"""

import yaml
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# Imports
from data_preprocessing import load_and_preprocess_data
from random_baseline import evaluate_random_baseline_on_test
from train import train_model
from model import get_model
from utils import seed_everything, load_label_map, transform_data
import matplotlib.pyplot as plt

# Wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModularBKPipeline:
    """Modular BK Classification Pipeline using existing components"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._configure_logging()
        self._setup_directories()
        self._set_seed()
        self._init_wandb()

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Auto-experiment name if missing
        if config['system']['experiment_name'] is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            threshold = config['data']['frequency_threshold']
            config['system']['experiment_name'] = f"bk_thresh{threshold}_{timestamp}"
        return config

    def _configure_logging(self):
        level = getattr(logging, self.config['system'].get('log_level', 'INFO').upper(), logging.INFO)
        logger.setLevel(level)
        # Reset handlers
        logger.handlers = []
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        if self.config['logging'].get('log_to_file', False):
            fh = logging.FileHandler(self.config['logging'].get('log_file', 'pipeline.log'))
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)
        logger.info(f"Logging configured. Level={self.config['system'].get('log_level', 'INFO')}")

    def _set_seed(self):
        seed = self.config['split']['random_seed']
        deterministic = bool(self.config['system'].get('set_deterministic', True))
        seed_everything(seed, deterministic=deterministic)
        logger.info(f"Set random seed to {seed} (deterministic={deterministic})")

    def _init_wandb(self):
        if self.config['logging'].get('wandb', False) and WANDB_AVAILABLE:
            wandb.init(
                project=self.config['logging'].get('wandb_project', 'bk-classification'),
                name=self.config['system']['experiment_name'],
                config=self.config,
                tags=self._get_wandb_tags()
            )
            logger.info("Wandb initialized")
        elif self.config['logging'].get('wandb', False) and not WANDB_AVAILABLE:
            logger.warning("Wandb requested but not available. Install with: pip install wandb")

    def _get_wandb_tags(self) -> List[str]:
        tags = []
        execution = self.config.get('execution', {})
        if execution.get('run_preprocessing', False): tags.append('preprocessing')
        if execution.get('run_baseline', False): tags.append('baseline')
        if execution.get('run_training', False): tags.append('training')
        return tags if tags else ['pipeline']

    def _setup_directories(self):
        base_save = self.config['system']['save_dir']
        data_out = self.config['data']['output_dir']
        self.run_desc = self._model_desc()
        self.run_dir = os.path.join(base_save, self.run_desc)
        self.ckpt_dir = os.path.join(self.run_dir, 'checkpoints')
        for p in [data_out, base_save, self.run_dir, self.ckpt_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)

    def _model_desc(self) -> str:
        name = self.config['model']['name'].split('/')[-1].replace('/', '-')
        mtype = self.config['model']['model_type']
        bs = self.config['training']['batch_size']
        epochs = self.config['training']['epochs']
        sample = self.config['data'].get('sample_size', None)
        sample_tag = f"s{sample}" if sample is not None else "sALL"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{mtype}_{name}_bs{bs}_e{epochs}_{sample_tag}_{ts}"

    def run_pipeline(self):
        logger.info("=" * 60)
        logger.info("STARTING MODULAR BK CLASSIFICATION PIPELINE")
        logger.info("=" * 60)

        execution_config = self.config.get('execution', {})
        results = {}

        try:
            if execution_config.get('run_preprocessing', True):
                logger.info("Step 1: Running data preprocessing...")
                preprocessing_results = self._run_preprocessing()
                results['preprocessing'] = preprocessing_results
                self._log_to_wandb('preprocessing', preprocessing_results)

            if execution_config.get('run_baseline', True) and self.config['baseline'].get('run_random_baseline', False):
                logger.info("Step 2: Running uniform random baseline evaluation...")
                baseline_results = self._run_baseline()
                results['baseline'] = baseline_results
                self._log_to_wandb('baseline', baseline_results)
            else:
                logger.info("Step 2: Baseline evaluation skipped")
                results['baseline'] = {'status': 'skipped'}

            if execution_config.get('run_training', False):
                logger.info("Step 3: Running model training...")
                training_results = self._run_training()
                results['training'] = training_results
                self._log_to_wandb('training', training_results)

            self._save_pipeline_results(results)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Results saved to: {self.config['system']['save_dir']}")
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            if self.config['logging'].get('wandb', False) and WANDB_AVAILABLE:
                wandb.finish(exit_code=1)
            raise
        finally:
            if self.config['logging'].get('wandb', False) and WANDB_AVAILABLE:
                wandb.finish()

    def _run_preprocessing(self) -> Dict:
        logger.info("Using data_preprocessing module...")
        data_config = self.config['data']

        df_processed, label_map, _ = load_and_preprocess_data(
            frequency_threshold=data_config['frequency_threshold'],
            data_source_path=data_config['raw_data_path'],
            output_data_path=os.path.join(data_config['output_dir'], 'k10plus_processed_rare_label_removed.csv'),
            label_map_path=os.path.join(data_config['output_dir'], 'label_map.json')
        )

        # Optional sampling
        sample_size = data_config.get('sample_size', None)
        if sample_size is not None and sample_size < len(df_processed):
            logger.info(f"Sampling {sample_size} records from {len(df_processed)} total records")
            df_processed = df_processed.sample(n=sample_size, random_state=self.config['split']['random_seed'])
        else:
            logger.info(f"Using entire dataset: {len(df_processed)} records")

        # Split
        from sklearn.model_selection import train_test_split
        train_ratio = self.config['split']['train_ratio']
        val_ratio = self.config['split']['val_ratio']
        test_ratio = self.config['split']['test_ratio']
        random_seed = self.config['split']['random_seed']

        train_df, temp_df = train_test_split(
            df_processed,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            shuffle=True
        )
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_test_ratio),
            random_state=random_seed,
            shuffle=True
        )

        # Save splits
        out_dir = data_config['output_dir']
        train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

        # Update num_labels
        self.config['model']['num_labels'] = len(label_map)

        results = {
            'total_samples': len(df_processed),
            'num_labels': len(label_map),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'frequency_threshold': data_config['frequency_threshold'],
            'sample_size_used': sample_size if sample_size is not None else len(df_processed)
        }
        if self.config['logging'].get('log_data_stats', False):
            logger.info(f"Preprocessing results: {results}")
        return results

    def _run_baseline(self) -> Dict:
        logger.info("Running random baseline on test set with progress...")
        config_obj = self._create_config_object()
        test_csv_path = os.path.join(self.config['data']['output_dir'], 'test.csv')
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"Test set not found: {test_csv_path}")

        baseline_results = evaluate_random_baseline_on_test(
            config_obj,
            test_csv_path,
            wandb_enabled=self.config['logging'].get('wandb', False) and WANDB_AVAILABLE
        )

        # Save baseline results
        baseline_dir = os.path.join(self.run_dir, 'baselines')
        os.makedirs(baseline_dir, exist_ok=True)
        with open(os.path.join(baseline_dir, 'random_baseline_test_results.json'), 'w') as f:
            json.dump(baseline_results, f, indent=2)
        return baseline_results

    def _run_training(self) -> Dict:
        """Run model training using existing train module"""
        logger.info("Using train module...")
        config_obj = self._create_config_object()

        # Load label map and data
        label_map = load_label_map(config_obj.label_map_path)
        config_obj.num_labels = len(label_map)

        data_out = self.config['data']['output_dir']
        train_df = pd.read_csv(os.path.join(data_out, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_out, 'val.csv'))
        test_df = pd.read_csv(os.path.join(data_out, 'test.csv'))

        logger.info(f"Loaded train split: {len(train_df)} samples")
        logger.info(f"Loaded val split: {len(val_df)} samples")

        # Transform data using preprocessing config
        preproc = self.config.get('preprocessing', {})
        adv = self.config.get('advanced', {})
        text_fields = preproc.get('text_fields', None)
        lowercase = bool(preproc.get('lowercase', False) and preproc.get('clean_text', False))
        rm_special = bool(preproc.get('remove_special_chars', False) and preproc.get('clean_text', False))
        raw_max_len = int(preproc.get('max_text_length', self.config['model']['max_length']))
        prefetch = adv.get('prefetch_factor', None)

        train_data = transform_data(
            train_df, label_map,
            max_length=config_obj.max_length,
            batch_size=config_obj.batch_size,
            model_name=config_obj.model_name,
            num_workers=getattr(config_obj, 'num_workers', 0),
            pin_memory=getattr(config_obj, 'pin_memory', False),
            prefetch_factor=prefetch,
            text_fields=text_fields,
            lowercase=lowercase,
            remove_special_chars=rm_special,
            raw_text_max_length=raw_max_len,
            shuffle=True,
        )
        dev_data = transform_data(
            val_df, label_map,
            max_length=config_obj.max_length,
            batch_size=config_obj.batch_size,
            model_name=config_obj.model_name,
            num_workers=getattr(config_obj, 'num_workers', 0),
            pin_memory=getattr(config_obj, 'pin_memory', False),
            prefetch_factor=prefetch,
            text_fields=text_fields,
            lowercase=lowercase,
            remove_special_chars=rm_special,
            raw_text_max_length=raw_max_len,
            shuffle=False,
        )
        test_data = transform_data(
            test_df, label_map,
            max_length=config_obj.max_length,
            batch_size=config_obj.batch_size,
            model_name=config_obj.model_name,
            num_workers=getattr(config_obj, 'num_workers', 0),
            pin_memory=getattr(config_obj, 'pin_memory', False),
            prefetch_factor=prefetch,
            text_fields=text_fields,
            lowercase=lowercase,
            remove_special_chars=rm_special,
            raw_text_max_length=raw_max_len,
            shuffle=False,
        )

        # Initialize model
        import torch
        model = get_model(config_obj.model_type, num_labels=config_obj.num_labels)
        device = torch.device("cuda" if config_obj.use_gpu else "cpu")
        model = model.to(device)

        # Optional gradient checkpointing
        try:
            if self.config.get('advanced', {}).get('gradient_checkpointing', False):
                if hasattr(model, 'bart') and hasattr(model.bart, 'gradient_checkpointing_enable'):
                    model.bart.gradient_checkpointing_enable()
        except Exception:
            pass

        logger.info(f"Training {config_obj.model_type} with {config_obj.num_labels} labels")
        logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

        # Train
        model, history, test_results = train_model(model, train_data, dev_data, test_data, device, config_obj)

        epochs = list(range(1, len(history) + 1))

        # history entry: (train_loss, val_loss, val_metrics_dict)
        train_losses = [h[0] for h in history]
        val_losses = [h[1] for h in history]
        # Pull selected metrics (guard missing keys)
        def series(key):
            return [h[2].get(key) if isinstance(h[2], dict) else None for h in history]
        val_subset = series('subset_accuracy')
        val_mcc = series('mcc')
        val_prec = series('precision_micro')
        val_recall = series('recall_micro')
        val_f1 = series('f1_micro')
        val_perc_macro = series('precision_macro')
        val_perc_recall = series('recall_macro')
        val_perc_f1 = series('f1_macro')
        val_acc = series('accuracy')

        # Plot losses
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        loss_plot_path = os.path.join(self.run_dir, "loss_plot.png")
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()

        # Plot validation metrics
        plt.figure(figsize=(10, 6))
        if any(v is not None for v in val_subset): plt.plot(epochs, val_subset, label="Subset Accuracy")
        if any(v is not None for v in val_prec):   plt.plot(epochs, val_prec,   label="Precision (micro)")
        if any(v is not None for v in val_recall): plt.plot(epochs, val_recall, label="Recall (micro)")
        if any(v is not None for v in val_f1):     plt.plot(epochs, val_f1,     label="F1 (micro)")
        if any(v is not None for v in val_perc_macro): plt.plot(epochs, val_perc_macro, label="Precision (macro)")
        if any(v is not None for v in val_perc_recall): plt.plot(epochs, val_perc_recall, label="Recall (macro)")
        if any(v is not None for v in val_perc_f1): plt.plot(epochs, val_perc_f1, label="F1 (macro)")
        if any(v is not None for v in val_mcc):    plt.plot(epochs, val_mcc,    label="MCC")
        if any(v is not None for v in val_acc):    plt.plot(epochs, val_acc,    label="Accuracy")
        plt.title("Validation Metrics over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        val_metrics_plot_path = os.path.join(self.run_dir, "val_metrics_plot.png")
        plt.tight_layout()
        plt.savefig(val_metrics_plot_path)
        plt.close()

        # Save test metrics
        test_metrics_path = os.path.join(self.run_dir, "test_metrics.json")
        with open(test_metrics_path, "w") as f:
            json.dump(test_results, f, indent=2)

        training_results = {
            'loss_plot': loss_plot_path,
            'val_metrics_plot': val_metrics_plot_path,
            'test_metrics_path': test_metrics_path,
            'total_epochs': len(history) if history else 0,
            'run_dir': self.run_dir,
        }
        logger.info("Training completed!")
        return training_results

    def _create_config_object(self):
        """Convert YAML config to argparse-like object for existing modules"""
        class ConfigObject:
            pass

        config_obj = ConfigObject()

        # Paths
        data_out = self.config['data']['output_dir']
        config_obj.data_path = os.path.join(data_out, 'k10plus_processed_rare_label_removed.csv')
        config_obj.label_map_path = os.path.join(data_out, 'label_map.json')

        # Split
        config_obj.sample_size = None
        config_obj.train_ratio = self.config['split']['train_ratio']

        # Model
        config_obj.model_type = self.config['model']['model_type']
        config_obj.num_labels = self.config['model']['num_labels']
        config_obj.model_name = self.config['model']['name']
        config_obj.max_length = self.config['model']['max_length']

        # Training
        config_obj.batch_size = int(self.config['training']['batch_size'])
        config_obj.epochs = int(self.config['training']['epochs'])
        config_obj.lr = float(self.config['training']['learning_rate'])
        config_obj.use_mixed_precision = bool(self.config['training'].get('use_mixed_precision', False))
        config_obj.weight_decay = float(self.config['training'].get('weight_decay', 0.0))
        config_obj.gradient_accumulation_steps = int(self.config['training'].get('gradient_accumulation_steps', 1))
        config_obj.warmup_steps = int(self.config['training'].get('warmup_steps', 0))
        config_obj.optimizer = self.config['training'].get('optimizer', 'adamw')
        config_obj.scheduler = self.config['training'].get('scheduler', 'none')

        # System
        config_obj.seed = self.config['split']['random_seed']
        config_obj.use_gpu = self.config['system']['use_gpu']
        config_obj.save_dir = self.ckpt_dir

        # Dataloader/advanced
        adv = self.config.get('advanced', {})
        config_obj.num_workers = int(adv.get('dataloader_num_workers', 0))
        config_obj.pin_memory = bool(adv.get('pin_memory', False))
        config_obj.prefetch_factor = int(adv.get('prefetch_factor', 2)) if 'prefetch_factor' in adv else None

        # Early stopping
        es = adv.get('early_stopping', {})
        config_obj.early_stopping_enabled = bool(es.get('enabled', False))
        config_obj.early_stopping_patience = int(es.get('patience', 5))
        config_obj.min_delta = float(es.get('min_delta', 0.0))

        # LR scheduler (advanced)
        lr_sched = adv.get('lr_scheduler', {})
        config_obj.lr_scheduler_type = lr_sched.get('type', self.config['training'].get('scheduler', 'none'))
        config_obj.warmup_ratio = float(lr_sched.get('warmup_ratio', 0.0) or 0.0)
        config_obj.min_lr = float(lr_sched.get('min_lr', 0.0) or 0.0)

        # Evaluation
        eval_conf = self.config.get('evaluation', {})
        config_obj.prediction_threshold = float(eval_conf.get('prediction_threshold', 0.5))
        config_obj.save_predictions = bool(eval_conf.get('save_predictions', False))
        config_obj.evaluation_metrics = eval_conf.get('metrics', None)

        # Checkpointing/monitoring
        sys_conf = self.config.get('system', {})
        config_obj.save_best_only = bool(sys_conf.get('save_best_only', False))
        config_obj.checkpoint_frequency = int(sys_conf.get('checkpoint_frequency', 1))
        config_obj.monitor_metric = sys_conf.get('monitor_metric', 'f1_macro')

        # Logging toggles
        log_conf = self.config.get('logging', {})
        config_obj.log_training_progress = bool(log_conf.get('log_training_progress', True))

        return config_obj

    def _log_to_wandb(self, step_name: str, results: Dict):
        if not (self.config['logging'].get('wandb', False) and WANDB_AVAILABLE):
            return
        wandb_logs = {f'{step_name}/{key}': value for key, value in results.items()}
        try:
            wandb.log(wandb_logs)
        except Exception:
            pass

    def _save_pipeline_results(self, results: Dict):
        results_path = os.path.join(self.run_dir, 'pipeline_results.json')

        def convert_types(obj):
            if isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        results_converted = convert_types(results)

        with open(results_path, 'w') as f:
            json.dump({
                'experiment_name': self.config['system']['experiment_name'],
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'results': results_converted
            }, f, indent=2)

        logger.info(f"Pipeline results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Modular BK Classification Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return

    pipeline = ModularBKPipeline(args.config)

    if args.dry_run:
        print("Configuration:")
        print(yaml.dump(pipeline.config, indent=2))
        return

    try:
        pipeline.run_pipeline()
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()