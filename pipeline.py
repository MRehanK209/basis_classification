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
from typing import Dict, List, Optional, Tuple
import warnings
import torch
import gc
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Imports
from data_preprocessing import load_and_preprocess_data
from random_baseline import evaluate_random_baseline_on_test
from train import train_model, train_hierarchical_model  # Add hierarchical import
from model import get_model
from utils import seed_everything, load_label_map, transform_data, convert_labels_to_binary, _build_sentences_from_fields
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

    def _find_best_checkpoint(self, ckpt_dir):
        cands = glob.glob(os.path.join(ckpt_dir, "best_model_*.pt"))
        return max(cands, key=os.path.getmtime) if cands else None

    def _load_backbone_from_checkpoint(self, model, ckpt_path: str):
    # Warm-start only the backbone (e.g., BART) and ignore classifier shape
        if not ckpt_path or not os.path.exists(ckpt_path):
            return model
        sd = torch.load(ckpt_path, map_location="cpu")
        state = sd.get("model_state", sd)
        # Keep only backbone parameters to avoid size mismatch on classifier
        backbone = {k: v for k, v in state.items() if k.startswith("bart.")}
        model.load_state_dict(backbone, strict=False)
        return model

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

        # Add BK_TOP column (top-level labels) for multi-stage stage 1
        def parent_code(code: str) -> str:
            rule = self.config.get('hierarchy', {}).get('parent_rule', 'before_dot')
            if rule == 'first_two_digits':
                return ''.join([ch for ch in code if ch.isdigit()])[:2]
            # default: before_dot
            return code.split('.')[0]

        def to_top_level_cell(bk_cell: str) -> str:
            if pd.isna(bk_cell) or not bk_cell:
                return ''
            parts = str(bk_cell).split('|')
            tops = sorted(set(parent_code(p) for p in parts if p))
            return '|'.join(tops)

        df_processed['BK_TOP'] = df_processed['BK'].apply(to_top_level_cell)

        # Build and save top-level label map
        all_top = set()
        for cell in df_processed['BK_TOP'].dropna():
            if cell:
                all_top.update(cell.split('|'))
        label_map_top = {label: idx for idx, label in enumerate(sorted(all_top))}
        with open(os.path.join(data_config['output_dir'], 'label_map_top.json'), 'w') as f:
            json.dump(label_map_top, f, indent=2)

        # Optional sampling
        sample_size = data_config.get('sample_size', None)
        if sample_size is not None and sample_size < len(df_processed):
            logger.info(f"Sampling {sample_size} records from {len(df_processed)} total records")
            df_processed = df_processed.sample(n=sample_size, random_state=self.config['split']['random_seed'])
        else:
            logger.info(f"Using entire dataset: {len(df_processed)} records")

        # Split
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

    def _run_single_stage(self, stage: str = '2') -> Dict:
        """
        Single-stage training wrapper.
        - stage '1' → top-level labels (BK_TOP, label_map_top.json)
        - stage '2' → full labels (BK, label_map.json)
        """
        import os
        stage_norm = str(stage).lower()
        data_out = self.config['data']['output_dir']

        if stage_norm in ('1', 'top', 'bk_top'):
            label_map_path = os.path.join(data_out, 'label_map_top.json')
            label_column = 'BK_TOP'
            save_dir = self.ckpt_dir  # or a dedicated dir if you prefer
        else:
            label_map_path = os.path.join(data_out, 'label_map.json')
            label_column = 'BK'
            save_dir = self.ckpt_dir

        return self._train_stage(
            label_map_path=label_map_path,
            label_column=label_column,
            save_dir=save_dir,
            warm_start_ckpt=None
        )

    def _run_training(self) -> Dict:
        logger.info("Using train module...")
        ft = self.config.get('fine_tuning', {})
        mode = ft.get('mode', 'standard')
        stage_cfg = str(ft.get('stage', '2')).lower()

        # NEW: Handle hierarchical joint training
        if mode == 'hierarchical_joint':
            return self._run_hierarchical_training()

        # Direct single-stage path (unchanged behavior)
        if mode != 'multi_stage' or stage_cfg in ('1', '2'):
            return self._run_single_stage(stage=stage_cfg)

        # Multi-stage in one go
        results = {}

        # Stage 1: top-level
        stage1_ckpt = os.path.join(self.run_dir, 'checkpoints_stage1')
        os.makedirs(stage1_ckpt, exist_ok=True)
        res1 = self._train_stage(
            label_map_path=os.path.join(self.config['data']['output_dir'], 'label_map_top.json'),
            label_column='BK_TOP',
            save_dir=stage1_ckpt,
            warm_start_ckpt=None
        )
        results['stage1'] = res1
        best1 = self._find_best_checkpoint(stage1_ckpt)

        # Stage 2: full labels, warm-start backbone from Stage 1
        stage2_ckpt = os.path.join(self.run_dir, 'checkpoints_stage2')
        os.makedirs(stage2_ckpt, exist_ok=True)
        res2 = self._train_stage(
            label_map_path=os.path.join(self.config['data']['output_dir'], 'label_map.json'),
            label_column='BK',
            save_dir=stage2_ckpt,
            warm_start_ckpt=best1
        )
        results['stage2'] = res2
        return results
    def _train_stage(self, label_map_path: str, label_column: str, save_dir: str, warm_start_ckpt: str | None = None) -> Dict:

        # 1) Build per-stage config
        config_obj = self._create_config_object()
        config_obj.label_map_path = label_map_path
        config_obj.save_dir = save_dir  # checkpoints for this stage

        # 2) Load label map and splits
        data_out = self.config['data']['output_dir']
        label_map = load_label_map(label_map_path)
        config_obj.num_labels = len(label_map)

        train_df = pd.read_csv(os.path.join(data_out, 'train.csv'))
        val_df   = pd.read_csv(os.path.join(data_out, 'val.csv'))
        test_df  = pd.read_csv(os.path.join(data_out, 'test.csv'))

        logger.info(f"[{label_column}] Loaded splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # 3) Preprocessing knobs
        preproc = self.config.get('preprocessing', {})
        adv = self.config.get('advanced', {})
        text_fields = preproc.get('text_fields', None)
        lowercase   = bool(preproc.get('lowercase', False) and preproc.get('clean_text', False))
        rm_special  = bool(preproc.get('remove_special_chars', False) and preproc.get('clean_text', False))
        raw_max_len = int(preproc.get('max_text_length', self.config['model']['max_length']))
        prefetch    = adv.get('prefetch_factor', None)

        # 4) Build child→parent mapping in CURRENT label space (for constraints/loss if enabled)
        hier = self.config.get('hierarchy', {})
        def parent_code(code: str) -> str:
            rule = hier.get('parent_rule', 'before_dot')
            if rule == 'first_two_digits':
                return ''.join([ch for ch in code if ch.isdigit()])[:2]
            return code.split('.')[0]

        child_to_parent = []
        for lab, c_idx in label_map.items():
            p = parent_code(lab)
            if p in label_map:
                p_idx = label_map[p]
                if p_idx != c_idx:
                    child_to_parent.append((c_idx, p_idx))

        config_obj.child_to_parent = child_to_parent
        config_obj.enforce_hierarchy = bool(hier.get('enforce_inference', False))
        config_obj.hierarchy_lambda  = float(hier.get('loss_penalty', 0.0))

        # 5) Dataloaders (note: label_column is BK or BK_TOP)
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
            label_column=label_column,
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
            label_column=label_column,
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
            label_column=label_column,
        )

        # 6) Model (with optional warm start + gradient checkpointing)
        device = torch.device("cuda" if config_obj.use_gpu else "cpu")
        model = get_model(config_obj.model_type, num_labels=config_obj.num_labels).to(device)
        if warm_start_ckpt:
            model = self._load_backbone_from_checkpoint(model, warm_start_ckpt)
            logger.info(f"Warmed backbone from: {warm_start_ckpt}")

        try:
            if self.config.get('advanced', {}).get('gradient_checkpointing', False):
                if hasattr(model, 'bart') and hasattr(model.bart, 'gradient_checkpointing_enable'):
                    model.bart.gradient_checkpointing_enable()
        except Exception:
            pass

        # 7) Train
        model, history, test_results = train_model(model, train_data, dev_data, test_data, device, config_obj)

        # 8) Plots (stage-tagged files under save_dir)
        stage_tag = 'stage1' if label_column == 'BK_TOP' else 'stage2'
        epochs = list(range(1, len(history) + 1))
        train_losses = [h[0] for h in history]
        val_losses   = [h[1] for h in history]

        # Dynamic metric plotting based on YAML evaluation.metrics
        eval_conf = self.config.get('evaluation', {})
        metrics_keys = eval_conf.get('metrics', None)
        if metrics_keys is None and history and isinstance(history[0][2], dict):
            metrics_keys = list(history[0][2].keys())
        def series(key):
            return [h[2].get(key) if isinstance(h[2], dict) else None for h in history]

        label_name = {
            'subset_accuracy':'Subset Accuracy','mcc':'MCC',
            'precision_micro':'Precision (micro)','recall_micro':'Recall (micro)','f1_micro':'F1 (micro)',
            'precision_macro':'Precision (macro)','recall_macro':'Recall (macro)','f1_macro':'F1 (macro)',
            'accuracy':'Accuracy',
        }

        # Loss plot
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses,   label="Val Loss")
        plt.title(f"Loss over Epochs ({stage_tag})")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        loss_plot_path = os.path.join(save_dir, f"loss_plot_{stage_tag}.png")
        plt.tight_layout(); plt.savefig(loss_plot_path); plt.close()

        # Metric plot
        plt.figure(figsize=(10, 6))
        if metrics_keys:
            for key in metrics_keys:
                vals = series(key)
                if any(v is not None for v in vals):
                    plt.plot(epochs, vals, label=label_name.get(key, key))
        plt.title(f"Validation Metrics over Epochs ({stage_tag})")
        plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend()
        val_metrics_plot_path = os.path.join(save_dir, f"val_metrics_plot_{stage_tag}.png")
        plt.tight_layout(); plt.savefig(val_metrics_plot_path); plt.close()

        # 9) Save test metrics
        test_metrics_path = os.path.join(save_dir, f"test_metrics_{stage_tag}.json")
        with open(test_metrics_path, "w") as f:
            json.dump(test_results, f, indent=2)

        results = {
            'loss_plot':         loss_plot_path,
            'val_metrics_plot':  val_metrics_plot_path,
            'test_metrics_path': test_metrics_path,
            'save_dir':          save_dir,
            'total_epochs':      len(history) if history else 0,
        }

        # Optional wandb logging
        try:
            if self.config['logging'].get('wandb', False) and WANDB_AVAILABLE:
                self._log_to_wandb(f"training_{stage_tag}", results)
        except Exception:
            pass


        # Hard cleanup to free VRAM/CPU RAM for the next stage
        try:
            del train_data, dev_data, test_data
            del model, history, test_results
        except Exception:
            pass
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info(f"[{stage_tag}] Training completed. Artifacts saved under: {save_dir}")
        return results

    def _run_hierarchical_training(self) -> Dict:
        """Run hierarchical joint training with improved model"""
        logger.info("Running improved hierarchical joint training...")
        
        # Build config object
        config_obj = self._create_config_object()
        config_obj.save_dir = self.ckpt_dir

        # Load label maps for both parent and child
        data_out = self.config['data']['output_dir']
        parent_label_map = load_label_map(os.path.join(data_out, 'label_map_top.json'))
        child_label_map = load_label_map(os.path.join(data_out, 'label_map.json'))
        
        config_obj.num_parent_labels = len(parent_label_map)
        config_obj.num_child_labels = len(child_label_map)

        # Load data splits
        train_df = pd.read_csv(os.path.join(data_out, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_out, 'val.csv'))
        test_df = pd.read_csv(os.path.join(data_out, 'test.csv'))

        logger.info(f"Hierarchical training - Parent labels: {len(parent_label_map)}, Child labels: {len(child_label_map)}")
        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Preprocessing settings
        preproc = self.config.get('preprocessing', {})
        text_fields = preproc.get('text_fields', None)
        lowercase = bool(preproc.get('lowercase', False) and preproc.get('clean_text', False))
        rm_special = bool(preproc.get('remove_special_chars', False) and preproc.get('clean_text', False))
        raw_max_len = int(preproc.get('max_text_length', self.config['model']['max_length']))

        # Create hierarchical dataloaders (both parent and child labels)
        train_data = self._transform_hierarchical_data(
            train_df, parent_label_map, child_label_map, config_obj,
            text_fields, lowercase, rm_special, raw_max_len, shuffle=True
        )
        dev_data = self._transform_hierarchical_data(
            val_df, parent_label_map, child_label_map, config_obj,
            text_fields, lowercase, rm_special, raw_max_len, shuffle=False
        )
        test_data = self._transform_hierarchical_data(
            test_df, parent_label_map, child_label_map, config_obj,
            text_fields, lowercase, rm_special, raw_max_len, shuffle=False
        )

        # Create improved hierarchical model
        device = torch.device("cuda" if config_obj.use_gpu else "cpu")
        
        hier_config = self.config.get('hierarchy', {})
        model = get_model(
            'hierarchical_bart',
            num_parent_labels=len(parent_label_map),
            num_child_labels=len(child_label_map),
            model_name=config_obj.model_name,
            parent_weight=hier_config.get('parent_weight', 0.2),
            fusion_type=hier_config.get('fusion_type', 'gated'),
            fusion_dim=hier_config.get('fusion_dim', 512),
            use_hierarchy_mask=hier_config.get('use_hierarchy_mask', True),
            hierarchy_penalty_weight=hier_config.get('hierarchy_penalty_weight', 0.1),
            scheduled_sampling=hier_config.get('scheduled_sampling', True),
            noise_robustness=hier_config.get('noise_robustness', True),
        ).to(device)

        # Copy hierarchical settings to config
        config_obj.teacher_forcing = hier_config.get('teacher_forcing', False)
        config_obj.sequential_training = hier_config.get('sequential_training', False)
        config_obj.parent_epochs = hier_config.get('parent_epochs', 3)
        config_obj.freeze_parent = hier_config.get('freeze_parent', False)
        
        # Pass label maps to config for hierarchy mask
        config_obj.parent_label_map = parent_label_map
        config_obj.child_label_map = child_label_map
        config_obj.hierarchy_config = hier_config
        
        # Set evaluation thresholds
        eval_config = self.config.get('evaluation', {})
        config_obj.evaluation_thresholds = eval_config.get('evaluation_thresholds', [0.1, 0.2, 0.3, 0.4, 0.5])

        # Train hierarchical model
        model, history, test_results = train_hierarchical_model(
            model, train_data, dev_data, test_data, device, config_obj
        )

        # Save the trained model (separate from JSON results)
        model_save_path = os.path.join(self.run_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_parent_labels': len(parent_label_map),
                'num_child_labels': len(child_label_map),
                'model_name': config_obj.model_name,
                'parent_weight': hier_config.get('parent_weight', 0.2),
                'fusion_type': hier_config.get('fusion_type', 'gated'),
                'fusion_dim': hier_config.get('fusion_dim', 512),
                'use_hierarchy_mask': hier_config.get('use_hierarchy_mask', True),
                'hierarchy_penalty_weight': hier_config.get('hierarchy_penalty_weight', 0.1),
                'scheduled_sampling': hier_config.get('scheduled_sampling', True),
                'noise_robustness': hier_config.get('noise_robustness', True)
            },
            'parent_label_map': parent_label_map,
            'child_label_map': child_label_map
        }, model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        # Save test predictions as CSV if enabled
        if self.config['evaluation'].get('save_predictions', False):
            self._save_hierarchical_test_predictions(
                model, test_data, device, parent_label_map, child_label_map, test_df
            )

        # Save results and plots
        self._save_hierarchical_plots(history, test_results)
        
        # Return JSON-serializable results (NO model object)
        return {
            'history': history,
            'test_results': test_results,
            'parent_labels': len(parent_label_map),
            'child_labels': len(child_label_map),
            'model_save_path': model_save_path
        }

    def _transform_hierarchical_data(self, df, parent_label_map, child_label_map, config_obj, 
                               text_fields, lowercase, rm_special, raw_max_len, shuffle=True):
        """Transform data for hierarchical training with both parent and child labels"""
        from utils import transform_hierarchical_data
        
        return transform_hierarchical_data(
            df, parent_label_map, child_label_map,
            max_length=config_obj.max_length,
            batch_size=config_obj.batch_size,
            model_name=config_obj.model_name,
            text_fields=text_fields,
            lowercase=lowercase,
            remove_special_chars=rm_special,
            raw_text_max_length=raw_max_len,
            shuffle=shuffle
        )

    def _save_hierarchical_plots(self, history, test_results):
        """Save training plots for hierarchical model"""
        if not history:
            return
            
        epochs = list(range(1, len(history) + 1))
        train_losses = [h[0] for h in history]
        val_losses = [h[1] for h in history]
        parent_losses = [h[3] for h in history] if len(history[0]) > 3 else []
        child_losses = [h[4] for h in history] if len(history[0]) > 4 else []
        
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, 'b-', label='Total Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Total Val Loss')
        if parent_losses:
            plt.plot(epochs, parent_losses, 'g--', label='Parent Train Loss')
        if child_losses:
            plt.plot(epochs, child_losses, 'm--', label='Child Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Hierarchical Training Loss')
        plt.legend()
        plt.grid(True)
        
        # Parent metrics
        if history and len(history[0]) > 2 and isinstance(history[0][2], dict):
            parent_f1 = [h[2].get('parent_f1_macro', 0) for h in history]
            child_f1 = [h[2].get('child_f1_macro', 0) for h in history]
            
            plt.subplot(1, 3, 2)
            plt.plot(epochs, parent_f1, 'g-', label='Parent F1-Macro')
            plt.xlabel('Epoch')
            plt.ylabel('F1-Macro')
            plt.title('Parent Performance')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.plot(epochs, child_f1, 'm-', label='Child F1-Macro')
            plt.xlabel('Epoch')
            plt.ylabel('F1-Macro') 
            plt.title('Child Performance')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.ckpt_dir, 'hierarchical_training_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved hierarchical training plots to: {plot_path}")

    def _save_hierarchical_test_predictions(self, model, test_data, device, parent_label_map, child_label_map, test_df):
        """Save hierarchical test predictions as CSV"""
        logger.info("Saving hierarchical test predictions...")
        
        model.eval()
        all_parent_pred, all_child_pred = [], []
        all_parent_labels, all_child_labels = [], []
        
        prediction_threshold = self.config['evaluation'].get('prediction_threshold', 0.5)
        
        with torch.no_grad():
            for batch in test_data:
                input_ids, attention_mask, parent_labels, child_labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                parent_labels = parent_labels.to(device)
                child_labels = child_labels.to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, mode="joint")
                
                # Parent predictions
                parent_probs = torch.sigmoid(outputs['parent_logits'])
                parent_predicted = (parent_probs > prediction_threshold).int().cpu()
                all_parent_pred.append(parent_predicted)
                all_parent_labels.append(parent_labels.cpu())
                
                # Child predictions
                child_probs = torch.sigmoid(outputs['child_logits'])
                child_predicted = (child_probs > prediction_threshold).int().cpu()
                all_child_pred.append(child_predicted)
                all_child_labels.append(child_labels.cpu())
        
        # Concatenate all predictions
        all_parent_pred = torch.cat(all_parent_pred, dim=0).numpy()
        all_parent_labels = torch.cat(all_parent_labels, dim=0).numpy()
        all_child_pred = torch.cat(all_child_pred, dim=0).numpy()
        all_child_labels = torch.cat(all_child_labels, dim=0).numpy()
        
        # Create inverse label maps
        inv_parent_labels = {idx: label for label, idx in parent_label_map.items()}
        inv_child_labels = {idx: label for label, idx in child_label_map.items()}
        
        # Save predictions as CSV
        pred_path = os.path.join(self.run_dir, "test_prediction.csv")
        with open(pred_path, "w", encoding="utf-8") as f:
            f.write("index,true_parent_labels,predicted_parent_labels,true_child_labels,predicted_child_labels\n")
            
            for i in range(len(all_parent_pred)):
                # Parent labels
                true_parent_idxs = np.where(all_parent_labels[i] == 1)[0].tolist()
                pred_parent_idxs = np.where(all_parent_pred[i] == 1)[0].tolist()
                true_parent_labels = "|".join(inv_parent_labels[idx] for idx in true_parent_idxs if idx in inv_parent_labels)
                pred_parent_labels = "|".join(inv_parent_labels[idx] for idx in pred_parent_idxs if idx in inv_parent_labels)
                
                # Child labels
                true_child_idxs = np.where(all_child_labels[i] == 1)[0].tolist()
                pred_child_idxs = np.where(all_child_pred[i] == 1)[0].tolist()
                true_child_labels = "|".join(inv_child_labels[idx] for idx in true_child_idxs if idx in inv_child_labels)
                pred_child_labels = "|".join(inv_child_labels[idx] for idx in pred_child_idxs if idx in inv_child_labels)
                
                f.write(f"{i},{true_parent_labels},{pred_parent_labels},{true_child_labels},{pred_child_labels}\n")
        
        logger.info(f"Test predictions saved to {pred_path}")

    def _create_config_object(self):
        """Convert YAML config to argparse-like object for existing modules"""
        class ConfigObject:
            pass

        config_obj = ConfigObject()

        # Paths
        data_out = self.config['data']['output_dir']
        config_obj.data_path = os.path.join(data_out, 'k10plus_processed_rare_label_removed.csv')
        config_obj.label_map_path = os.path.join(data_out, 'label_map.json')

        # Fine-tuning mode and stage to carry context
        ft = self.config.get('fine_tuning', {})
        config_obj.fine_tuning_mode = ft.get('mode', 'standard')
        stage_val = str(ft.get('stage', '2')).lower()   # accepts '1' | '2' | 'both'
        config_obj.fine_tuning_stage = stage_val
        

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
            elif hasattr(obj, '__class__') and 'torch' in str(obj.__class__):
                # Skip PyTorch objects (models, tensors, etc.)
                return f"<{obj.__class__.__name__}>"
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