#!/usr/bin/env python3
"""
Modular BK Classification Pipeline
Uses existing modules: data_preprocessing, random_baseline, train, model, utils, config
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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from data_preprocessing import BKDataProcessor, load_and_preprocess_data
from random_baseline import evaluate_random_baseline
from train import train_model, main as train_main
from model import get_model
from utils import seed_everything, load_label_map
from config import get_config
from random_baseline import evaluate_random_baseline_on_test
from utils import evaluate_model
import matplotlib.pyplot as plt

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

# Wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModularBKPipeline:
    """Modular BK Classification Pipeline using existing components"""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with YAML configuration"""
        self.config = self._load_config(config_path)
        self._setup_directories()
        self._set_seed()
        self._init_wandb()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Auto-generate experiment name if not provided
        if config['system']['experiment_name'] is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            threshold = config['data']['frequency_threshold']
            config['system']['experiment_name'] = f"bk_thresh{threshold}_{timestamp}"
            
        return config
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.config['data']['output_dir'],
            self.config['system']['save_dir'],
            os.path.join(self.config['system']['save_dir'], 'checkpoints'),
            os.path.join(self.config['system']['save_dir'], 'logs'),
            os.path.join(self.config['system']['save_dir'], 'baselines')
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _set_seed(self):
        """Set random seeds for reproducibility"""
        seed = self.config['split']['random_seed']
        seed_everything(seed)
        logger.info(f"Set random seed to {seed}")
    
    def _init_wandb(self):
        """Initialize Wandb if enabled"""
        if self.config['logging'].get('wandb', False) and WANDB_AVAILABLE:
            # Try to authenticate with API key from .env file
            api_key = os.getenv('WANDB_API_KEY')
            if api_key:
                try:
                    wandb.login(key=api_key)
                    logger.info("Successfully authenticated with Wandb using API key from .env")
                except Exception as e:
                    logger.warning(f"Failed to authenticate with Wandb API key: {e}")
            else:
                logger.info("No WANDB_API_KEY found in environment, using existing login")
            
            wandb.init(
                project=self.config['logging'].get('wandb_project', 'bk-classification'),
                name=self.config['system']['experiment_name'],
                config=self.config,
                tags=self._get_wandb_tags()
            )
            logger.info("Wandb initialized successfully")
        elif self.config['logging'].get('wandb', False) and not WANDB_AVAILABLE:
            logger.warning("Wandb requested but not available. Install with: pip install wandb")
    
    def _get_wandb_tags(self) -> List[str]:
        """Generate wandb tags based on execution mode"""
        tags = []
        execution = self.config.get('execution', {})
        
        if execution.get('run_preprocessing', False):
            tags.append('preprocessing')
        if execution.get('run_baseline', False):
            tags.append('baseline')
        if execution.get('run_training', False):
            tags.append('training')
            
        return tags if tags else ['pipeline']

    def _setup_directories(self):
        """Create necessary directories"""
        base_save = self.config['system']['save_dir']
        data_out = self.config['data']['output_dir']

        # Per-run descriptor and directories
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
        # sample size tag (use 'ALL' when None)
        sample = self.config['data'].get('sample_size', None)
        sample_tag = f"s{sample}" if sample is not None else "sALL"
        # timestamp tag
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{mtype}_{name}_bs{bs}_e{epochs}_{sample_tag}_{ts}"

    def run_pipeline(self):
        """Run the complete configurable pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING MODULAR BK CLASSIFICATION PIPELINE")
        logger.info("=" * 60)
        
        execution_config = self.config.get('execution', {})
        results = {}
        
        try:
            # Step 1: Data Preprocessing (if enabled)
            if execution_config.get('run_preprocessing', True):
                logger.info("Step 1: Running data preprocessing...")
                preprocessing_results = self._run_preprocessing()
                results['preprocessing'] = preprocessing_results
                self._log_to_wandb('preprocessing', preprocessing_results)
            
            # Step 2: Random Baseline Evaluation (if enabled in execution AND baseline config)
            if execution_config.get('run_baseline', True) and self.config['baseline'].get('run_random_baseline', False):
                logger.info("Step 2: Running uniform random baseline evaluation...")
                baseline_results = self._run_baseline()
                results['baseline'] = baseline_results
                self._log_to_wandb('baseline', baseline_results)
            else:
                logger.info("Step 2: Baseline evaluation skipped")
                results['baseline'] = {'status': 'skipped'}
            
            # Step 3: Model Training (if enabled)
            if execution_config.get('run_training', False):
                logger.info("Step 3: Running model training...")
                training_results = self._run_training()
                results['training'] = training_results
                self._log_to_wandb('training', training_results)
            
            # Save pipeline results
            self._save_pipeline_results(results)
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Results saved to: {self.config['system']['save_dir']}")
            logger.info("=" * 60)
            
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
        """Run data preprocessing using existing data_preprocessing module"""
        logger.info("Using data_preprocessing module...")
        
        # Get preprocessing config
        data_config = self.config['data']
        preprocessing_config = self.config['preprocessing']
        
        # Check sample_size
        sample_size = data_config.get('sample_size', None)
        if sample_size is None:
            logger.info("sample_size is null - loading entire dataset")
        else:
            logger.info(f"sample_size is {sample_size} - will sample from dataset")
        
        # Use existing data preprocessing function
        df_processed, label_map, processor = load_and_preprocess_data(
            frequency_threshold=data_config['frequency_threshold'],
            data_source_path=data_config['raw_data_path'],
            output_data_path=os.path.join(data_config['output_dir'], 'k10plus_processed_rare_label_removed.csv'),
            label_map_path=os.path.join(data_config['output_dir'], 'label_map.json')
        )
        
        # Apply sampling if specified
        if sample_size is not None and sample_size < len(df_processed):
            logger.info(f"Sampling {sample_size} records from {len(df_processed)} total records")
            df_processed = df_processed.sample(n=sample_size, random_state=self.config['split']['random_seed'])
        else:
            logger.info(f"Using entire dataset: {len(df_processed)} records")
        
        # Split data using sklearn's train_test_split for consistency
        from sklearn.model_selection import train_test_split
        
        train_ratio = self.config['split']['train_ratio']
        val_ratio = self.config['split']['val_ratio']
        test_ratio = self.config['split']['test_ratio']
        random_seed = self.config['split']['random_seed']
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df_processed,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            shuffle=True
        )
        
        # Second split: val vs test
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_test_ratio),
            random_state=random_seed,
            shuffle=True
        )
        
        # Save splits
        splits = {'train': train_df, 'val': val_df, 'test': test_df}
        
        for split_name, split_df in splits.items():
            output_path = os.path.join(data_config['output_dir'], f"{split_name}.csv")
            split_df.to_csv(output_path, index=False)
            logger.info(f"Saved {split_name} split: {len(split_df)} samples to {output_path}")
        
        # Update config with actual number of labels
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
        
        logger.info(f"Preprocessing completed: {results}")
        return results
    
    def _run_baseline(self) -> Dict:
        """Run baseline evaluation using test set only with batch-by-batch progress"""
        logger.info("Running random baseline on test set with progress...")
        
        # Check if baseline should run
        if not self.config['baseline'].get('run_random_baseline', False):
            logger.info("Baseline evaluation skipped (run_random_baseline=false)")
            return {'status': 'skipped'}
        
        try:
            # Import the new function with progress tracking
            
            # Create config object
            config_obj = self._create_config_object()
            
            # Path to test set
            test_csv_path = os.path.join(self.config['data']['output_dir'], 'test.csv')
            
            # Check if test set exists
            if not os.path.exists(test_csv_path):
                raise FileNotFoundError(f"Test set not found: {test_csv_path}")
            
            # Run baseline with wandb logging and progress (only final results logged to wandb)
            wandb_enabled = self.config['logging'].get('wandb', False) and WANDB_AVAILABLE
            baseline_results = evaluate_random_baseline_on_test(
                config_obj, 
                test_csv_path, 
                wandb_enabled=wandb_enabled
            )
            
            # Save results locally
            baseline_dir = os.path.join(self.config['system']['save_dir'], 'baselines')
            os.makedirs(baseline_dir, exist_ok=True)
            results_path = os.path.join(baseline_dir, 'random_baseline_test_results.json')
            
            with open(results_path, 'w') as f:
                json.dump(baseline_results, f, indent=2)
            
            logger.info(f"Baseline results saved to {results_path}")
            return baseline_results
            
        except Exception as e:
            logger.error(f"Baseline evaluation failed: {str(e)}")
            # Fallback to original method
            logger.info("Falling back to original baseline method...")
            config_obj = self._create_config_object()
            from random_baseline import evaluate_random_baseline
            baseline_results = evaluate_random_baseline(config_obj)
            
            # Save fallback results
            baseline_dir = os.path.join(self.config['system']['save_dir'], 'baselines')
            os.makedirs(baseline_dir, exist_ok=True)
            results_path = os.path.join(baseline_dir, 'random_baseline_fallback_results.json')
            
            with open(results_path, 'w') as f:
                json.dump(baseline_results, f, indent=2)
            
            logger.info(f"Fallback baseline results saved to {results_path}")
            return baseline_results
    
    def _run_training(self) -> Dict:
        """Run model training using existing train module"""
        logger.info("Using train module...")
        
        # Create argparse config object from YAML config
        config_obj = self._create_config_object()
        
        # Import training components
        from utils import load_label_map, transform_data
        import torch
        
        # Load data and labels
        label_map = load_label_map(config_obj.label_map_path)
        config_obj.num_labels = len(label_map)
        
        # Load preprocessed splits instead of using load_and_split_data
        train_df = pd.read_csv(os.path.join(self.config['data']['output_dir'], 'train.csv'))
        val_df = pd.read_csv(os.path.join(self.config['data']['output_dir'], 'val.csv'))
        
        logger.info(f"Loaded train split: {len(train_df)} samples")
        logger.info(f"Loaded val split: {len(val_df)} samples")
        
        # Transform data
        train_data = transform_data(train_df, label_map,
                                   max_length=config_obj.max_length,
                                   batch_size=config_obj.batch_size,
                                   model_name=config_obj.model_name,
                                   num_workers=config_obj.num_workers,
                                   pin_memory=config_obj.pin_memory)
        dev_data = transform_data(val_df, label_map,
                                 max_length=config_obj.max_length,
                                 batch_size=config_obj.batch_size,
                                 model_name=config_obj.model_name,
                                 num_workers=config_obj.num_workers,
                                 pin_memory=config_obj.pin_memory)
        
        # Initialize model
        model = get_model(config_obj.model_type, num_labels=config_obj.num_labels)
        device = torch.device("cuda" if config_obj.use_gpu else "cpu")
        model = model.to(device)
        
        logger.info(f"Training {config_obj.model_type} with {config_obj.num_labels} labels")
        logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
        
        # Train model
        model, history = train_model(model, train_data, dev_data, device, config_obj)
        
        epochs = list(range(1, len(history) + 1))

        # history entry: (avg_train_loss, train_metrics, val_metrics)
        # train/val metrics order: [0]=acc, [1]=mcc, [2]=precision, [3]=recall, [4]=f1, [5]=subset_acc
        train_acc     = [h[1][0] for h in history]
        val_acc       = [h[2][0] for h in history]
        train_mcc     = [h[1][1] for h in history]
        val_mcc       = [h[2][1] for h in history]
        train_prec    = [h[1][2] for h in history]
        val_prec      = [h[2][2] for h in history]
        train_recall  = [h[1][3] for h in history]
        val_recall    = [h[2][3] for h in history]
        train_f1      = [h[1][4] for h in history]
        val_f1        = [h[2][4] for h in history]
        train_subset  = [h[1][5] for h in history]
        val_subset    = [h[2][5] for h in history]
        
        # Plot metrics across epochs
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        plots = [
            ("Accuracy",          train_acc,    val_acc),
            ("Matthews (MCC)",    train_mcc,    val_mcc),
            ("Precision",         train_prec,   val_prec),
            ("Recall",            train_recall, val_recall),
            ("F1",                train_f1,     val_f1),
            ("Subset Accuracy",   train_subset, val_subset),
        ]

        for ax, (title, tr, va) in zip(axes.ravel(), plots):
            ax.plot(epochs, tr, label="Train")
            ax.plot(epochs, va, label="Val")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "metrics_plot.png")
        plt.savefig(plot_path)
        plt.close()

        # Evaluate on test set
        test_df = pd.read_csv(os.path.join(self.config['data']['output_dir'], 'test.csv'))
        test_data = transform_data(test_df, label_map,
                                max_length=config_obj.max_length,
                                batch_size=config_obj.batch_size,
                                model_name=config_obj.model_name,
                                num_workers=config_obj.num_workers,
                                pin_memory=config_obj.pin_memory)
        test_metrics = evaluate_model(model, test_data, device, desc="Evaluation on Test Set")

        # Assemble results (includes test metrics)
        training_results = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_mcc': train_mcc,
            'val_mcc': val_mcc,
            'train_prec': train_prec,
            'val_prec': val_prec,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'train_subset': train_subset,
            'val_subset': val_subset,
            'test_accuracy':        test_metrics[0],
            'test_mcc':             test_metrics[1],
            'test_precision':       test_metrics[2],
            'test_recall':          test_metrics[3],
            'test_f1':              test_metrics[4],
            'test_subset_accuracy': test_metrics[5],
            'total_epochs':         len(history) if history else 0,
            'run_dir':              self.run_dir,
            'metrics_plot':         plot_path
        }
        
        logger.info("Training completed!")
        return training_results
    
    def _create_config_object(self):
        """Convert YAML config to argparse-like object for existing modules"""
        class ConfigObject:
            pass
        
        config_obj = ConfigObject()
        
        # Map YAML config to argparse config
        config_obj.data_path = os.path.join(self.config['data']['output_dir'], 'k10plus_processed_rare_label_removed.csv')
        config_obj.label_map_path = os.path.join(self.config['data']['output_dir'], 'label_map.json')
        
        # No need for complex sample_size logic since we use pre-split test data for baseline
        # The baseline will use test.csv directly, so sample_size is not relevant
        config_obj.sample_size = None  # Not used in new baseline function
        config_obj.train_ratio = self.config['split']['train_ratio']
        
        # Model parameters
        config_obj.model_type = self.config['model']['model_type']
        config_obj.num_labels = self.config['model']['num_labels']
        config_obj.model_name = self.config['model']['name']
        config_obj.max_length = self.config['model']['max_length']
        
        # Training parameters
        config_obj.batch_size = int(self.config['training']['batch_size'])
        config_obj.epochs = int(self.config['training']['epochs'])
        config_obj.lr = float(self.config['training']['learning_rate'])
        config_obj.use_mixed_precision = bool(self.config['training']['use_mixed_precision'])
        
        # System parameters
        config_obj.seed = self.config['split']['random_seed']
        config_obj.use_gpu = self.config['system']['use_gpu']
        config_obj.save_dir = self.ckpt_dir
        config_obj.num_workers = int(self.config['advanced'].get('dataloader_num_workers', 0))
        config_obj.pin_memory = bool(self.config['advanced'].get('pin_memory', False))
        return config_obj
    
    def _log_to_wandb(self, step_name: str, results: Dict):
        """Log results to Wandb"""
        if not (self.config['logging'].get('wandb', False) and WANDB_AVAILABLE):
            return
        
        # Log with step prefix
        wandb_logs = {f'{step_name}/{key}': value for key, value in results.items()}
        wandb.log(wandb_logs)
    
    def _save_pipeline_results(self, results: Dict):
        """Save complete pipeline results"""
        results_path = os.path.join(self.config['system']['save_dir'], 'pipeline_results.json')
        
        # Convert any numpy types to native Python types
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
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print configuration and exit")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    # Initialize pipeline
    pipeline = ModularBKPipeline(args.config)
    
    if args.dry_run:
        print("Configuration:")
        print(yaml.dump(pipeline.config, indent=2))
        return
    
    # Run pipeline
    try:
        results = pipeline.run_pipeline()
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()