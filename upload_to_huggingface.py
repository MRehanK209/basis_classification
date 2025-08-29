#!/usr/bin/env python3
"""
Upload the best performing two-stage BART model to Hugging Face Hub

This script uploads the best model from two-stage fine-tuning to Hugging Face Hub
for easy access in inference and sharing with the community.

Model Performance:
- Subset Accuracy: 25.7%
- MCC: 0.498
- F1-Micro: 47.9%
- F1-Macro: 21.4%
"""

import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModel,
    BartModel,
    BartConfig
)
from huggingface_hub import HfApi, create_repo, upload_folder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BartWithClassifier(nn.Module):
    """BART classifier for multi-label BK classification"""
    
    def __init__(self, num_labels=1884, model_name="facebook/bart-large", dropout=0.1):
        super(BartWithClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.bart = BartModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]  # Take [CLS] token representation
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

def upload_model_to_hf(
    model_path: str,
    label_map_path: str,
    hf_model_name: str,
    hf_token: str = None,
    private: bool = False
):
    """
    Upload the trained model to Hugging Face Hub
    
    Args:
        model_path: Path to the trained model checkpoint
        label_map_path: Path to label mapping JSON
        hf_model_name: Name for the model on HF Hub (e.g., "username/bk-classification-bart")
        hf_token: Hugging Face token (optional if already logged in)
        private: Whether to make the repo private
    """
    
    # Create temporary directory for Hugging Face format
    temp_dir = Path("./temp_hf_model")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Load label mapping
        logger.info("Loading label mapping...")
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        
        num_labels = len(label_map)
        logger.info(f"Loaded {num_labels} BK labels")
        
        # Load trained model
        logger.info("Loading trained model...")
        model = BartWithClassifier(num_labels=num_labels, model_name="facebook/bart-large")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            epoch = checkpoint.get('epoch', 'unknown')
            best_metric = checkpoint.get('best_metric', 'unknown')
            logger.info(f"Loaded model from epoch {epoch}, best metric: {best_metric}")
        else:
            raise ValueError("Could not find 'model_state' in checkpoint")
        
        model.eval()
        
        # Save BART backbone in HF format
        logger.info("Saving BART backbone...")
        model.bart.save_pretrained(temp_dir / "bart_backbone")
        
        # Save tokenizer
        logger.info("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        tokenizer.save_pretrained(temp_dir)
        
        # Save classifier head weights
        logger.info("Saving classifier head...")
        classifier_state = {
            'weight': model.classifier.weight.data,
            'bias': model.classifier.bias.data
        }
        torch.save(classifier_state, temp_dir / "classifier_head.pt")
        
        # Create model configuration
        logger.info("Creating model configuration...")
        config = {
            "model_type": "bart_with_classifier",
            "base_model": "facebook/bart-large",
            "num_labels": num_labels,
            "dropout": 0.1,
            "task": "multi_label_classification",
            "domain": "bibliographic_classification",
            "language": "de",  # German
            "classification_system": "BK (Basisklassifikation)",
            "performance": {
                "subset_accuracy": 0.257,
                "mcc": 0.498,
                "f1_micro": 0.479,
                "f1_macro": 0.214,
                "precision_micro": 0.661,
                "recall_micro": 0.376,
                "precision_macro": 0.338,
                "recall_macro": 0.175
            },
            "training_details": {
                "approach": "two_stage_fine_tuning",
                "stage1_epochs": 15,
                "stage2_epochs": 15,
                "batch_size": 64,
                "learning_rate": 2e-5,
                "dataset_size": 250831,
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15
            }
        }
        
        with open(temp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save label mapping
        logger.info("Saving label mapping...")
        with open(temp_dir / "label_map.json", 'w') as f:
            json.dump(label_map, f, indent=2)
        
        # Create reverse mapping
        idx_to_label = {v: k for k, v in label_map.items()}
        with open(temp_dir / "idx_to_label.json", 'w') as f:
            json.dump(idx_to_label, f, indent=2)
        
        # Create model card
        logger.info("Creating model card...")
        model_card = f"""
---
language: de
license: mit
library_name: transformers
pipeline_tag: text-classification
tags:
- bibliographic-classification
- bk-codes
- german-libraries
- multi-label-classification
- bart
datasets:
- k10plus-catalog
metrics:
- accuracy
- f1
- precision
- recall
- matthews_correlation
---

# BK Classification - Two-Stage BART

This model performs automatic classification of German bibliographic records using BK (Basisklassifikation) codes.

## Model Description

This is a **two-stage fine-tuned BART-large model** for multi-label classification of bibliographic metadata into BK classification codes. The model achieved **state-of-the-art performance** on the K10plus library catalog dataset.

### Performance

- **Subset Accuracy**: 25.7%
- **Matthews Correlation Coefficient (MCC)**: 0.498
- **F1-Score (Micro)**: 47.9%
- **F1-Score (Macro)**: 21.4%
- **Precision (Micro)**: 66.1%
- **Recall (Micro)**: 37.6%

### Training Approach

The model uses a **two-stage fine-tuning approach**:

1. **Stage 1**: Train on parent BK categories (48 labels)
2. **Stage 2**: Fine-tune on all BK codes (1,884 labels) using Stage 1 as initialization

This approach outperformed both standard fine-tuning and hierarchical joint training.

### Dataset

- **Source**: K10plus German library catalog (2010-2020)
- **Total Records**: 250,831 bibliographic entries
- **Labels**: 1,884 unique BK classification codes
- **Input Fields**: Title, Summary, Keywords, LOC Keywords, RVK codes

### Usage

```python
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{hf_model_name}")

# Load model components (see repository for full inference code)
# This model requires custom loading due to the classifier head
```

### Citation

If you use this model, please cite:

```bibtex
@misc{{bk-classification-bart,
  title={{Automatic BK Classification using Two-Stage BART Fine-tuning}},
  author={{Khalid, M. Rehan}},
  year={{2025}},
  howpublished={{Hugging Face Model Hub}},
  url={{https://huggingface.co/{hf_model_name}}}
}}
```

### Contact

- **Author**: M. Rehan Khalid
- **Email**: m.khalid@stud.uni-goettingen.de
- **Affiliation**: University of Göttingen

### License

MIT License
"""
        
        with open(temp_dir / "README.md", 'w') as f:
            f.write(model_card)
        
        # Create requirements file
        requirements = """
torch>=2.0.0
transformers>=4.30.0
huggingface_hub>=0.16.0
numpy>=1.24.0
"""
        with open(temp_dir / "requirements.txt", 'w') as f:
            f.write(requirements)
        
        # Create inference example
        inference_code = f'''
"""
Example inference code for BK Classification BART model
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import json

class BartWithClassifier(nn.Module):
    """BART classifier for multi-label BK classification"""
    
    def __init__(self, num_labels=1884, model_name="facebook/bart-large", dropout=0.1):
        super(BartWithClassifier, self).__init__()
        
        self.num_labels = num_labels
        # Load from local bart_backbone directory
        from transformers import BartModel
        self.bart = BartModel.from_pretrained("./bart_backbone")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]  # Take [CLS] token representation
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

def load_model_from_hf(model_name="{hf_model_name}"):
    """Load the complete model from Hugging Face Hub"""
    
    # Download files
    classifier_path = hf_hub_download(repo_id=model_name, filename="classifier_head.pt")
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    label_map_path = hf_hub_download(repo_id=model_name, filename="label_map.json")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load label mapping
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    # Initialize model
    model = BartWithClassifier(num_labels=config["num_labels"])
    
    # Load classifier head
    classifier_state = torch.load(classifier_path, map_location='cpu')
    model.classifier.load_state_dict(classifier_state)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer, label_map

# Example usage
if __name__ == "__main__":
    model, tokenizer, label_map = load_model_from_hf()
    
    # Example text
    text = """
    Title: Künstliche Intelligenz in der Bibliothek
    Summary: Ein Überblick über moderne KI-Methoden für Bibliothekswesen
    Keywords: künstliche intelligenz, bibliothek, automatisierung
    LOC_Keywords: artificial intelligence, library science
    RVK: AN 73000
    """
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=768)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.sigmoid(logits)
        
        # Get top predictions
        top_k = 5
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        print("Top predictions:")
        idx_to_label = {{v: k for k, v in label_map.items()}}
        for prob, idx in zip(top_probs, top_indices):
            label = idx_to_label[idx.item()]
            print(f"  {{label}}: {{prob:.3f}}")
'''
        
        with open(temp_dir / "inference_example.py", 'w') as f:
            f.write(inference_code)
        
        # Create repository and upload
        logger.info(f"Creating repository: {hf_model_name}")
        
        api = HfApi(token=hf_token)
        
        # Try to create repository
        try:
            repo_url = create_repo(
                repo_id=hf_model_name,
                repo_type="model",
                token=hf_token,
                exist_ok=True  # Don't fail if repo already exists
            )
            logger.info(f"Repository created/verified: {repo_url}")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            # Try to continue anyway in case repo exists
        
        # Upload files one by one for better error handling
        logger.info("Uploading files to Hugging Face Hub...")
        
        try:
            # Upload each file individually
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    logger.info(f"Uploading {file_path.name}...")
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=hf_model_name,
                        repo_type="model",
                        token=hf_token
                    )
            
            # Upload bart_backbone folder if it exists
            bart_backbone_dir = temp_dir / "bart_backbone"
            if bart_backbone_dir.exists():
                logger.info("Uploading BART backbone...")
                for file_path in bart_backbone_dir.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(temp_dir)
                        logger.info(f"Uploading {relative_path}...")
                        api.upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=str(relative_path),
                            repo_id=hf_model_name,
                            repo_type="model",
                            token=hf_token
                        )
        except Exception as upload_error:
            logger.error(f"Upload failed: {upload_error}")
            logger.info("Trying alternative upload method...")
            
            # Fallback: try upload_folder
            try:
                api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=hf_model_name,
                    repo_type="model",
                    token=hf_token,
                    allow_patterns=["*.json", "*.pt", "*.txt", "*.md", "*.py"],
                    ignore_patterns=[".git/*"]
                )
            except Exception as folder_error:
                logger.error(f"Folder upload also failed: {folder_error}")
                raise
        
        logger.info(f"Successfully uploaded model to: https://huggingface.co/{hf_model_name}")
        
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "results/bart_classifier_bart-large_bs64_e15_sALL_2stage_bart_20250811_000132/checkpoints_stage2/best_model_15.pt"
    LABEL_MAP_PATH = "data/label_map.json"
    HF_MODEL_NAME = "mrehank209/bk-classification-bart-two-stage"  # Fixed to correct username
    
    # Make sure you're logged in to Hugging Face
    # Run: huggingface-cli login
    
    print("Starting upload to Hugging Face Hub...")
    print(f"Model will be uploaded to: {HF_MODEL_NAME}")
    print("Make sure you're logged in: huggingface-cli login")
    
    # Uncomment the line below after logging in
    upload_model_to_hf(MODEL_PATH, LABEL_MAP_PATH, HF_MODEL_NAME, private=False)
    
    print("""
To upload the model:
1. Install huggingface_hub: pip install huggingface_hub
2. Login: huggingface-cli login
3. Uncomment the upload line above and run this script
4. Change HF_MODEL_NAME to your username/model-name
""")
