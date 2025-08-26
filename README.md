# BK Classification Pipeline

A comprehensive system for automatic classification of bibliographic records using hierarchical BK (Basisklassifikation) codes. This project implements multiple approaches from data extraction from the K10plus library catalog to fine-tuning state-of-the-art transformer models with different training strategies.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Extraction](#data-extraction)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures & Training Approaches](#model-architectures--training-approaches)
- [Training Pipeline](#training-pipeline)
- [Evaluation and Results](#evaluation-and-results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results Comparison](#results-comparison)
- [Configuration](#configuration)
- [Directory Structure](#directory-structure)

## Project Overview

This project aims to automatically classify bibliographic records using BK (Basisklassifikation) codes, a German library classification system. The system leverages advanced NLP techniques and implements **three different training approaches** to predict multiple classification codes from bibliographic metadata.

### Key Features

- **Asynchronous data extraction** from K10plus SRU API
- **Multiple training strategies**: Direct, Two-stage, and Hierarchical approaches
- **Comprehensive evaluation** with multiple metrics and thresholds
- **Modular pipeline** supporting different execution modes
- **Robust preprocessing** with rare label filtering
- **Advanced training** with anti-overfitting techniques

## Data Extraction

### K10plus Fetcher

The data extraction module (`k10plus_fetcher/`) implements an asynchronous scraper for the K10plus library catalog:

**Features:**
- Fetches bibliographic records from 2010-2020
- Extracts structured metadata: Title, Author, Summary, Keywords, RVK, BK codes
- Handles rate limiting and error recovery
- Processes ~251,172 records with BK classifications

**Key Fields Extracted:**
- `PPN`: Unique record identifier
- `Title`: Book title
- `Author`: Author information
- `Year`: Publication year
- `Summary`: Book summary/abstract
- `Keywords`: Subject keywords
- `LOC_Keywords`: Library of Congress keywords
- `RVK`: RVK classification codes
- `BK`: Target BK classification codes (multi-label)

**Usage:**
```bash
cd k10plus_fetcher
python main.py
```

## Data Preprocessing

### Data Cleaning and Filtering

The preprocessing pipeline (`data_preprocessing.py`) performs:

1. **Data Consolidation**: Merges yearly CSV files into single dataset
2. **Quality Filtering**: Removes records without BK codes
3. **Rare Label Removal**: Filters labels with frequency ≤ 10 occurrences
4. **Label Mapping**: Creates consistent label-to-index mappings
5. **Hierarchical Label Creation**: Generates parent labels (BK_TOP) from full BK codes
6. **Train/Val/Test Split**: 70/15/15 stratified split

**Parent Label Generation:**
```python
# BK_TOP creation from BK codes using "before_dot" rule
# Example: "86.47|85.15" → "86|85"
def parent_code(code: str) -> str:
    return code.split('.')[0]  # Takes part before dot
```

**Statistics:**
- **Raw Records**: 251,172 bibliographic entries
- **After Filtering**: ~250K records with valid BK codes
- **Final Child Labels**: 1,884 unique BK classification codes
- **Parent Labels**: ~50 unique top-level categories
- **Label Frequency Threshold**: 10 (configurable)

### Text Processing

The system combines multiple text fields for classification:

```python
# Combined input format
input_text = f"""
Title: {title}
Summary: {summary}  
Keywords: {keywords}
LOC_Keywords: {loc_keywords}
RVK: {rvk_codes}
"""
```

## Model Architectures & Training Approaches

The project implements **three distinct approaches** for BK classification:

### 1. Direct Fine-tuning (Standard BART)

**Configuration:**
```yaml
model:
  model_type: "bart_classifier"
fine_tuning:
  mode: "standard"
```

**Architecture:**
```python
class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=1884):
        self.bart = BartModel.from_pretrained("facebook/bart-large")
        self.classifier = nn.Linear(hidden_size, num_labels)
```

**Approach:**
- Single-stage training directly on all 1,884 BK labels
- Standard multi-label classification with BCEWithLogitsLoss
- No hierarchical structure consideration

### 2. Two-Stage Fine-tuning 

**Configuration:**
```yaml
model:
  model_type: "bart_classifier"  
fine_tuning:
  mode: "multi_stage"
  stage: "both"  # Runs both stages automatically
```

**Approach:**
- **Stage 1**: Train BART classifier on parent labels (BK_TOP) only
  - ~300 parent categories (before-dot codes)
  - Example: "86", "85", "83" etc.
  - Results stored in `checkpoints_stage1/`
  
- **Stage 2**: Fine-tune on full labels (BK) using Stage 1 checkpoint as warm start
  - All 1,884 child labels
  - Leverages learned representations from parent classification
  - Results stored in `checkpoints_stage2/`

**Key Benefits:**
- Progressive learning from coarse to fine-grained categories
- Better initialization for final classification task
- Reduced overfitting on complex label space

### 3. Hierarchical Joint Training

**Configuration:**
```yaml
model:
  model_type: "hierarchical_bart"
fine_tuning:
  mode: "hierarchical_joint"
hierarchy:
  sequential_training: true
  parent_epochs: 5
  parent_weight: 0.15
  fusion_type: "gated"
```

**Architecture:**
```python
class ImprovedHierarchicalBartClassifier(nn.Module):
    def __init__(self, num_parent_labels, num_child_labels):
        # Shared BART backbone
        self.bart = BartModel.from_pretrained("facebook/bart-large")
        
        # Dual classification heads
        self.parent_classifier = nn.Sequential(...)  # Parent prediction
        self.child_classifier = nn.Sequential(...)   # Child prediction
        
        # Hierarchical fusion mechanism
        self.parent_gate = nn.Linear(...)  # Gated fusion
```

**Training Process:**
1. **Phase 1**: Parent-only training (5 epochs)
   - Train only parent classifier
   - Focus on learning coarse-grained categories
   
2. **Phase 2**: Joint training (remaining epochs)
   - Train both parent and child classifiers simultaneously
   - Hierarchical loss: `total_loss = 0.85 * child_loss + 0.15 * parent_loss`
   - Gated fusion mechanism combines parent predictions with BART features

**Advanced Features:**
- **Hierarchical constraints**: Child predictions must be consistent with parent predictions
- **Scheduled sampling**: Gradually transitions from ground truth to predicted parent labels
- **Noise robustness**: Adds small noise during training for better generalization
- **Dynamic loss weighting**: Parent weight decreases over training to focus on child task

## Training Pipeline

### Configuration-Driven Execution

The pipeline automatically selects the training approach based on configuration:

```python
# Pipeline logic in pipeline.py
def _run_training(self):
    ft = self.config.get('fine_tuning', {})
    mode = ft.get('mode', 'standard')
    
    if mode == 'hierarchical_joint':
        return self._run_hierarchical_training()
    elif mode == 'multi_stage':
        return self._run_multi_stage_training()  # Two-stage approach
    else:
        return self._run_single_stage(stage='2')  # Direct approach
```

### Training Configurations by Approach

| Approach | Batch Size | Epochs | Learning Rate | Model | Key Features |
|----------|------------|--------|---------------|-------|--------------|
| **Direct** | 64 | 15 | 2e-5 | BART Classifier | Single-stage, standard training |
| **Two-Stage** | 64 | 15+15 | 2e-5 | BART Classifier | Parent→Child progressive training |
| **Hierarchical** | 16 | 20 | 1e-5 | Hierarchical BART | Dual-head, joint optimization |

### Anti-Overfitting Strategies

All approaches implement robust anti-overfitting techniques:
- **Weight decay**: 0.01-0.05 for regularization
- **Mixed precision**: Faster training with reduced memory
- **Early stopping**: Monitors MCC/F1-macro with patience
- **Gradient accumulation**: Effective larger batch sizes
- **Learning rate scheduling**: Cosine annealing with warmup

## Evaluation and Results

### Metrics

The system evaluates using multiple metrics across different prediction thresholds:

**Primary Metrics:**
- **MCC (Matthews Correlation Coefficient)**: Primary optimization target
- **F1-Score**: Micro and macro-averaged
- **Precision/Recall**: Micro and macro-averaged
- **Subset Accuracy**: Exact match accuracy

**Evaluation Thresholds**: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

## Results Comparison

### Two-Stage BART Results (Best Performing)

**Stage 1 (Parent Classification):**
- **Subset Accuracy**: 56.4% 
- **MCC**: 0.734
- **F1-Micro**: 0.741
- **F1-Macro**: 0.663
- **Training**: 15 epochs on ~300 parent labels

**Stage 2 (Full Classification):**
- **Subset Accuracy**: 25.7%
- **MCC**: 0.498
- **F1-Micro**: 0.479
- **F1-Macro**: 0.214
- **Training**: 15 epochs on 1,884 full labels (warm start from Stage 1)

### Hierarchical BART Results

**Best Performance (Threshold = 0.25):**

**Parent Level:**
- **Subset Accuracy**: 50.7%
- **MCC**: 0.724
- **F1-Micro**: 0.700
- **F1-Macro**: 0.640

**Child Level:**
- **Subset Accuracy**: 12.2%
- **MCC**: 0.409
- **F1-Micro**: 0.401
- **F1-Macro**: 0.139

### Performance Analysis & Key Findings

| Approach | Parent Accuracy | Child Accuracy | MCC (Child) | Training Time | Best For |
|----------|----------------|----------------|-------------|---------------|----------|
| **Two-Stage BART** | **56.4%** | **25.7%** | **0.498** | ~30 hours | Best overall performance |
| **Hierarchical BART** | 50.7% | 12.2% | 0.409 | ~24 hours | Advanced modeling techniques |
| **Direct BART** | N/A | ~11% | ~0.38 | ~15 hours | Baseline comparison |

**Key Insights:**

1. **Two-stage training significantly outperforms other approaches**
   - 25.7% vs 12.2% subset accuracy on child classification
   - 0.498 vs 0.409 MCC score
   - Better transfer learning from parent to child tasks

2. **Parent classification is much easier than child classification**
   - 50-56% accuracy for parent vs 12-26% for child
   - Coarse-grained categories are more predictable

3. **Progressive learning works better than joint optimization**
   - Two-stage approach outperforms simultaneous hierarchical training
   - Sequential learning allows better specialization

4. **Threshold optimization is crucial**
   - Performance varies significantly across thresholds
   - Optimal threshold around 0.25-0.5 depending on metric

## Setup and Installation

### Requirements

```bash
# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `scikit-learn>=1.3.0`
- `pandas>=2.0.0`
- `aiohttp>=3.8.0` (for data fetching)
- `matplotlib>=3.7.0` (for plotting)
- `seaborn>=0.12.0` (for analysis)

### Hardware Requirements

- **GPU**: NVIDIA GPU with ≥24GB VRAM (recommended for large batch sizes)
- **RAM**: ≥32GB system memory
- **Storage**: ≥50GB free space

## Usage

### Quick Start - Different Training Approaches

1. **Two-Stage Training (Recommended):**
```yaml
# config/config.yaml
model:
  model_type: "bart_classifier"
fine_tuning:
  mode: "multi_stage"
  stage: "both"
training:
  batch_size: 64
  epochs: 15
```

2. **Hierarchical Training:**
```yaml
# config/config.yaml  
model:
  model_type: "hierarchical_bart"
fine_tuning:
  mode: "hierarchical_joint"
training:
  batch_size: 16
  epochs: 20
```

3. **Direct Training:**
```yaml
# config/config.yaml
model:
  model_type: "bart_classifier" 
fine_tuning:
  mode: "standard"
training:
  batch_size: 64
  epochs: 15
```

### Run Pipeline

```bash
# Run complete pipeline
python pipeline.py --config config/config.yaml

# Run specific components
python pipeline.py --config config/config.yaml --preprocessing-only
python pipeline.py --config config/config.yaml --training-only
```

## Generated Analysis and Plots

The `testing.ipynb` notebook provides comprehensive data analysis including:

### Data Exploration
1. **Label Distribution Analysis**: Power-law distribution of BK codes
2. **Data Quality Heatmaps**: Missing value patterns across fields
3. **Temporal Analysis**: Publication year distribution
4. **Text Length Statistics**: Input text characteristics

### Performance Analysis  
1. **Training Curves**: Loss evolution and metric progression
2. **Threshold Analysis**: Performance across different prediction thresholds
3. **Per-Class Performance**: Precision/recall for individual BK codes
4. **Hierarchical Consistency**: Parent-child prediction alignment
5. **Error Analysis**: Common misclassification patterns

### Visualizations Generated
- `loss_plot_stage1.png` / `loss_plot_stage2.png`: Training loss curves
- `val_metrics_plot_stage1.png` / `val_metrics_plot_stage2.png`: Validation metrics
- Label frequency distributions
- Confusion matrices for top classes
- Performance comparison charts

## Configuration Guide

### Key Configuration Options

```yaml
# Execution control
execution:
  run_preprocessing: true    # Process raw data
  run_baseline: false       # Skip random baseline  
  run_training: true        # Run model training

# Model selection (choose one approach)
model:
  name: "facebook/bart-large"
  model_type: "bart_classifier"     # or "hierarchical_bart"
  max_length: 768

# Training approach (choose one)
fine_tuning:
  mode: "multi_stage"        # "standard", "multi_stage", or "hierarchical_joint"
  stage: "both"              # For multi_stage: "1", "2", or "both"

# Hierarchical settings (for hierarchical_joint mode)
hierarchy:
  parent_rule: "before_dot"  # How to extract parent labels
  sequential_training: true  # Parent pre-training
  parent_epochs: 5          # Parent-only training epochs
  parent_weight: 0.15       # Parent loss weight in joint training
  fusion_type: "gated"      # "gated", "attention", or "simple"

# Data processing
data:
  frequency_threshold: 10    # Minimum label frequency
  sample_size: null         # Use all data (or specify number)

# Training parameters  
training:
  batch_size: 64            # 64 for bart_classifier, 16 for hierarchical_bart
  epochs: 15                # 15 for two-stage, 20 for hierarchical
  learning_rate: 2e-5       # 2e-5 for two-stage, 1e-5 for hierarchical
  weight_decay: 0.01        # Regularization strength
```

## Research Insights

### Why Two-Stage Training Works Best

1. **Curriculum Learning**: Learning progresses from simple (parent) to complex (child) tasks
2. **Better Initialization**: Parent classification provides semantically meaningful representations
3. **Reduced Label Noise**: Parent categories are more consistent and easier to learn
4. **Transfer Learning**: Knowledge from parent task transfers effectively to child task

### Hierarchical vs Two-Stage Comparison

**Two-Stage Advantages:**
- Simpler architecture, easier to optimize
- Better empirical performance
- More stable training process
- Clear separation of learning phases

**Hierarchical Advantages:**
- End-to-end optimization
- Joint representation learning
- Theoretical elegance
- Advanced modeling techniques

### Future Research Directions

1. **Multi-task Learning**: Incorporate additional bibliographic prediction tasks
2. **Graph Neural Networks**: Model hierarchical relationships explicitly
3. **Active Learning**: Human-in-the-loop for difficult cases
4. **Cross-lingual Transfer**: Extend to other language catalogs
5. **Attention Analysis**: Understand what textual features drive predictions

## License and Citation

This project implements academic research in automated library classification. When using this code, please cite appropriately and respect the K10plus terms of service for data usage.

**Contact**: [m.khalid@stud.uni-goettingen.de]
**Last Updated**: [25 August, 2025]
**Version**: 2.0.0