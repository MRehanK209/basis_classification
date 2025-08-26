# BK Classification Pipeline

A comprehensive system for automatic classification of bibliographic records using hierarchical BK (Basisklassifikation) codes. This project implements a complete pipeline from data extraction from the K10plus library catalog to fine-tuning state-of-the-art transformer models.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Extraction](#data-extraction)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation and Results](#evaluation-and-results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results and Performance](#results-and-performance)
- [Configuration](#configuration)
- [Directory Structure](#directory-structure)

## Project Overview

This project aims to automatically classify bibliographic records using BK (Basisklassifikation) codes, a German library classification system. The system leverages advanced NLP techniques and hierarchical modeling to predict multiple classification codes from bibliographic metadata.

### Key Features

- **Asynchronous data extraction** from K10plus SRU API
- **Hierarchical multi-label classification** using BART transformers
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
5. **Train/Val/Test Split**: 70/15/15 stratified split

**Statistics:**
- **Raw Records**: 251,172 bibliographic entries
- **After Filtering**: ~140K records with valid BK codes
- **Final Labels**: 1,884 unique BK classification codes
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

## Model Architecture

### 1. Standard BART Classifier

Basic multi-label classification using BART-large:

```python
class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=1884, model_name="facebook/bart-large"):
        self.bart = BartModel.from_pretrained(model_name)
        self.classifier = nn.Linear(hidden_size, num_labels)
```

### 2. Hierarchical BART Classifier

Advanced hierarchical model leveraging BK code structure:

**Key Features:**
- **Dual-head architecture**: Separate parent and child classifiers
- **Hierarchical fusion**: Gated fusion mechanism for parent-child relationships
- **Constraint enforcement**: Hierarchical consistency during inference
- **Sequential training**: Parent task pre-training followed by joint training

**Architecture Components:**
```python
class ImprovedHierarchicalBartClassifier(nn.Module):
    - Shared BART backbone
    - Parent classifier (before-dot codes)
    - Child classifier (full codes)
    - Gated fusion mechanism
    - Hierarchy constraint modules
```

## Training Pipeline

### Training Configuration

**Anti-Overfitting Strategy:**
- Learning rate: 1e-5 (conservative)
- Weight decay: 0.05 (high regularization)
- Batch size: 16 with gradient accumulation (steps=2)
- Mixed precision training enabled
- Early stopping (patience=5, monitor=MCC)

**Hierarchical Training:**
- Sequential training: 5 epochs parent → joint training
- Parent weight: 0.15 (85% focus on child task)
- Scheduled sampling for robustness
- Hierarchy penalty: 0.05 (light constraint)

### Loss Functions

1. **Standard Training**: BCEWithLogitsLoss
2. **Hierarchical Training**: Weighted combination of parent and child losses

```python
total_loss = (1 - parent_weight) * child_loss + parent_weight * parent_loss
```

## Evaluation and Results

### Metrics

The system evaluates using multiple metrics across different prediction thresholds:

**Primary Metrics:**
- **MCC (Matthews Correlation Coefficient)**: Primary optimization target
- **F1-Score**: Micro and macro-averaged
- **Precision/Recall**: Micro and macro-averaged
- **Subset Accuracy**: Exact match accuracy

**Evaluation Thresholds**: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

### Latest Results (Hierarchical BART)

**Best Performance (Threshold = 0.25):**

**Parent Level Classification:**
- Subset Accuracy: 50.7%
- MCC: 0.724
- F1-Micro: 0.700
- F1-Macro: 0.640

**Child Level Classification:**
- Subset Accuracy: 12.2%
- MCC: 0.409
- F1-Micro: 0.401
- F1-Macro: 0.139

**Model Configuration:**
- Model: facebook/bart-large (hierarchical)
- Batch Size: 16
- Epochs: 20 (early stopped)
- Total Parameters: ~400M
- Training Time: ~24 hours

### Performance Analysis

The testing notebook (`testing.ipynb`) provides comprehensive data analysis including:

1. **Label Distribution Analysis**: Frequency plots and statistics
2. **Data Quality Assessment**: Missing value analysis
3. **Classification Performance**: Per-class precision/recall
4. **Hierarchical Consistency**: Parent-child relationship validation

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

### Hardware Requirements

- **GPU**: NVIDIA GPU with ≥24GB VRAM (recommended)
- **RAM**: ≥32GB system memory
- **Storage**: ≥50GB free space

## Usage

### Quick Start

1. **Run Complete Pipeline:**
```bash
python pipeline.py --config config/config.yaml
```

2. **Run Specific Components:**
```bash
# Training only (requires preprocessed data)
python pipeline.py --config config/config.yaml 
```

### Configuration

Modify `config/config.yaml` to customize:

```yaml
# Execution control
execution:
  run_preprocessing: true
  run_baseline: false
  run_training: true

# Model selection
model:
  name: "facebook/bart-large"
  model_type: "hierarchical_bart"  # or "bart_classifier"
  max_length: 768

# Training parameters
training:
  batch_size: 16
  epochs: 20
  learning_rate: 1e-5
  weight_decay: 0.05
```

## Results and Performance

### Model Comparison

| Model | Architecture | MCC | F1-Micro | F1-Macro | Subset Acc. |
|-------|-------------|-----|----------|----------|-------------|
| Random Baseline | - | ~0.001 | ~0.002 | ~0.001 | ~0.00% |
| BART Classifier | Single-head | 0.387 | 0.384 | 0.126 | 11.1% |
| Hierarchical BART | Dual-head | **0.409** | **0.401** | **0.139** | **12.2%** |

### Key Findings

1. **Hierarchical modeling** provides consistent improvements over standard classification
2. **Parent-level classification** achieves much higher performance (50.7% vs 12.2% subset accuracy)
3. **Optimal threshold** around 0.25 balances precision and recall
4. **Label imbalance** remains a significant challenge (1,884 classes)

### Generated Plots and Analysis

The `testing.ipynb` notebook generates various visualizations:

1. **Label Frequency Distribution**: Shows power-law distribution of BK codes
2. **Data Quality Heatmaps**: Missing value patterns across fields
3. **Performance by Threshold**: Precision-recall curves
4. **Hierarchical Consistency**: Parent-child prediction alignment
5. **Training Curves**: Loss and metric evolution during training

## Directory Structure
