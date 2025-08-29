# Hugging Face Hub Setup Instructions

This guide helps you upload your best-performing two-stage BART model to Hugging Face Hub and use the updated inference notebook.

## Quick Overview

1. **Upload Model**: Push your trained model to Hugging Face Hub
2. **Update Inference**: Use the new inference notebook that loads from HF Hub
3. **Share**: Your model becomes accessible worldwide for research

---

## Step 1: Upload Model to Hugging Face Hub

### Prerequisites
```bash
# Install required packages
pip install huggingface_hub

# Login to Hugging Face (you'll need an account)
huggingface-cli login
```

### Upload Your Model
```bash
# 1. Edit the upload script
nano upload_to_huggingface.py

# 2. Change this line (around line 232):
HF_MODEL_NAME = "YOUR_USERNAME/bk-classification-bart-two-stage"

# 3. Uncomment the upload line (around line 242):
upload_model_to_hf(MODEL_PATH, LABEL_MAP_PATH, HF_MODEL_NAME, private=False)

# 4. Run the upload script
python upload_to_huggingface.py
```

### Expected Output
```
Starting upload to Hugging Face Hub...
Model will be uploaded to: YOUR_USERNAME/bk-classification-bart-two-stage
Loading label mapping...
Loaded 1884 BK labels
Loading trained model...
Loaded model from epoch 15, best metric: 0.214
Creating model configuration...
Saving model files...
Uploading files to Hugging Face Hub...
Successfully uploaded model to: https://huggingface.co/YOUR_USERNAME/bk-classification-bart-two-stage
```

---

## Step 2: Use the Updated Inference Notebook

### Files Created
- `upload_to_huggingface.py` - Upload script
- `inference_hf.ipynb` - Updated inference notebook (loads from HF Hub)
- `inference.ipynb` - Original notebook (loads locally) - keep as backup

### Update the Notebook
1. **Open**: `inference_hf.ipynb`
2. **Edit Cell 3**: Change the model name to your repository:
   ```python
   # Change this line:
   model_name="MRehanK209/bk-classification-bart-two-stage"
   # To:
   model_name="YOUR_USERNAME/bk-classification-bart-two-stage"
   ```
3. **Uncomment**: The model loading line in Cell 3
4. **Run**: All cells to test


## Expected Performance

After uploading, your model should maintain the same performance:
- **Subset Accuracy**: 25.7%
- **MCC**: 0.498
- **F1-Micro**: 47.9%
- **F1-Macro**: 21.4%

---

## Sharing Your Model

Once uploaded, your model will be available at:
```
https://huggingface.co/YOUR_USERNAME/bk-classification-bart-two-stage
```

### Features of Your HF Model Repository
- **Model Card**: Detailed documentation
- **Tags**: Automatically tagged for discoverability  
- **Performance Metrics**: Visible on model page
- **Usage Examples**: Ready-to-run code
- **Citation Information**: For academic use

---
