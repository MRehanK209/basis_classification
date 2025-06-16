import argparse
import random
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
import json
from sklearn.metrics import matthews_corrcoef
from optimizer import AdamW
import matplotlib.pyplot as plt

TQDM_DISABLE = False


class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=1355):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large")
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Add an additional fully connected layer to obtain the logits
        logits = self.classifier(cls_output)

        # Return the probabilities
        
        return logits
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    
    args = parser.parse_args()
    return args


def accuracy_binary(predicted_labels_np, true_labels_np):
    accuracies = []
    matthews_coefficients = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

        # Use fixed label set to avoid warnings
        
        matth_coef = matthews_corrcoef(
            true_labels_np[:, label_idx],
            predicted_labels_np[:, label_idx]
        )

        matthews_coefficients.append(matth_coef)
    return np.mean(accuracies), np.mean(matthews_coefficients)

def convert_labels_to_binary(bk_list, label_map):
    """
    Converts BK codes to multi-hot binary labels of fixed length using a provided label_map.
    
    Args:
        bk_list (List[str]): List of BK values like "86.86|86.47"
        label_map (Dict[str, int]): Mapping from label string to index.
    
    Returns:
        List[List[int]]: Multi-hot encoded labels with consistent length.
    """
    label_count = len(label_map)
    binary_labels = []

    for entry in bk_list:
        label_vec = [0] * label_count
        labels = str(entry).split('|') if pd.notna(entry) else []
        for label in labels:
            if label in label_map:
                label_vec[label_map[label]] = 1
        binary_labels.append(label_vec)

    return binary_labels

def transform_data(dataset,label_map, max_length=768, batch_size = 32):

    sentences = (
    "Title: " + dataset["Title"].fillna('') + "\n" +
    "Summary: " + dataset["Summary"].fillna('') + "\n" +
    "Keywords: " + dataset["Keywords"].fillna('') + "\n" +
    "LOC_Keywords: " + dataset["LOC_Keywords"].fillna('') + "\n" +
    "RVK: " + dataset["RVK"].fillna('')
    ).tolist()

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    if 'BK' in dataset.columns:
        binary_labels = torch.tensor(convert_labels_to_binary(dataset["BK"].tolist(),label_map))
        dataset = TensorDataset(input_ids, attention_mask, binary_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else :
        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return dataloader


def train_model(model, train_data, dev_data, device, args):
    """
    Train the model. You can use any training loop you want. We recommend starting with
    AdamW as your optimizer. You can take a look at the SST training loop for reference.
    Think about your loss function and the number of epochs you want to train for.
    You can also use the evaluate_model function to evaluate the
    model on the dev set. Print the training loss, training accuracy, and dev accuracy at
    the end of each epoch.

    Return the trained model.
    """
    ### TODO

    # criterion = FocalLoss()
    history = []
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train() 
        total_loss = 0
        all_pred = []
        all_labels = []

        for batch in tqdm(train_data, desc=f"Training Epoch {epoch + 1}", disable=TQDM_DISABLE):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute loss
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Accumulate loss and predictions for metrics
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted_labels = (probs > 0.5).int().cpu().numpy()
            all_pred.extend(predicted_labels)
            all_labels.extend(labels.int().cpu().numpy())

        avg_train_loss = total_loss / len(train_data)
        train_accuracy, train_mathhews_coefficient = accuracy_binary(np.array(all_pred), np.array(all_labels))
        validation_accuracy, validation_mathhews_coefficient = evaluate_model(model, dev_data, device)
        avg_val_loss = compute_validation_loss(model, dev_data, criterion, device)

        epoch_ckpt_path = f"results/checkpoints/checkpoint_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), epoch_ckpt_path)
        print(f"Saved checkpoint: {epoch_ckpt_path}")

        history.append((avg_train_loss,avg_val_loss,train_accuracy,validation_accuracy,train_mathhews_coefficient,validation_mathhews_coefficient))

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}, Training Mathew Coefficient: {train_mathhews_coefficient:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.4f}, Validation Mathew Coefficient: {validation_mathhews_coefficient:.4f}")
    plot_training_metrics(history)

    return model

def test_model(model, test_data, test_ids, device):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    ### TODO
    model.eval()  # Set model to evaluation mode
    all_predictions = []

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            predictions = (logits > 0.5).int()
            all_predictions.append(predictions.cpu().numpy())

    all_predictions = [item for sublist in all_predictions for item in sublist]

    result_df = pd.DataFrame({
        'id': test_ids,
        'Predicted_BK': all_predictions
    })

    return result_df
    
def compute_validation_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the seven paraphrase types the model has predicted correctly for each data point.
    So, if the models prediction is [1,1,0,0,1,1,0] and the true label is [0,0,0,0,1,1,0], this predicition
    has an accuracy of 5/7, i.e. 71.4% .
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            predicted_labels = (probs > 0.5).int()


            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    accuracy1, mathhews_coefficient1 = accuracy_binary(predicted_labels_np,true_labels_np)
    return accuracy1,mathhews_coefficient1

def plot_training_metrics(history, save_path="results/metrics_plot.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Unpack history
    train_losses = [x[0] for x in history]
    val_losses = [x[1] for x in history]
    train_accuracies = [x[2] for x in history]
    val_accuracies = [x[3] for x in history]
    train_mccs = [x[4] for x in history]
    val_mccs = [x[5] for x in history]
    epochs = range(1, len(history) + 1)

    plt.figure(figsize=(18, 5))

    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Plot 2: Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    # Plot 3: MCC
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_mccs, label="Train MCC")
    plt.plot(epochs, val_mccs, label="Val MCC")
    plt.xlabel("Epoch")
    plt.ylabel("Matthews Coefficient")
    plt.title("MCC Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Training metrics saved to: {save_path}")

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def finetune_paraphrase_detection(args):
    model = BartWithClassifier()
    # model.bart.gradient_checkpointing_enable()

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = model.to(device)

    # train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    # test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t")

    # TODO You might do a split of the train data into train/validation set here
    # (or in the csv files directly)
    train_dataset = pd.read_csv("data/k10plus_processed_rare_label_removed.csv").sample(12000)
    label_map_path = "data/label_map.json"

    # Load the JSON data
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    np.random.seed(42)
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(np.arange(train_dataset.shape[0]))
    data_shuffled = train_dataset.iloc[shuffled_indices]
    train_size = int(train_ratio * len(data_shuffled))
    train_df = data_shuffled.iloc[:train_size]
    dev_df = data_shuffled.iloc[train_size:]

    train_data = transform_data(train_df, label_map)
    dev_data = transform_data(dev_df,label_map)
    # test_data = transform_data(test_dataset)


    print(f"Loaded {len(train_dataset)} training samples.")

    
    model = train_model(model, train_data, dev_data, device,args)

    print("Training finished.")

    accuracy, matthews_corr = evaluate_model(model, train_data, device)
    print(f"The accuracy of the model is: {accuracy:.4f}")
    print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.4f}")

    
if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
