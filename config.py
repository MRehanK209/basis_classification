import argparse

def get_config():
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="data/k10plus_processed_rare_label_removed.csv")
    parser.add_argument("--label_map_path", type=str, default="data/label_map.json")
    parser.add_argument("--sample_size", type=int, default=100000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="bart_classifier", choices=["bart_classifier", "hierarchical_bart"])
    parser.add_argument("--num_labels", type=int, default=2000)
    parser.add_argument("--model_name", type=str, default="facebook/bart-large")
    parser.add_argument("--max_length", type=int, default=768)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--use_mixed_precision", action="store_true")
    
    # System parameters
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--save_dir", type=str, default="results/checkpoints")
    
    return parser.parse_args()
