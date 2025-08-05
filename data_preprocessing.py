import pandas as pd
import numpy as np
import json
import os
from collections import Counter
from pathlib import Path

class BKDataProcessor:
    """
    Handles BK label filtering and label map generation with configurable thresholds
    """
    
    def __init__(self, frequency_threshold=10):
        self.frequency_threshold = frequency_threshold
        self.label_map = {}
        self.filtered_labels = set()
        self.rare_labels = set()
        
    def analyze_label_frequency(self, bk_series):
        """Analyze frequency of all BK labels"""
        all_labels = []
        for bk in bk_series.dropna():
            if pd.notna(bk):
                all_labels.extend(str(bk).split("|"))
        
        label_counts = Counter(all_labels)
        
        # Separate rare and frequent labels
        self.rare_labels = {label for label, count in label_counts.items() 
                           if count <= self.frequency_threshold}
        self.filtered_labels = {label for label, count in label_counts.items() 
                               if count > self.frequency_threshold}
        
        print(f"Total unique labels: {len(label_counts)}")
        print(f"Labels with frequency > {self.frequency_threshold}: {len(self.filtered_labels)}")
        print(f"Rare labels (frequency <= {self.frequency_threshold}): {len(self.rare_labels)}")
        
        return label_counts
    
    def filter_rare_labels(self, bk_series):
        """Remove rare labels from BK codes"""
        def filter_row(bk_value):
            if pd.isna(bk_value):
                return None
            
            labels = str(bk_value).split("|")
            filtered = [l for l in labels if l not in self.rare_labels]
            return "|".join(filtered) if filtered else None
        
        return bk_series.apply(filter_row)
    
    def build_label_map(self, bk_series, save_path=None):
        """Build label map from filtered BK codes"""
        # Extract all labels after filtering
        all_labels = set()
        for bk in bk_series.dropna():
            if pd.notna(bk):
                all_labels.update(str(bk).split("|"))
        
        # Create sorted label map
        self.label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(self.label_map, f, indent=2)
            print(f"Saved label map with {len(self.label_map)} labels to: {save_path}")
        
        return self.label_map
    
    def process_dataset(self, df, save_filtered_data=True, data_path="data/k10plus_processed_rare_label_removed.csv",
                       label_map_path="data/label_map.json"):
        """Complete processing pipeline"""
        print(f"Processing dataset with frequency threshold: {self.frequency_threshold}")
        print(f"Original dataset shape: {df.shape}")
        
        # Step 1: Analyze label frequency
        label_counts = self.analyze_label_frequency(df["BK"])
        
        # Step 2: Filter rare labels
        df_processed = df.copy()
        df_processed["BK"] = self.filter_rare_labels(df["BK"])
        
        # Remove rows with no labels after filtering
        df_processed = df_processed[~df_processed["BK"].isna()]
        print(f"Dataset shape after filtering: {df_processed.shape}")
        
        # Step 3: Build label map
        self.build_label_map(df_processed["BK"], save_path=label_map_path)
        
        # Step 4: Save processed data
        if save_filtered_data:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df_processed.to_csv(data_path, index=False)
            print(f"Saved filtered dataset to: {data_path}")
        
        return df_processed, self.label_map

def load_and_preprocess_data(frequency_threshold=10, 
                            data_source_path="data/k10plus_2010_to_2020.csv",
                            output_data_path="data/k10plus_processed_rare_label_removed.csv",
                            label_map_path="data/label_map.json"):
    """Main function to load and preprocess data with specified threshold"""
    
    # Load raw data
    if os.path.exists(data_source_path):
        df = pd.read_csv(data_source_path)
    else:
        # If processed data doesn't exist, create it from k10plus_2010_to_2020 folder
        print("Loading data from k10plus_2010_to_2020 folder...")
        df = load_from_yearly_files()
    
    # Initialize processor
    processor = BKDataProcessor(frequency_threshold=frequency_threshold)
    
    # Process the data
    df_processed, label_map = processor.process_dataset(
        df, 
        data_path=output_data_path,
        label_map_path=label_map_path
    )
    
    return df_processed, label_map, processor

def load_from_yearly_files(folder_path="k10plus_2010_to_2020"):
    """Load and combine data from yearly CSV files"""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        
        # Extract year from filename
        parts = file.split('_')
        for part in parts:
            if part.isdigit() and len(part) == 4:
                df["Extraction_Year"] = part
                
        # Remove rows without BK codes
        df = df[~df['BK'].isna()]
        dfs.append(df)
    
    # Combine all dataframes
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Basic cleaning from your notebook
    def is_number(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    # Filter numeric BK codes only
    final_df["BK_split"] = final_df["BK"].fillna("").apply(lambda x: x.split("|") if x else [])
    final_df["BK_all_numeric"] = final_df["BK_split"].apply(lambda bk_list: all(is_number(x) for x in bk_list))
    final_df = final_df[final_df['BK_all_numeric'] == True]
    
    # Filter by number of labels (<=6 as in your notebook)
    final_df["num_labels"] = final_df["BK_split"].apply(len)
    final_df = final_df[final_df["num_labels"] <= 6].copy()
    
    # Remove rows with BK but no content
    content_cols = ["Title", "Summary", "Keywords", "LOC_Keywords", "RVK"]
    mask = (
        final_df["BK"].notna() &
        final_df[content_cols].isna().all(axis=1)
    )
    final_df = final_df[~mask].copy()
    
    return final_df

if __name__ == "__main__":
    # Example usage
    df_processed, label_map, processor = load_and_preprocess_data(frequency_threshold=10)
    print(f"Final dataset: {len(df_processed)} records with {len(label_map)} unique labels")