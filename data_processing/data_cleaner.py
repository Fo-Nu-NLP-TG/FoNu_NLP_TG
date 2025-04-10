import re
import pandas as pd
import os
from langdetect import detect, LangDetectException

class DataCleaner:
    """Clean and preprocess translation data"""
    
    def __init__(self, data_dir="./data/processed"):
        self.data_dir = data_dir
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters (customize as needed)
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        return text.strip()
    
    def filter_by_length(self, df, src_col, tgt_col, min_len=3, max_len=100):
        """Filter out sentence pairs that are too short or too long"""
        mask = (df[src_col].str.split().str.len() >= min_len) & \
               (df[src_col].str.split().str.len() <= max_len) & \
               (df[tgt_col].str.split().str.len() >= min_len) & \
               (df[tgt_col].str.split().str.len() <= max_len)
        return df[mask]
    
    def filter_by_language(self, df, src_col, tgt_col, src_lang='ee', tgt_lang='en'):
        """Filter out sentence pairs with wrong languages using langdetect"""
        def is_correct_language(row):
            try:
                # For Ewe, we might need a custom approach as langdetect might not support it
                if src_lang == 'ee':
                    src_detected = True  # Skip detection for Ewe
                else:
                    src_detected = detect(row[src_col]) == src_lang
                
                tgt_detected = detect(row[tgt_col]) == tgt_lang
                return src_detected and tgt_detected
            except LangDetectException:
                return False
        
        return df[df.apply(is_correct_language, axis=1)]
    
    def preprocess_dataset(self, file_name, src_col, tgt_col, src_lang='ee', tgt_lang='en'):
        """Apply full preprocessing pipeline to a dataset"""
        file_path = os.path.join(self.data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        print(f"Preprocessing {file_name}...")
        df = pd.read_csv(file_path)
        
        # 1. Basic cleaning
        print("Applying text cleaning...")
        df[src_col] = df[src_col].apply(self.clean_text)
        df[tgt_col] = df[tgt_col].apply(self.clean_text)
        
        # 2. Remove empty rows
        df = df[(df[src_col].str.strip() != '') & (df[tgt_col].str.strip() != '')]
        
        # 3. Length filtering
        print("Filtering by length...")
        original_len = len(df)
        df = self.filter_by_length(df, src_col, tgt_col, min_len=3, max_len=100)
        print(f"Removed {original_len - len(df)} pairs based on length criteria")
        
        # 4. Optional language detection (can be slow)
        # print("Filtering by language...")
        # original_len = len(df)
        # df = self.filter_by_language(df, src_col, tgt_col, src_lang, tgt_lang)
        # print(f"Removed {original_len - len(df)} pairs based on language detection")
        
        # Save preprocessed data
        output_path = os.path.join(self.data_dir, f"clean_{file_name}")
        df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
        
        # Update corpus files
        self.update_corpus_file(df[src_col], f"{src_lang}_corpus.txt")
        self.update_corpus_file(df[tgt_col], f"{tgt_lang}_corpus.txt")
        
        return df
    
    def update_corpus_file(self, series, corpus_filename):
        """Update corpus file with cleaned text"""
        corpus_path = os.path.join(self.data_dir, corpus_filename)
        with open(corpus_path, "w", encoding="utf-8") as f:
            for text in series:
                f.write(str(text) + "\n")
        print(f"Updated corpus file: {corpus_path}")
    
    def preprocess_all_datasets(self):
        """Preprocess all available datasets"""
        # Process Ewe-English dataset
        if os.path.exists(os.path.join(self.data_dir, "ewe_english.csv")):
            self.preprocess_dataset("ewe_english.csv", "Ewe", "English", "ee", "en")
        
        # Process Ewe-French dataset
        if os.path.exists(os.path.join(self.data_dir, "ewe_french.csv")):
            self.preprocess_dataset("ewe_french.csv", "Ewe", "French", "ee", "fr")
        
        print("All preprocessing complete!")

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.preprocess_all_datasets()