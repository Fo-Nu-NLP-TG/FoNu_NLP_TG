import pandas as pd
import os
import kagglehub
import urllib.request
import zipfile
import gzip
import shutil

class MultiSourceDataLoader:
    """Loader for multiple translation datasets including Kaggle, Zindi, and OPUS NLLB"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create subdirectories
        self.kaggle_dir = os.path.join(data_dir, "kaggle")
        self.zindi_dir = os.path.join(data_dir, "zindi")
        self.opus_dir = os.path.join(data_dir, "opus")
        
        for directory in [self.kaggle_dir, self.zindi_dir, self.opus_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_kaggle_dataset(self):
        """Load the Ewe-English bilingual pairs from Kaggle"""
        try:
            # The kagglehub.dataset_download() doesn't accept dest_path parameter
            # Let's use the correct syntax
            path = kagglehub.dataset_download("tchaye59/eweenglish-bilingual-pairs")
            
            print(f"Kaggle dataset downloaded to: {path}")
            
            # Load the dataset
            df = pd.read_csv(os.path.join(path, "EWE_ENGLISH.csv"))
            
            # Clean the dataset
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            
            print(f"Loaded {len(df)} Ewe-English pairs from Kaggle")
            return df
        except Exception as e:
            print(f"Error loading Kaggle dataset: {e}")
            print("If you already have the dataset, please specify its path")
            return None
    
    def load_zindi_dataset(self, train_path=None, test_path=None):
        """Load the Zindi French-Ewe/Fongbe dataset"""
        if train_path is None:
            train_path = os.path.join(self.zindi_dir, "Train.csv")
        if test_path is None:
            test_path = os.path.join(self.zindi_dir, "Test.csv")
        
        try:
            # Load training data
            if os.path.exists(train_path):
                train_df = pd.read_csv(train_path)
                print(f"Loaded {len(train_df)} rows from Zindi training data")
                
                # Filter for Ewe only
                ewe_df = train_df[train_df['Target_Language'] == 'Ewe']
                print(f"Found {len(ewe_df)} French-Ewe pairs in Zindi dataset")
                
                # Load test data if available
                if os.path.exists(test_path):
                    test_df = pd.read_csv(test_path)
                    print(f"Loaded {len(test_df)} rows from Zindi test data")
                    return ewe_df, test_df
                
                return ewe_df, None
            else:
                print(f"Zindi dataset not found at {train_path}")
                return None, None
        except Exception as e:
            print(f"Error loading Zindi dataset: {e}")
            return None, None
    
    def download_opus_nllb(self):
        """Download the OPUS NLLB Ewe-French dataset"""
        moses_url = "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/ee-fr.txt.zip"
        zip_path = os.path.join(self.opus_dir, "ee-fr.txt.zip")
        
        try:
            # Download the zip file if it doesn't exist
            if not os.path.exists(zip_path):
                print(f"Downloading OPUS NLLB dataset from {moses_url}")
                urllib.request.urlretrieve(moses_url, zip_path)
            
            # Extract the zip file
            extract_dir = os.path.join(self.opus_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"Extracted OPUS NLLB dataset to {extract_dir}")
            
            # Find the extracted files
            ee_path = os.path.join(extract_dir, "NLLB.ee-fr.ee")
            fr_path = os.path.join(extract_dir, "NLLB.ee-fr.fr")
            
            if os.path.exists(ee_path) and os.path.exists(fr_path):
                # Read the files
                with open(ee_path, 'r', encoding='utf-8') as ee_file, \
                     open(fr_path, 'r', encoding='utf-8') as fr_file:
                    ee_lines = ee_file.readlines()
                    fr_lines = fr_file.readlines()
                
                # Create a DataFrame
                opus_df = pd.DataFrame({
                    'Ewe': [line.strip() for line in ee_lines],
                    'French': [line.strip() for line in fr_lines]
                })
                
                print(f"Loaded {len(opus_df)} Ewe-French pairs from OPUS NLLB")
                return opus_df
            else:
                print(f"Extracted files not found at expected paths")
                return None
        except Exception as e:
            print(f"Error downloading or extracting OPUS NLLB dataset: {e}")
            return None
    
    def combine_datasets(self, filter_ewe_only=True):
        """Combine all datasets into a unified format"""
        datasets = []
        
        # Load Kaggle dataset (Ewe-English)
        kaggle_df = self.load_kaggle_dataset()
        if kaggle_df is not None:
            # Rename columns for consistency
            kaggle_df = kaggle_df.rename(columns={'EWE': 'Ewe', 'ENGLISH': 'English'})
            datasets.append(('kaggle', kaggle_df))
        
        # Load Zindi dataset (French-Ewe/Fongbe)
        zindi_df, zindi_test = self.load_zindi_dataset()
        if zindi_df is not None:
            # Filter for Ewe if requested
            if filter_ewe_only:
                zindi_df = zindi_df[zindi_df['Target_Language'] == 'Ewe']
            
            # Rename columns for consistency
            zindi_df = zindi_df.rename(columns={'French': 'French', 'Target': 'Ewe'})
            datasets.append(('zindi', zindi_df))
        
        # Load OPUS NLLB dataset (Ewe-French)
        opus_df = self.download_opus_nllb()
        if opus_df is not None:
            datasets.append(('opus', opus_df))
        
        # Combine all datasets
        combined_data = {
            'ewe_english': None,
            'ewe_french': None,
            'test_data': zindi_test
        }
        
        for source, df in datasets:
            if 'English' in df.columns and 'Ewe' in df.columns:
                # This is an Ewe-English dataset
                if combined_data['ewe_english'] is None:
                    combined_data['ewe_english'] = df[['Ewe', 'English']].copy()
                else:
                    combined_data['ewe_english'] = pd.concat([
                        combined_data['ewe_english'], 
                        df[['Ewe', 'English']]
                    ])
            
            if 'French' in df.columns and 'Ewe' in df.columns:
                # This is an Ewe-French dataset
                if combined_data['ewe_french'] is None:
                    combined_data['ewe_french'] = df[['Ewe', 'French']].copy()
                else:
                    combined_data['ewe_french'] = pd.concat([
                        combined_data['ewe_french'], 
                        df[['Ewe', 'French']]
                    ])
        
        # Remove duplicates
        for key in ['ewe_english', 'ewe_french']:
            if combined_data[key] is not None:
                combined_data[key] = combined_data[key].drop_duplicates()
                print(f"Combined {key} dataset has {len(combined_data[key])} unique pairs")
        
        return combined_data
    
    def save_processed_data(self, combined_data, output_dir=None):
        """Save the processed datasets to CSV files"""
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, "processed")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each dataset
        for key, df in combined_data.items():
            if df is not None:
                output_path = os.path.join(output_dir, f"{key}.csv")
                df.to_csv(output_path, index=False)
                print(f"Saved {key} dataset to {output_path}")
        
        # Create corpus files for tokenizer training
        if combined_data['ewe_english'] is not None:
            with open(os.path.join(output_dir, "ewe_corpus.txt"), "w", encoding="utf-8") as f:
                for text in combined_data['ewe_english']['Ewe'].tolist():
                    f.write(str(text) + "\n")
            
            with open(os.path.join(output_dir, "english_corpus.txt"), "w", encoding="utf-8") as f:
                for text in combined_data['ewe_english']['English'].tolist():
                    f.write(str(text) + "\n")
        
        if combined_data['ewe_french'] is not None:
            with open(os.path.join(output_dir, "french_corpus.txt"), "w", encoding="utf-8") as f:
                for text in combined_data['ewe_french']['French'].tolist():
                    f.write(str(text) + "\n")
            
            # Add to Ewe corpus if not already created
            if not os.path.exists(os.path.join(output_dir, "ewe_corpus.txt")):
                with open(os.path.join(output_dir, "ewe_corpus.txt"), "w", encoding="utf-8") as f:
                    for text in combined_data['ewe_french']['Ewe'].tolist():
                        f.write(str(text) + "\n")
        
        return output_dir

if __name__ == "__main__":
    # Create an instance of the data loader
    loader = MultiSourceDataLoader()
    
    # Combine datasets from all sources
    print("Combining datasets...")
    combined_data = loader.combine_datasets()
    
    # Save the processed data
    if combined_data:
        output_dir = loader.save_processed_data(combined_data)
        print(f"All data processing complete. Files saved to: {output_dir}")
    else:
        print("No data was combined. Please check the dataset sources.")
