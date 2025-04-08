import numpy as np # Linear algebra
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("tchaye59/eweenglish-bilingual-pairs")
print("Path to dataset files:", path)

# List all files in the directory where the dataset is downloaded
for filename in os.listdir(path):
    print(filename)

ee = pd.read_csv(path + "/EWE_ENGLISH.csv")
print(ee.head())