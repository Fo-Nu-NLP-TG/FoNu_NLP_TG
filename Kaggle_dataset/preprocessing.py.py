from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt # Plotting
import numpy as np # Linear algebra
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)
import os # Operating System
from sklearn.preprocessing import StandardScaler # StandardScaler


import kagglehub

# Download latest version
path = kagglehub.dataset_download("tchaye59/eweenglish-bilingual-pairs")
print("Path to dataset files:", path)

# List all files in the directory where the dataset is downloaded
for filename in os.listdir(path):
    print(filename)

#### DATA PLOTTING ####

# For non-numeric columns (e.g., text like words or categories), 
# it creates a bar chart showing how many times each unique value appears 
# (using value_counts()).
# For numeric columns (e.g., numbers), 
# it creates a histogram showing the distribution of values across bins.

# Distribution graphs (histogram/bar chart) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    # Calculates the number of unique values in each column of the DataFrame
    nunique = df.nunique()
    # The DataFrame is filtered to include only columns where the number of unique values is between 1 and 50. This avoids:
    # Columns with only 1 unique value (no variation, so no point in plotting).
    # Columns with too many unique values (e.g., continuous data with >50 unique values might not suit a bar chart).
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    
    # df.shape returns the number of rows and columns in the DataFrame.
    nRow, nCol = df.shape
    # The list of column names is extracted from the DataFrame.
    columnNames = list(df)
    # Calculates the number of rows needed to display the graphs.
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    # np.ceil rounds the number of rows up to the nearest whole number.
    nGraphRow = int(np.ceil(nGraphRow))


    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    # dpi = dots per inch
    
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        # Extracts all rows :of the i-th column from the dataframe
        columnDf = df.iloc[:, i]
        # Checks if the data type of the first element in the column is not a number.
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            # If the data is not numeric, it plots a bar chart of the value counts.
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            # If the data is numeric, it plots a histogram.
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    # Calculates the correlation matrix of the DataFrame
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    # Computes the correlation matrix using Pearson correlation by default 
    # (measures linear relationships between numeric columns).
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):