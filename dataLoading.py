import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load the data
def load_data(file_path='metadata.csv'):
    """
    Load the CORD-19 metadata file
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print("File not found. Please ensure metadata.csv is in the correct directory.")
        return None

def basic_exploration(df):
    """
    Perform basic data exploration
    """
    print("=== BASIC DATA EXPLORATION ===")
    
    # DataFrame dimensions
    print(f"\n1. DataFrame Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Data types
    print("\n2. Data Types:")
    print(df.dtypes)
    
    # First few rows
    print("\n3. First 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\n4. Missing Values by Column:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    }).sort_values('Missing Count', ascending=False)
    
    print(missing_info.head(10))
    
    # Basic statistics for numerical columns
    print("\n5. Basic Statistics:")
    print(df.describe())
    
    return missing_info

if __name__ == "__main__":
    # Load data
    df = load_data()
    
    if df is not None:
        # Basic exploration
        missing_info = basic_exploration(df)