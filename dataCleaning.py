import pandas as pd # type: ignore
import numpy as np # type: ignore
from datetime import datetime

def clean_data(df):
    """
    Clean and prepare the CORD-19 dataset
    """
    print("=== DATA CLEANING AND PREPARATION ===")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle publication date
    print("\n1. Handling publication dates...")
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['publication_year'] = df_clean['publish_time'].dt.year
    
    # Fill missing years with 2020 (most common year for COVID papers)
    df_clean['publication_year'] = df_clean['publication_year'].fillna(2020)
    
    # Create abstract word count
    print("\n2. Creating abstract word count...")
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    # Create title word count
    df_clean['title_word_count'] = df_clean['title'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    # Handle missing values in key columns
    print("\n3. Handling missing values...")
    
    # For title - keep only rows with titles
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['title'])
    print(f"Removed {initial_count - len(df_clean)} rows with missing titles")
    
    # Fill missing abstracts with empty string
    df_clean['abstract'] = df_clean['abstract'].fillna('')
    
    # Fill missing journal information
    df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    
    # Extract source information
    df_clean['source'] = df_clean['source_x'].fillna('Unknown Source')
    
    print(f"\nFinal cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    
    return df_clean

def prepare_analysis_data(df_clean):
    """
    Prepare specific datasets for analysis
    """
    # Papers by year
    papers_by_year = df_clean['publication_year'].value_counts().sort_index()
    
    # Top journals
    top_journals = df_clean['journal'].value_counts().head(20)
    
    # Word frequency in titles
    all_titles = ' '.join(df_clean['title'].astype(str))
    words = re.findall(r'\b[a-zA-Z]+\b', all_titles.lower())
    word_freq = Counter(words)
    
    return papers_by_year, top_journals, word_freq

if __name__ == "__main__":
    # This would be called after loading the data
    pass