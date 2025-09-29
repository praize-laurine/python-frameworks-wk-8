# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import re

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('metadata.csv')
        return df
    except FileNotFoundError:
        st.error("File 'metadata.csv' not found. Please ensure it's in the correct directory.")
        return None

@st.cache_data
def clean_data(df):
    """Clean the dataset"""
    df_clean = df.copy()
    
    # Handle dates
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['publication_year'] = df_clean['publish_time'].dt.year.fillna(2020)
    
    # Handle missing values
    df_clean = df_clean.dropna(subset=['title'])
    df_clean['abstract'] = df_clean['abstract'].fillna('')
    df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    df_clean['source'] = df_clean['source_x'].fillna('Unknown Source')
    
    # Create word counts
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    return df_clean

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
    st.write("Explore COVID-19 research papers from the CORD-19 dataset")
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Clean data
    with st.spinner('Cleaning data...'):
        df_clean = clean_data(df)
    
    # Sidebar
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to:", [
        "Dataset Overview", 
        "Temporal Analysis", 
        "Journal Analysis", 
        "Text Analysis",
        "Sample Data"
    ])
    
    # Dataset Overview Section
    if section == "Dataset Overview":
        st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Papers", len(df_clean))
        
        with col2:
            st.metric("Columns", df_clean.shape[1])
        
        with col3:
            year_range = f"{int(df_clean['publication_year'].min())} - {int(df_clean['publication_year'].max())}"
            st.metric("Publication Years", year_range)
        
        # Data summary
        st.subheader("Data Summary")
        st.write(f"**Dataset Shape:** {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
        
        # Missing values
        st.subheader("Missing Values")
        missing_data = df_clean.isnull().sum()
        missing_percent = (missing_data / len(df_clean)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        st.dataframe(missing_df.head(10))
    
    # Temporal Analysis Section
    elif section == "Temporal Analysis":
        st.markdown('<h2 class="section-header">📈 Temporal Analysis</h2>', unsafe_allow_html=True)
        
        # Year range selector
        min_year = int(df_clean['publication_year'].min())
        max_year = int(df_clean['publication_year'].max())
        year_range = st.slider(
            "Select year range:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        
        # Filter data by year range
        filtered_data = df_clean[
            (df_clean['publication_year'] >= year_range[0]) & 
            (df_clean['publication_year'] <= year_range[1])
        ]
        
        # Publications by year
        yearly_counts = filtered_data['publication_year'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        yearly_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f'Publications by Year ({year_range[0]}-{year_range[1]})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Monthly trends for recent years
        if year_range[1] - year_range[0] <= 3:
            st.subheader("Monthly Publication Trends")
            filtered_data['publication_month'] = filtered_data['publish_time'].dt.to_period('M')
            monthly_counts = filtered_data['publication_month'].value_counts().sort_index()
            
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            monthly_counts.plot(kind='line', ax=ax2, marker='o', color='coral')
            ax2.set_title('Monthly Publication Trends', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Number of Publications')
            ax2.grid(alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig2)
    
    # Journal Analysis Section
    elif section == "Journal Analysis":
        st.markdown('<h2 class="section-header">📚 Journal Analysis</h2>', unsafe_allow_html=True)
        
        # Number of top journals to show
        n_journals = st.slider("Number of top journals to display:", 5, 20, 10)
        
        # Top journals
        top_journals = df_clean['journal'].value_counts().head(n_journals)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_journals.plot(kind='barh', ax=ax, color='lightcoral')
        ax.set_title(f'Top {n_journals} Journals Publishing COVID-19 Research', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Publications')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        
        # Journal selection for detailed view
        selected_journal = st.selectbox("Select a journal for detailed information:", 
                                       top_journals.index.tolist())
        
        journal_papers = df_clean[df_clean['journal'] == selected_journal]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Papers from selected journal", len(journal_papers))
        
        with col2:
            avg_words = journal_papers['abstract_word_count'].mean()
            st.metric("Average abstract length", f"{avg_words:.0f} words")
        
        # Show sample papers from selected journal
        st.subheader(f"Sample papers from {selected_journal}")
        sample_papers = journal_papers[['title', 'publication_year', 'authors']].head(5)
        st.dataframe(sample_papers)
    
    # Text Analysis Section
    elif section == "Text Analysis":
        st.markdown('<h2 class="section-header">📝 Text Analysis</h2>', unsafe_allow_html=True)
        
        # Word cloud
        st.subheader("Word Cloud of Paper Titles")
        
        # Generate word frequency
        all_titles = ' '.join(df_clean['title'].astype(str))
        words = re.findall(r'\b[a-zA-Z]+\b', all_titles.lower())
        word_freq = Counter(words)
        
        # Remove stop words
        stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 
                     'as', 'an', 'from', 'at', 'that', 'is', 'this', 'are', 'be', 'was'}
        filtered_words = {word: count for word, count in word_freq.items() 
                         if word not in stop_words and len(word) > 2}
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             max_words=100).generate_from_frequencies(filtered_words)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Most Frequent Words in Paper Titles', fontsize=16, fontweight='bold')
        st.pyplot(fig)
        
        # Top words table
        st.subheader("Top 20 Most Frequent Words")
        top_words = pd.DataFrame(filtered_words.items(), columns=['Word', 'Frequency'])
        top_words = top_words.sort_values('Frequency', ascending=False).head(20)
        st.dataframe(top_words)
        
        # Abstract length distribution
        st.subheader("Abstract Length Distribution")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(df_clean['abstract_word_count'], bins=50, color='lightgreen', alpha=0.7)
        ax2.set_title('Distribution of Abstract Word Count', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Word Count')
        ax2.set_ylabel('Number of Papers')
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        
        # Abstract length statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean abstract length", f"{df_clean['abstract_word_count'].mean():.0f} words")
        with col2:
            st.metric("Median abstract length", f"{df_clean['abstract_word_count'].median():.0f} words")
        with col3:
            st.metric("Longest abstract", f"{df_clean['abstract_word_count'].max()} words")
    
    # Sample Data Section
    elif section == "Sample Data":
        st.markdown('<h2 class="section-header">🔍 Sample Data</h2>', unsafe_allow_html=True)
        
        # Number of rows to show
        n_rows = st.slider("Number of rows to display:", 5, 100, 10)
        
        # Column selection
        available_columns = df_clean.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to display:",
            available_columns,
            default=['title', 'journal', 'publication_year', 'authors']
        )
        
        if selected_columns:
            st.dataframe(df_clean[selected_columns].head(n_rows))
        
        # Data download
        st.subheader("Download Data")
        csv = df_clean[selected_columns].to_csv(index=False) if selected_columns else df_clean.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="cord19_filtered.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()