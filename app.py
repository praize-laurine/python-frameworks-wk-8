# app/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import sys
import os
import numpy as np
from collections import Counter
import re

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure the page
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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data(size=500):
    """Create realistic sample data for demonstration"""
    print("Creating sample data for demonstration...")
    
    # Generate realistic sample data
    journals = [
        'Nature', 'Science', 'The Lancet', 'BMJ', 'JAMA',
        'New England Journal of Medicine', 'Cell', 'PLOS One',
        'BioRxiv', 'MedRxiv', 'Journal of Virology'
    ]
    
    topics = [
        'COVID-19', 'SARS-CoV-2', 'pandemic', 'vaccine', 'transmission',
        'lockdown', 'variants', 'public health', 'epidemiology', 'treatment'
    ]
    
    # Create sample data
    np.random.seed(42)
    
    sample_data = {
        'cord_uid': [f'uid_{i:06d}' for i in range(size)],
        'title': [
            f"Study of {np.random.choice(topics)} in {np.random.choice(['urban', 'rural', 'clinical'])} settings" 
            for _ in range(size)
        ],
        'abstract': [
            f"This research investigates {np.random.choice(topics)} through {np.random.choice(['randomized trials', 'observational studies', 'meta-analysis'])}. " +
            f"Results show significant findings in {np.random.choice(['treatment efficacy', 'transmission rates', 'public health impact'])}."
            for _ in range(size)
        ],
        'journal': np.random.choice(journals, size),
        'publish_time': pd.date_range('2020-01-01', '2022-12-31', periods=size),
        'authors': [
            f"Author_{i}_A, Author_{i}_B, Author_{i}_C" for i in range(size)
        ],
        'url': [f"https://example.com/paper/{i}" for i in range(size)]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Created sample data with {size} records")
    return df

@st.cache_data
def load_data():
    """Load data with fallback to sample data"""
    data_path = "../data/metadata.csv"
    
    try:
        if os.path.exists(data_path):
            st.success("📦 Loading real data from metadata.csv...")
            df = pd.read_csv(data_path)
            st.success(f"✅ Real data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return df, False
        else:
            st.warning("⚠️ Real data file not found. Using sample data for demonstration.")
            st.info("💡 To use real data, download metadata.csv from Kaggle and place it in the data/ directory")
            return create_sample_data(500), True
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.warning("🔄 Falling back to sample data...")
        return create_sample_data(500), True

@st.cache_data
def clean_data(df):
    """Clean and preprocess the dataset"""
    if df is None:
        st.error("No data to clean!")
        return None
        
    df_clean = df.copy()
    
    # Handle publication dates
    if 'publish_time' in df_clean.columns:
        df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
        df_clean['publication_year'] = df_clean['publish_time'].dt.year
        df_clean['publication_year'] = df_clean['publication_year'].fillna(2020)
    else:
        # Add publication year if not present
        df_clean['publication_year'] = 2020
    
    # Handle missing titles - FIXED SYNTAX ERROR HERE
    if 'title' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['title'])
    else:
        # Fixed the list comprehension syntax
        df_clean['title'] = [f"Research Paper {i}" for i in range(len(df_clean))]
    
    # Fill missing values
    if 'abstract' in df_clean.columns:
        df_clean['abstract'] = df_clean['abstract'].fillna('No abstract available')
        df_clean['has_abstract'] = df_clean['abstract'] != 'No abstract available'
    else:
        df_clean['abstract'] = 'Sample abstract'
        df_clean['has_abstract'] = True
    
    if 'journal' in df_clean.columns:
        df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    else:
        df_clean['journal'] = 'Sample Journal'
    
    # Create derived features
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    df_clean['title_word_count'] = df_clean['title'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    return df_clean

def display_dashboard(df):
    """Display the main dashboard"""
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", f"{len(df):,}")
    
    with col2:
        year_range = f"{int(df['publication_year'].min())}-{int(df['publication_year'].max())}"
        st.metric("Time Span", year_range)
    
    with col3:
        if 'has_abstract' in df.columns:
            papers_with_abstracts = df['has_abstract'].sum()
            st.metric("Papers with Abstracts", f"{papers_with_abstracts:,}")
        else:
            st.metric("Papers with Abstracts", "N/A")
    
    with col4:
        avg_abstract_length = df['abstract_word_count'].mean()
        st.metric("Avg Abstract Length", f"{avg_abstract_length:.0f} words")
    
    # Quick insights
    st.subheader("🚀 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Publications by year chart
        yearly_counts = df['publication_year'].value_counts().sort_index()
        st.bar_chart(yearly_counts)
        st.caption("Publications by Year")
    
    with col2:
        # Top journals
        top_journals = df['journal'].value_counts().head(10)
        st.bar_chart(top_journals)
        st.caption("Top 10 Journals")

def display_temporal_analysis(df):
    """Display temporal analysis section"""
    st.header("📈 Temporal Analysis")
    
    # Year range selector
    min_year = int(df['publication_year'].min())
    max_year = int(df['publication_year'].max())
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        year_range = st.slider(
            "Select Year Range:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    
    # Filter data based on selection
    filtered_df = df[
        (df['publication_year'] >= year_range[0]) & 
        (df['publication_year'] <= year_range[1])
    ]
    
    # Publications by year
    yearly_counts = filtered_df['publication_year'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_counts.plot(kind='bar', ax=ax, color='skyblue', alpha=0.7)
    ax.set_title(f'Publications from {year_range[0]} to {year_range[1]}', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Publications')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Show data table
    st.subheader("📋 Publication Statistics")
    summary_data = []
    for year in yearly_counts.index:
        year_data = filtered_df[filtered_df['publication_year'] == year]
        summary_data.append({
            'Year': year,
            'Publications': len(year_data),
            'With Abstracts': year_data['has_abstract'].sum() if 'has_abstract' in year_data.columns else 'N/A',
            'Avg Abstract Length': year_data['abstract_word_count'].mean()
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)

def display_journal_analysis(df):
    """Display journal analysis section"""
    st.header("📚 Journal Analysis")
    
    # Number of top journals to show
    n_journals = st.slider("Number of top journals to display:", 5, 25, 10)
    
    # Top journals
    top_journals = df['journal'].value_counts().head(n_journals)
    
    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    top_journals.plot(kind='barh', ax=ax, color='lightcoral', alpha=0.7)
    ax.set_title(f'Top {n_journals} Journals Publishing COVID-19 Research', 
                fontweight='bold')
    ax.set_xlabel('Number of Publications')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    
    # Journal details
    st.subheader("🔍 Journal Details")
    selected_journal = st.selectbox(
        "Select a journal for detailed information:",
        options=top_journals.index.tolist()
    )
    
    journal_papers = df[df['journal'] == selected_journal]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Papers", len(journal_papers))
    
    with col2:
        with_abstracts = journal_papers['has_abstract'].sum() if 'has_abstract' in journal_papers.columns else 'N/A'
        st.metric("Papers with Abstracts", with_abstracts)
    
    with col3:
        avg_abstract_len = journal_papers['abstract_word_count'].mean()
        st.metric("Avg Abstract Length", f"{avg_abstract_len:.0f} words")
    
    # Sample papers from selected journal
    st.subheader("📄 Sample Papers")
    sample_size = st.slider("Number of sample papers to show:", 1, 20, 5)
    sample_papers = journal_papers[['title', 'publication_year', 'authors']].head(sample_size)
    st.dataframe(sample_papers)

def display_text_analysis(df):
    """Display text analysis section"""
    st.header("📝 Text Analysis")
    
    # Word cloud
    st.subheader("Word Cloud of Paper Titles")
    
    # Generate word frequency
    all_titles = ' '.join(df['title'].astype(str))
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

def display_data_explorer(df):
    """Display interactive data explorer"""
    st.header("🔍 Data Explorer")
    
    # Filters
    st.subheader("🔧 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_year = int(df['publication_year'].min())
        max_year = int(df['publication_year'].max())
        year_filter = st.slider("Publication Year", min_year, max_year, (min_year, max_year))
    
    with col2:
        min_words = int(df['abstract_word_count'].min())
        max_words = int(df['abstract_word_count'].max())
        word_filter = st.slider("Abstract Word Count", min_words, max_words, (0, max_words))
    
    with col3:
        has_abstract_filter = st.selectbox("Has Abstract", ["All", "Yes", "No"])
    
    # Apply filters
    filtered_df = df[
        (df['publication_year'] >= year_filter[0]) & 
        (df['publication_year'] <= year_filter[1]) &
        (df['abstract_word_count'] >= word_filter[0]) & 
        (df['abstract_word_count'] <= word_filter[1])
    ]
    
    if has_abstract_filter == "Yes" and 'has_abstract' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['has_abstract'] == True]
    elif has_abstract_filter == "No" and 'has_abstract' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['has_abstract'] == False]
    
    st.info(f"📊 Showing {len(filtered_df)} papers after filtering")
    
    # Column selection
    st.subheader("📋 Data View")
    available_columns = df.columns.tolist()
    default_columns = ['title', 'journal', 'publication_year', 'authors', 'abstract_word_count']
    selected_columns = st.multiselect(
        "Select columns to display:",
        options=available_columns,
        default=default_columns
    )
    
    # Number of rows to show
    n_rows = st.slider("Number of rows to display:", 5, 100, 10)
    
    if selected_columns:
        st.dataframe(filtered_df[selected_columns].head(n_rows))

def main():
    # Header
    st.markdown('<div class="main-header">🔬 CORD-19 Research Paper Explorer</div>', 
                unsafe_allow_html=True)
    st.write("Explore COVID-19 research papers from the CORD-19 dataset")
    
    # Load data
    with st.spinner('Loading data...'):
        result = load_data()
    
    if result is None:
        st.error("❌ Failed to load data. Please check the data file and try again.")
        return
    
    df, is_sample_data = result
    
    if df is None:
        st.error("❌ No data available. Please check your data source.")
        return
    
    # Clean data
    with st.spinner('Processing data...'):
        df_clean = clean_data(df)
    
    if df_clean is None:
        st.error("❌ Failed to clean data.")
        return
    
    if is_sample_data:
        st.warning("🔸 Currently using SAMPLE DATA. To use real COVID-19 research data, download metadata.csv from Kaggle and place it in the data/ directory.")
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    section = st.sidebar.radio(
        "Choose a section:",
        ["🏠 Dashboard", "📈 Temporal Analysis", "📚 Journal Analysis", "📝 Text Analysis", "🔍 Data Explorer"]
    )
    
    # Display selected section
    if section == "🏠 Dashboard":
        display_dashboard(df_clean)
    elif section == "📈 Temporal Analysis":
        display_temporal_analysis(df_clean)
    elif section == "📚 Journal Analysis":
        display_journal_analysis(df_clean)
    elif section == "📝 Text Analysis":
        display_text_analysis(df_clean)
    elif section == "🔍 Data Explorer":
        display_data_explorer(df_clean)

if __name__ == "__main__":
    main()