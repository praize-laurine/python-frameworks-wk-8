import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from wordcloud import WordCloud # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore

def set_plot_style():
    """Set consistent plotting style"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_publications_over_time(papers_by_year, save_path=None):
    """Plot number of publications over time"""
    set_plot_style()
    
    plt.figure(figsize=(12, 6))
    papers_by_year.plot(kind='bar', color='skyblue')
    plt.title('Number of COVID-19 Publications by Year', fontsize=16, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_top_journals(top_journals, save_path=None):
    """Plot top publishing journals"""
    set_plot_style()
    
    plt.figure(figsize=(12, 8))
    top_journals.head(10).plot(kind='barh', color='lightcoral')
    plt.title('Top 10 Journals Publishing COVID-19 Research', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Publications')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_wordcloud(word_freq, save_path=None):
    """Create word cloud from titles"""
    set_plot_style()
    
    # Remove common stop words
    stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 
                  'as', 'an', 'from', 'at', 'that', 'is', 'this', 'are', 'be', 'was'}
    
    filtered_words = {word: count for word, count in word_freq.items() 
                     if word not in stop_words and len(word) > 2}
    
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                         max_words=100).generate_from_frequencies(filtered_words)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Most Frequent Words in Paper Titles', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_source_distribution(df_clean, save_path=None):
    """Plot distribution of papers by source"""
    set_plot_style()
    
    source_counts = df_clean['source'].value_counts().head(10)
    
    plt.figure(figsize=(10, 6))
    source_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Papers by Source (Top 10)', fontsize=16, fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_abstract_length_distribution(df_clean, save_path=None):
    """Plot distribution of abstract lengths"""
    set_plot_style()
    
    plt.figure(figsize=(10, 6))
    plt.hist(df_clean['abstract_word_count'], bins=50, color='lightgreen', alpha=0.7)
    plt.title('Distribution of Abstract Word Count', fontsize=16, fontweight='bold')
    plt.xlabel('Word Count')
    plt.ylabel('Number of Papers')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def perform_complete_analysis(df_clean):
    """Perform complete analysis and generate all visualizations"""
    from data_cleaning import prepare_analysis_data
    
    papers_by_year, top_journals, word_freq = prepare_analysis_data(df_clean)
    
    print("=== DATA ANALYSIS RESULTS ===")
    print(f"\nTotal papers: {len(df_clean)}")
    print(f"Time range: {df_clean['publication_year'].min()} - {df_clean['publication_year'].max()}")
    print(f"\nTop 5 years by publication count:")
    print(papers_by_year.head())
    
    print(f"\nTop 5 journals:")
    print(top_journals.head())
    
    print(f"\nTop 10 words in titles:")
    for word, count in list(word_freq.most_common(10)):
        print(f"{word}: {count}")
    
    # Generate visualizations
    plot_publications_over_time(papers_by_year, 'publications_by_year.png')
    plot_top_journals(top_journals, 'top_journals.png')
    create_wordcloud(word_freq, 'wordcloud.png')
    plot_source_distribution(df_clean, 'source_distribution.png')
    plot_abstract_length_distribution(df_clean, 'abstract_length.png')
    
    return papers_by_year, top_journals, word_freq

if __name__ == "__main__":
    pass