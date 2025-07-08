import pandas as pd
import os

# Define constants
DATASET_DIR = 'dataset'

def print_stats():
    metadata_df = pd.DataFrame()

    # Load data
    for lang in os.listdir(DATASET_DIR):
        metadata_path = os.path.join(DATASET_DIR, lang, "metadata.csv")
        if os.path.exists(metadata_path):
            lang_df = pd.read_csv(metadata_path)
            lang_df['language'] = lang
            metadata_df = pd.concat([metadata_df, lang_df])

    if not metadata_df.empty:
        # Convert duration from seconds to minutes
        metadata_df['duration'] = metadata_df['duration'].apply(lambda x: x / 60)

        # Calculate per-user statistics
        user_stats_df = calculate_user_stats(metadata_df)
        
        # Calculate per-language statistics
        lang_stats_df = calculate_language_stats(metadata_df)
        
        # Calculate gender statistics
        gender_stats_df = calculate_gender_stats(metadata_df)
        
        # Calculate content statistics
        content_stats = calculate_content_stats(metadata_df)
        
        # Calculate overall statistics
        overall_stats = calculate_overall_stats(metadata_df)

        # Print all statistics
        print_all_statistics(user_stats_df, lang_stats_df, gender_stats_df, 
                           content_stats, overall_stats)
    else:
        print("No metadata.csv files found.")

def calculate_user_stats(df):
    """Calculate detailed user statistics"""
    user_stats = df.groupby(['user_id', 'language']).agg({
        'sentence': 'count',
        'duration': 'sum',
        'gender': 'first'  # Get user's gender
    }).reset_index()
    
    # Add average duration per sentence
    user_stats['avg_duration_per_sentence'] = user_stats['duration'] / user_stats['sentence']
    
    # Calculate number of languages per user
    languages_per_user = df.groupby('user_id')['language'].nunique()
    
    # Merge this information back
    user_stats['total_languages'] = user_stats['user_id'].map(languages_per_user)
    
    return user_stats.rename(columns={
        'sentence': 'total_sentences',
        'duration': 'total_duration'
    }).sort_values(by='total_sentences', ascending=False)

def calculate_language_stats(df):
    """Calculate detailed language statistics"""
    return df.groupby('language').agg({
        'sentence': 'count',
        'duration': 'sum',
        'user_id': pd.Series.nunique,
        'gender': 'nunique'  # Count unique genders
    }).reset_index().rename(columns={
        'sentence': 'total_sentences',
        'duration': 'total_duration',
        'user_id': 'unique_users',
        'gender': 'gender_diversity'
    })

def calculate_gender_stats(df):
    """Calculate gender-based statistics"""
    return df.groupby(['language', 'gender']).agg({
        'sentence': 'count',
        'duration': 'sum',
        'user_id': pd.Series.nunique
    }).reset_index().rename(columns={
        'sentence': 'total_sentences',
        'duration': 'total_duration',
        'user_id': 'unique_users'
    })

def calculate_content_stats(df):
    """Calculate content-related statistics"""
    return {
        'avg_sentence_length': df.groupby('language')['sentence'].apply(
            lambda x: x.str.len().mean()
        ),
        'avg_duration_per_sentence': df.groupby('language')['duration'].mean(),
        'translation_completion': df.groupby('language')['english'].apply(
            lambda x: (x.notna().sum() / len(x)) * 100
        )
    }

def calculate_overall_stats(df):
    """Calculate overall dataset statistics"""
    return {
        'total_sentences': int(df['sentence'].count()),
        'total_duration': float(df['duration'].sum()),
        'total_users': df['user_id'].nunique(),
        'total_languages': df['language'].nunique(),
        'gender_distribution': df['gender'].value_counts().to_dict()
    }

def print_all_statistics(user_stats, lang_stats, gender_stats, content_stats, overall_stats):
    """Print all statistics in a formatted way"""
    print("\n=== Per-User Statistics ===")
    print("Top 10 Contributors:")
    print(user_stats.head(10).to_string(index=False))

    print("\n=== Per-Language Statistics ===")
    print(lang_stats.sort_values(by='total_sentences', ascending=False).to_string(index=False))

    print("\n=== Gender Distribution by Language ===")
    print(gender_stats.sort_values(by=['language', 'total_sentences'], 
                                 ascending=[True, False]).to_string(index=False))

    print("\n=== Content Statistics ===")
    print("\nAverage Sentence Length (characters):")
    print(content_stats['avg_sentence_length'].round(2))
    print("\nAverage Duration per Sentence (minutes):")
    print(content_stats['avg_duration_per_sentence'].round(2))
    print("\nTranslation Completion Rate (%):")
    print(content_stats['translation_completion'].round(2))

    print("\n=== Overall Statistics ===")
    print(f"Total sentences: {overall_stats['total_sentences']:,}")
    print(f"Total duration: {overall_stats['total_duration'] / 60:.2f} hours")
    print(f"Total unique users: {overall_stats['total_users']:,}")
    print(f"Total languages: {overall_stats['total_languages']}")
    print("\nGender Distribution:")
    for gender, count in overall_stats['gender_distribution'].items():
        print(f"  {gender}: {count:,} recordings")

if __name__ == '__main__':
    print_stats()