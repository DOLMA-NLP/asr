import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = "common_sentences.csv"
OUTPUT_DIR = "languages"

# Log the start of the program
logger.info("Starting the script...")

# Read the CSV file
try:
    df = pd.read_csv(INPUT_FILE, header=0, names=["English", "Gilaki", "Mazanderani", "Talysh", "Laki Kurdish", "Luri Bakhtiari", "Hawrami", "Southern Kurdish", "Zazaki"])
    df.columns = df.columns.str.lower()
    logger.info(f"Loaded data from {INPUT_FILE} with {len(df)} rows.")
except Exception as e:
    logger.error(f"Failed to read input file {INPUT_FILE}. Error: {e}")
    raise

def sort_rows_by_translation_availability(df):
    # Count non-empty translations for each row
    translation_counts = df.drop('english', axis=1).notna().sum(axis=1)
    logger.debug("Calculated translation availability for each row.")

    # Sort the DataFrame based on the translation counts
    sorted_df = df.loc[translation_counts.sort_values(ascending=False).index]
    logger.info("Sorted DataFrame based on translation availability.")

    # Add a column showing the number of available translations
    sorted_df['translation_count'] = translation_counts[sorted_df.index]
    logger.debug("Added 'translation_count' column to DataFrame.")

    # Reorder columns to put 'english' and 'translation_count' first
    columns = ['english', 'translation_count'] + [col for col in sorted_df.columns if col not in ['english', 'translation_count']]
    sorted_df = sorted_df[columns]
    logger.info("Reordered DataFrame columns.")

    return sorted_df

# Sort the DataFrame
sorted_df = sort_rows_by_translation_availability(df)

# Create the languages folder if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Created output directory: {OUTPUT_DIR}")

# Save each language to its own CSV file
for column in sorted_df.columns:
    if column not in ['english', 'translation_count']:
        language_df = sorted_df[['english', column]].dropna(subset=[column])
        # Rename the non-English column to 'sentence'
        language_df = language_df.rename(columns={column: 'sentence'})
        output_file = os.path.join(OUTPUT_DIR, f"{column.replace(' ', '_')}.csv")
        
        try:
            language_df.to_csv(output_file, index=False, header=True)
            logger.info(f"Saved {len(language_df)} sentences for {column} to {output_file}.")
        except Exception as e:
            logger.error(f"Failed to save {column} sentences to {output_file}. Error: {e}")

logger.info("All language files have been saved.")
