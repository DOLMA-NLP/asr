import os
import pandas as pd
from datasets import load_dataset, DatasetDict

def process_and_upload_datasets(base_dir="dataset", repo_name="DOLMA-speech"):
    # Step 1: Collect all unique user_ids across all languages
    global_user_ids = set()
    languages = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for lang in languages:
        metadata_path = os.path.join(base_dir, lang, "metadata.csv")
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            global_user_ids.update(metadata['user_id'].unique())
    
    # Create a global user_id to speaker_id mapping
    user_to_speaker_id = {user_id: f"SPEAKER_{i:02d}" for i, user_id in enumerate(global_user_ids, start=1)}
    
    # Step 2: Process each language using the global user_to_speaker_id mapping
    for lang in languages:
        print(f"Processing {lang}...")
        metadata_path = os.path.join(base_dir, lang, "metadata.csv")
        
        # Skip if metadata.csv doesn't exist
        if not os.path.exists(metadata_path):
            print(f"No metadata found for {lang}. Skipping this language...")
            continue
        
        # Load metadata and clean up unlisted files
        metadata = pd.read_csv(metadata_path)
        expected_files = set(metadata['file_name'])

        # Delete .mp3 files not listed in the metadata
        lang_dir = os.path.join(base_dir, lang)
        # for mp3_file in os.listdir(lang_dir):
        #     if mp3_file.endswith(".mp3"):
        #         # Remove .mp3 extension for comparison
        #         base_name = os.path.splitext(mp3_file)[0]
        #         if base_name not in expected_files:
        #             os.remove(os.path.join(lang_dir, mp3_file))
        #             print(f"Deleted {mp3_file} from {lang} because it has no metadata entry.")
        
        # Function to map user_id to global speaker_id
        def map_user_id_to_speaker_id(examples):
            examples['speaker_id'] = [user_to_speaker_id[uid] for uid in examples['user_id']]
            return examples
        
        # Load and process the dataset
        ds = load_dataset("audiofolder", data_dir=lang_dir)
        ds = ds.map(map_user_id_to_speaker_id, batched=True)
        ds = ds.remove_columns('user_id')
        
        # Ensure dataset is a DatasetDict before uploading
        if not isinstance(ds, DatasetDict):
            print(f"Warning: Dataset for {lang} is not a DatasetDict. Skipping...")
            continue
        
        # Push to Hugging Face Hub
        try:
            ds.push_to_hub(f"{repo_name}", config_name=f"{lang}", private=True)
            print(f"Successfully uploaded {lang} dataset to {repo_name}/{lang}")
        except Exception as e:
            print(f"Error uploading {lang} dataset: {str(e)}")

    print("All datasets processed and uploaded.")

# Run the function
process_and_upload_datasets()
