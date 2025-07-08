# Automatic Speech Recognition for Low-Resourced Middle Eastern Languages

<div align="center">
  
  <h3>📄 Paper: <a href="[PAPER_LINK_PLACEHOLDER]">Interspeech 2025</a></h3>
  
  <p>
    <a href="https://huggingface.co/datasets/razhan/DOLMA-speech">
      <img src="https://img.shields.io/badge/🤗%20Dataset-DOLMA--speech-blue" alt="DOLMA-speech Dataset">
    </a>
    <a href="https://huggingface.co/datasets/razhan/hawrami_speech">
      <img src="https://img.shields.io/badge/🤗%20Dataset-Hawrami%20Studio-green" alt="Hawrami Studio Dataset">
    </a>
    <a href="https://huggingface.co/collections/razhan/dolma-asr-models-686d7c2f95e8b3d776ec2d31">
      <img src="https://img.shields.io/badge/🤗%20Models-DOLMA%20ASR%20Collection-purple" alt="All Trained Models Collection">
    </a>
  </p>
  
</div>

## Abstract

Despite significant advancements in language and speech technologies, many languages in the Middle East remain underserved, leading to a technological disparity that negatively impacts these languages. This paper presents a pioneering effort to address this issue by focusing on speech technologies for low-resourced languages in the Middle East. We introduce a community-driven volunteer-based initiative to collect audio recordings for six languages spoken by an estimated population of 30 million speakers. Through this initiative, we collect over 40 hours of speech data, with 75% of utterances based on multilingual parallel corpora. In our experiments, we demonstrate the impact of data collection and fine-tuning models on the performance of speech technologies for these languages. This research serves as a crucial step towards preserving and promoting linguistic diversity in the Middle East while ensuring equal access to speech technologies for all language communities.

## Overview

This repository contains the code for collecting speech data and fine-tuning Whisper models for six low-resourced Middle Eastern languages: **Gilaki**, **Laki Kurdish**, **Hawrami**, **Mazandarani**, **Southern Kurdish**, and **Zazaki**.

### Key Features
- 🎙️ **Community-driven data collection** via Telegram bot
- 🎯 **40+ hours of speech data** across 6 languages
- 🌐 **Multilingual parallel corpora** with English translations
- 🚀 **Fine-tuned Whisper models** for ASR
- 💬 **Applications**: ASR, speech-to-speech translation, and more

## Data Collection

### Telegram Bot (`tg_bot/`)

The Telegram bot facilitates community-driven audio data collection with quality control mechanisms.

#### 1. Data Preparation (`prepare_data.py`)
Processes multilingual parallel corpora and creates language-specific CSV files:
```bash
cd tg_bot
python prepare_data.py
```
This script:
- Reads `common_sentences.csv` containing parallel sentences
- Sorts sentences by translation availability
- Creates individual CSV files for each language in `languages/`

#### 2. Bot Deployment (`bot.py`)
The main bot implementation with features:
- **User onboarding**: Gender and language selection
- **Recording workflow**: Present sentences, record audio, review & confirm
- **Quality control**: Automatic duration checks, manual review options
- **Progress tracking**: Skip problematic texts, track recorded sentences

To run the bot:
```bash
# Set up environment variables
export TOKEN_ID="your_telegram_bot_token"
export SEND_TO_CHANNEL="true"  # Optional
export CHANNEL_ID="your_channel_id"  # Optional

python bot.py
```

#### 3. Data Analysis (`report.py`)
Generate comprehensive statistics about collected data:
```bash
python report.py
```
Provides:
- Per-user contribution statistics
- Language-wise data distribution
- Gender demographics
- Content and duration analytics

### Dataset Structure

The collected data includes:
- **Audio files**: MP3 format recordings
- **Metadata**: CSV files with sentence text, English translations, speaker info
- **English TTS**: Kokoro-generated speech for all English sentences

## Model Fine-tuning

### Whisper Fine-tuning (`finetune_whisper.py`)

Fine-tune OpenAI's Whisper models for both **monolingual** and **multilingual** ASR:

#### Configuration
Edit `train.sh` to customize training parameters:
```bash
python finetune_whisper.py \
    --model_name_or_path="openai/whisper-base" \
    --language="persian" \
    --num_train_epochs="3" \
    --output_dir="./whisper-base-me" \
    --per_device_train_batch_size="32" \
    --learning_rate="1e-5" \
    --do_train \
    --do_eval
```

#### Training Modes
1. **Monolingual**: Train separate models for each language
2. **Multilingual**: Train a single model for all languages

The script automatically:
- Loads data from `razhan/DOLMA-speech` dataset
- Processes audio and text pairs
- Handles both transcription and translation tasks
- Applies text normalization and preprocessing

## Results

Our experiments show:
- **Monolingual models** achieve 29-31% WER reduction
- **Best performance**: Hawrami (37.9% WER)
- **Significant improvements** for Arabic-script languages

## Citation

```bibtex
@inproceedings{hameed2025asr,
  title={Automatic Speech Recognition for Low-Resourced Middle Eastern Languages},
  author={Hameed, Razhan and Ahmadi, Sina and Hadi, Hanah and Sennrich, Rico},
  booktitle={Interspeech 2025},
  year={2025}
}
```

## Acknowledgments

This work was supported by the Swiss National Science Foundation (MUTAMUR project) and Stanford SILICON. Special thanks to all community volunteers who contributed recordings.
