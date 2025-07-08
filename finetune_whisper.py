"""Here is my code I want to fine tune the model on both translation and transcription simultaneously. edit the code as necessary I am using a dataset called razhan/DOLMA-speech it can be loaded as the following load_dataset("razhan/DOLMA-speech", "hawrami", split="train") it has the following columns id,file_name,sentence,english,gender,language,original_full_path,duration,speaker_id

No I want you to process each sample twice once for transcription once for translation in prepare dataset"""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np

import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from klpt.preprocess import Preprocess

preprocessor_ckb = Preprocess("Sorani", "Arabic", numeral="Latin")
normalizer = BasicTextNormalizer()

def preprocess_func(text):
    text = preprocessor_ckb.unify_numerals(text)
    text = normalizer(text)
    return text


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={"help": "Deprecated. Please use the `language` and `task` arguments instead."},
    )
    suppress_tokens: List[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated. The use of `suppress_tokens` should not be required for the majority of fine-tuning examples."
                "Should you need to use `suppress_tokens`, please manually update them in the fine-tuning script directly."
            )
        },
    )
    apply_spec_augment: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    transcription_column_name: str = field(
        default="sentence",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    translation_column_name: str = field(
        default="english",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]  
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()



    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset
    raw_datasets = DatasetDict()

    # List of languages and their corresponding codes
    languages = {
        "hawrami": "fa",
        "gilaki": "de", 
        "zazaki": "es",
        "mazanderani": "it",
        "laki_kurdish": "fr",
        "southern_kurdish": "nl",
        "talysh": "pt",
    }

    # Load and combine datasets for all languages
    train_datasets = []
    eval_datasets = []
    
    for lang_name, lang_code in languages.items():
        # Load train split
        train_dataset = load_dataset("razhan/DOLMA-speech", lang_name, split="train")
        # Remove id column and add it back with consistent type
        columns_to_remove = ["id"]
        train_dataset = train_dataset.remove_columns(columns_to_remove)
        # Add row indices as new id column
        train_dataset = train_dataset.add_column("id", list(range(len(train_dataset))))
        # Update language code
        train_dataset = train_dataset.map(lambda x: {"language": lang_code})
        
        # Apply max_train_samples if specified
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        
        train_datasets.append(train_dataset)

        if training_args.do_eval:
            # Load test split
            eval_dataset = load_dataset("razhan/DOLMA-speech", lang_name, split="test")
            # Remove id column and add it back with consistent type
            eval_dataset = eval_dataset.remove_columns(columns_to_remove)
            # Add row indices as new id column
            eval_dataset = eval_dataset.add_column("id", list(range(len(eval_dataset))))
            # Update language code
            eval_dataset = eval_dataset.map(lambda x: {"language": lang_code})
            
            # Apply max_eval_samples if specified
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
            
            eval_datasets.append(eval_dataset)

    # Combine all language datasets
    raw_datasets["train"] = datasets.concatenate_datasets(train_datasets)
    if training_args.do_eval:
        raw_datasets["eval"] = datasets.concatenate_datasets(eval_datasets)

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": model_args.apply_spec_augment})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if model_args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # We only need to set the language and task ids in a multilingual setting
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)
        model.generation_config.language = data_args.language
        model.generation_config.task = data_args.task
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )

    # TODO (Sanchit): deprecate these arguments in v4.41
    if model_args.forced_decoder_ids is not None:
        logger.warning(
            "The use of `forced_decoder_ids` is deprecated and will be removed in v4.41."
            "Please use the `language` and `task` arguments instead"
        )
        model.generation_config.forced_decoder_ids = model_args.forced_decoder_ids
    else:
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None

    if model_args.suppress_tokens is not None:
        logger.warning(
            "The use of `suppress_tokens` is deprecated and will be removed in v4.41."
            "Should you need `suppress_tokens`, please manually set them in the fine-tuning script."
        )
        model.generation_config.suppress_tokens = model_args.suppress_tokens

    # 6. Resample speech dataset if necessary
    
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    transcription_column_name = data_args.transcription_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    def prepare_dataset_translation(batch):
        """Process dataset for both transcription and translation"""
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=forward_attention_mask
        )
        
        # process audio length
        batch["input_features"] = inputs.get("input_features")[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets for translation with language-specific prefix
        tokenizer.set_prefix_tokens(language=batch["language"], task="translate")
        input_str_translate = batch["english"].lower() 
        batch["labels_translate"] = tokenizer(input_str_translate).input_ids
        
        # process targets for transcription with language-specific prefix
        tokenizer.set_prefix_tokens(language=batch["language"], task="transcribe")
        input_str_transcribe = preprocess_func(batch[transcription_column_name])
        batch["labels_transcribe"] = tokenizer(input_str_transcribe).input_ids


        return batch
    

    
    # def prepare_dataset(batch):
    #     # process audio
    #     sample = batch[audio_column_name]
    #     inputs = feature_extractor(
    #         sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=forward_attention_mask
    #     )
    #     # process audio length
    #     batch[model_input_name] = inputs.get(model_input_name)[0]
    #     batch["input_length"] = len(sample["array"])
    #     if forward_attention_mask:
    #         batch["attention_mask"] = inputs.get("attention_mask")[0]

    #     # process targets
    #     input_str = preprocess_func(batch[transcription_column_name])

    #     tokenizer.set_prefix_tokens(language=batch["language"], task="transcribe") 
    #     batch["labels"] = tokenizer(input_str).input_ids
    #     return batch

    def flatten_dataset(batch):
        """Flatten the dataset to combine transcription and translation samples efficiently"""
        batch_size = len(batch["input_features"])
        
        # Create lists for both transcription and translation samples
        combined_features = []
        combined_masks = []
        combined_labels = []
        combined_tasks = []
        combined_lengths = []

        # Pre-allocate lists with total size
        combined_features = [None] * (batch_size * 2)
        combined_masks = [None] * (batch_size * 2) if "attention_mask" in batch else None
        combined_labels = [None] * (batch_size * 2)
        combined_tasks = ["transcribe"] * batch_size + ["translate"] * batch_size
        combined_lengths = [None] * (batch_size * 2)

        # Fill lists efficiently
        for i in range(batch_size):
            # Transcription sample
            combined_features[i] = batch["input_features"][i]
            combined_labels[i] = batch["labels_transcribe"][i]
            combined_lengths[i] = batch["input_length"][i]
            
            # Translation sample
            combined_features[i + batch_size] = batch["input_features"][i]
            combined_labels[i + batch_size] = batch["labels_translate"][i]
            combined_lengths[i + batch_size] = batch["input_length"][i]
            
            # Handle attention masks if present
            if combined_masks is not None:
                mask = batch["attention_mask"][i]
                combined_masks[i] = mask
                combined_masks[i + batch_size] = mask

        result = {
            "input_features": combined_features,
            "labels": combined_labels,
            "task": combined_tasks,
            "input_length": combined_lengths
        }

        if combined_masks is not None:
            result["attention_mask"] = combined_masks

        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        # Map and process the dataset
        vectorized_datasets = raw_datasets.map(
            prepare_dataset_translation,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="preprocess dataset",
        )
        
        # Flatten the dataset to combine transcription and translation samples
        flattened_datasets = DatasetDict()
        for split, dataset in vectorized_datasets.items():
            flattened_datasets[split] = dataset.map(
                flatten_dataset, 
                batched=True, 
                remove_columns=dataset.column_names,
                desc=f"flatten {split} dataset"
            )

        # filter data that is shorter than min_input_length or longer than
        # max_input_length
        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length



        # Apply filter to flattened_datasets
        filtered_datasets = DatasetDict()
        for split, dataset in flattened_datasets.items():
            filtered_datasets[split] = dataset.filter(
                is_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["input_length"],
            )

        # Update vectorized_datasets to use the filtered version
        vectorized_datasets = filtered_datasets


    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    # 8. Load Metric
    wer_metric = evaluate.load("wer", cache_dir=model_args.cache_dir)
    cer_metric = evaluate.load("cer", cache_dir=model_args.cache_dir)
    bleu_metric = evaluate.load("sacrebleu", cache_dir=model_args.cache_dir)
    chrf_metric = evaluate.load("chrf", cache_dir=model_args.cache_dir)


    # These needs to be modified if we finetune for large v3 and turbo
    language_ids = {
        "fa": 50300,
        "de": 50261, 
        "es": 50262,
        "it": 50274,
        "fr": 50265,
        "nl": 50271,
        "pt": 50267,
    }

    task_to_id =  {
        "transcribe": 50359,
        "translate": 50358
    }


    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        
        # Replace padding tokens with pad_token_id
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        
        # Get language IDs and task IDs from the first and second tokens of each sequence
        sequence_lang_ids = pred.label_ids[:, 0]  # First token of each sequence
        sequence_task_ids = pred.label_ids[:, 1]  # Second token of each sequence (task)
        
        # Create reverse mappings for language codes and tasks
        lang_id_to_code = {token_id: code for code, token_id in language_ids.items()}
        code_to_name = {code: name for name, code in languages.items()}
        task_id_to_name = {v: k for k, v in task_to_id.items()}
        
        # Decode predictions and labels
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_str = [preprocess_func(s) for s in pred_str]
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        label_str = [preprocess_func(s) for s in label_str]
        
        # Calculate metrics per language and task
        metrics = {}
        for lang_token_id in set(sequence_lang_ids):
            if lang_token_id in lang_id_to_code:
                lang_code = lang_id_to_code[lang_token_id]
                lang_name = code_to_name[lang_code]
                
                for task_token_id in set(sequence_task_ids):
                    if task_token_id in task_id_to_name:
                        task_name = task_id_to_name[task_token_id]
                        
                        # Create mask for this language and task combination
                        mask = (sequence_lang_ids == lang_token_id) & (sequence_task_ids == task_token_id)
                        
                        # Get predictions and references for this language and task
                        task_preds = [pred_str[i] for i, m in enumerate(mask) if m]
                        task_refs = [label_str[i] for i, m in enumerate(mask) if m]
                        
                        if task_preds:  # Only compute if we have predictions
                            if task_name == "transcribe":
                                # For transcription, compute WER and CER
                                wer = wer_metric.compute(predictions=task_preds, references=task_refs)
                                cer = cer_metric.compute(predictions=task_preds, references=task_refs)
                                metrics[f"{lang_name}_wer"] = wer
                                metrics[f"{lang_name}_cer"] = cer
                            else:  # task_name == "translate"
                                # For translation, compute BLEU and chrF scores
                                bleu = bleu_metric.compute(predictions=task_preds, references=[[r] for r in task_refs])
                                chrf = chrf_metric.compute(predictions=task_preds, references=task_refs)
                                metrics[f"{lang_name}_bleu"] = bleu["score"]
                                metrics[f"{lang_name}_chrf"] = chrf["score"]
        
        # Calculate average metrics per task
        transcribe_wer = [v for k, v in metrics.items() if "transcribe_wer" in k]
        transcribe_cer = [v for k, v in metrics.items() if "transcribe_cer" in k]
        translate_bleu = [v for k, v in metrics.items() if "translate_bleu" in k]
        translate_chrf = [v for k, v in metrics.items() if "translate_chrf" in k]
        
        if transcribe_wer:
            metrics["avg_transcribe_wer"] = np.mean(transcribe_wer)
        if transcribe_cer:
            metrics["avg_transcribe_cer"] = np.mean(transcribe_cer)
        if translate_bleu:
            metrics["avg_translate_bleu"] = np.mean(translate_bleu)
        if translate_chrf:
            metrics["avg_translate_chrf"] = np.mean(translate_chrf)
        
        return metrics

    # 9. Create a single speech processor
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    # 11. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        processing_class=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # 12. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 13. Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 14. Write Training Stats
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "automatic-speech-recognition"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()