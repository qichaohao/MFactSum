import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import torch
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from typing import Optional, List, Dict, Union, Callable, Tuple, Any
from filelock import FileLock
from datasets import load_dataset, DatasetDict
import transformers
from rouge_score import rouge_scorer, scoring
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from modeling_bart import BartForConditionalGeneration

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from trainer import CustomTrainer
os.environ["WANDB_MODE"] = "disabled"
check_min_version("4.30.0")  # current version: 4.30.2

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    resize_position_embeddings: Optional[bool] = field(default=None)

@dataclass
class DataTrainingArguments:
    lang: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    text_column: Optional[str] = field(default=None)
    summary_column: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_source_length: Optional[int] = field(default=1024)
    max_target_length: Optional[int] = field(default=128)
    val_max_target_length: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=False,)
    max_train_samples: Optional[int] = field(default=None,)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)
    moe_use_role_aware: bool = field(default=False)
    num_beams: Optional[int] = field(default=5, metadata={"help": "Number of beams for beam search. Default 5 for better quality."})
    ignore_pad_token_for_loss: bool = field(default=True)
    source_prefix: Optional[str] = field(default="")
    forced_bos_token: Optional[str] = field(default=None)
    do_contrastive_learning: bool = field(default=True)
    use_event_factuality: bool = field(default=True)
    contrastive_loss_weight: float = field(default=0.1)
    contrastive_margin: float = field(default=0.15)
    num_contrastive_negatives: int = field(default=3)
    use_expert_mode: bool = field(default=False)
    num_experts: int = field(default=8)
    moe_top_k: int = field(default=2)
    moe_temperature: float = field(default=1.0)
    moe_noise_std: float = field(default=0.1)
    moe_aux_loss_weight: float = field(default=0.01)
    num_moe_layers: int = field(default=1)


    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        
        if self.num_contrastive_negatives < 1 or self.num_contrastive_negatives > 6:
            raise ValueError(f"num_contrastive_negatives must be between 1 and 6, got {self.num_contrastive_negatives}")
        
        if self.use_expert_mode:
            if self.num_experts < 2 or self.num_experts > 256:
                raise ValueError(f"num_experts must be between 2 and 16, got {self.num_experts}")
            
            if self.moe_top_k < 1 or self.moe_top_k > min(4, self.num_experts):
                raise ValueError(f"moe_top_k must be between 1 and min(4, num_experts={self.num_experts}), got {self.moe_top_k}")
            
            if self.moe_temperature <= 0:
                raise ValueError(f"moe_temperature must be positive, got {self.moe_temperature}")
            
            if self.moe_noise_std < 0 or self.moe_noise_std > 1.0:
                raise ValueError(f"moe_noise_std must be between 0 and 1.0, got {self.moe_noise_std}")
            
            if self.moe_aux_loss_weight < 0 or self.moe_aux_loss_weight > 1.0:
                raise ValueError(f"moe_aux_loss_weight must be between 0 and 1.0, got {self.moe_aux_loss_weight}")


summarization_name_mapping = {
    "samsum": ("dialogue", "summary"),
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    #logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in ["t5-small","t5-base","t5-large","t5-3b","t5-11b",]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
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
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        for split_name, path in data_files.items():
            print(f"Loading split: {split_name} from {path}")
            ds = load_dataset(
                extension,
                data_files={split_name: path},
                cache_dir=model_args.cache_dir,
            )[split_name]
            raw_datasets[split_name] = ds
        raw_datasets = DatasetDict(raw_datasets)
        all_columns = set()
        for split in raw_datasets:
            all_columns |= set(raw_datasets[split].column_names)

        for split in raw_datasets:
            ds = raw_datasets[split]
            for col in all_columns:
                if col not in ds.column_names:
                    ds = ds.add_column(col, [""] * len(ds))
            raw_datasets[split] = ds

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=None,  
        local_files_only=True  
    )

    if data_args.use_expert_mode:
        config.num_experts = data_args.num_experts
        config.moe_top_k = data_args.moe_top_k
        config.moe_temperature = data_args.moe_temperature
        config.moe_noise_std = data_args.moe_noise_std if training_args.do_train else 0.0
        config.num_moe_layers = data_args.num_moe_layers if data_args.num_moe_layers > 0 else config.encoder_layers
        config.moe_use_role_aware = data_args.moe_use_role_aware
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        use_expert_mode=data_args.use_expert_mode,
    )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets, all_negs, events = [], [], [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])
                negs = []
                neg_idx = 1
                while f'neg{neg_idx}' in examples and examples[f'neg{neg_idx}'][i]:
                    negs.append(examples[f'neg{neg_idx}'][i])
                    neg_idx += 1
                all_negs.append(negs)
                if 'events' in examples and examples['events'][i]:
                    event_texts = []
                    for event in examples['events'][i]:
                        if isinstance(event, dict):
                            event_parts = []
                            if 'subject' in event and event['subject']:
                                event_parts.append(event['subject'])
                            if 'action' in event and event['action']:
                                event_parts.append(event['action'])
                            if 'object' in event and event['object']:
                                event_parts.append(event['object'])
                            event_text = ' '.join(event_parts)
                            event_texts.append(event_text)
                        else:
                            event_texts.append(str(event))
                    events.append('; '.join(event_texts))
                else:
                    events.append('')
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        pos_encodings = tokenizer(targets, max_length=data_args.max_source_length, padding=padding, truncation=True)
        model_inputs['pos_input_ids'] = pos_encodings['input_ids']
        model_inputs['pos_attention_mask'] = pos_encodings['attention_mask']
        if all_negs and len(all_negs[0]) > 0:
            max_neg_count = max(len(negs) for negs in all_negs)
            for neg_idx in range(max_neg_count):
                neg_texts = [negs[neg_idx] if neg_idx < len(negs) else "" for negs in all_negs]
                neg_encodings = tokenizer(neg_texts, max_length=data_args.max_source_length, padding=padding, truncation=True)
                model_inputs[f'neg{neg_idx}_input_ids'] = neg_encodings['input_ids']
                model_inputs[f'neg{neg_idx}_attention_mask'] = neg_encodings['attention_mask']
        if events:
            event_encodings = tokenizer(events, max_length=data_args.max_source_length, padding=padding, truncation=True)
            model_inputs['event_input_ids'] = event_encodings['input_ids']
            model_inputs['event_attention_mask'] = event_encodings['attention_mask']
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
        def __init__(
                self,
                tokenizer,
                model=None,
                label_pad_token_id=-100,
                pad_to_multiple_of=None,
                padding=True,
                is_training=True,  
                use_event_factuality=False,
        ):
            super().__init__(
                tokenizer=tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=pad_to_multiple_of,
                padding=padding,
            )
            self.is_training = is_training
            self.use_event_factuality = use_event_factuality

        def __call__(self, features: List[Dict[str, List[int]]], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
            is_evaluation = os.environ.get('EVALUATION_PHASE', '0') == '1' or not self.is_training
            if not self.use_event_factuality:
                for feature in features:
                    for field in list(feature.keys()):
                        if 'event_' in field:
                            del feature[field]
            missing_fields = []
            required_fields = ['pos_input_ids', 'pos_attention_mask']
            if self.use_event_factuality:
                required_fields += ['event_input_ids', 'event_attention_mask']
            neg_idx = 0
            while f'neg{neg_idx}_input_ids' in features[0] or f'neg{neg_idx}_input_ids' not in features[0]:
                if neg_idx == 0 and f'neg{neg_idx}_input_ids' not in features[0]:
                    missing_fields.append(f'neg{neg_idx}_input_ids')
                    missing_fields.append(f'neg{neg_idx}_attention_mask')
                    break
                elif neg_idx > 0 and f'neg{neg_idx}_input_ids' not in features[0]:
                    break
                else:
                    required_fields.append(f'neg{neg_idx}_input_ids')
                    required_fields.append(f'neg{neg_idx}_attention_mask')
                    neg_idx += 1
            
            for field in required_fields:
                if field not in features[0]:
                    missing_fields.append(field)
            
            if missing_fields:
                has_neg_missing = any('neg' in field for field in missing_fields)
                if not is_evaluation or not has_neg_missing:
                    logger.warning(f"Missing fields in features: {missing_fields}")
                for feature in features:
                    for field in missing_fields:
                        if field.endswith('input_ids'):
                            feature[field] = [self.tokenizer.pad_token_id] * data_args.max_source_length
                        elif field.endswith('attention_mask'):
                            feature[field] = [0] * data_args.max_source_length
                        else:
                            feature[field] = [self.tokenizer.pad_token_id] * len(feature["input_ids"])
            max_lengths = {}
            for field in required_fields:
                if field in features[0]:
                    max_lengths[field] = max(len(feature[field]) for feature in features)
            for feature in features:
                for field in required_fields:
                    if field in feature and field in max_lengths:
                        max_len = max_lengths[field]
                        if len(feature[field]) < max_len:
                            if field.endswith('input_ids'):
                                feature[field] = feature[field] + [self.tokenizer.pad_token_id] * (max_len - len(feature[field]))
                            elif field.endswith('attention_mask'):
                                feature[field] = feature[field] + [0] * (max_len - len(feature[field]))
                        elif len(feature[field]) > max_len:
                            feature[field] = feature[field][:max_len]
            batch = super().__call__(features, return_tensors=return_tensors)
            for field in required_fields:
                if field in features[0]:
                    if field not in batch:
                        if field.endswith('input_ids'):
                            batch[field] = torch.tensor([feature[field] for feature in features], dtype=torch.long)
                        elif field.endswith('attention_mask'):
                            batch[field] = torch.tensor([feature[field] for feature in features], dtype=torch.long)
            return batch
    
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    is_training_phase = training_args.do_train and not (training_args.do_eval or training_args.do_predict)
    data_collator = CustomDataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        is_training=is_training_phase,
        use_event_factuality=data_args.use_event_factuality,
    )

    # Metric
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = scorer.score(pred, label)
            aggregator.add_scores(scores)
        result = aggregator.aggregate()
        return {
            "rouge1": round(result["rouge1"].mid.fmeasure * 100, 4),
            "rouge2": round(result["rouge2"].mid.fmeasure * 100, 4),
            "rougeL": round(result["rougeL"].mid.fmeasure * 100, 4),
            "gen_len": np.mean([np.count_nonzero(p != tokenizer.pad_token_id) for p in preds])
        }
    if training_args.generation_max_length is None:
        training_args.generation_max_length = (
            data_args.val_max_target_length if data_args.val_max_target_length is not None 
            else data_args.max_target_length
        )
    if data_args.num_beams is not None:
        training_args.generation_num_beams = data_args.num_beams
    if training_args.max_grad_norm == 1.0:  
        training_args.max_grad_norm = 0.5
        logger.info("Setting max_grad_norm to 0.5 for better training stability")
    else:
        logger.info(f"Using user-specified max_grad_norm: {training_args.max_grad_norm}")
    if training_args.load_best_model_at_end:
        training_args.metric_for_best_model = "rouge1"
        training_args.greater_is_better = True
        logger.info("Using ROUGE-1 for best model selection (greater_is_better=True)")

    if data_args.do_contrastive_learning:
        model.config.output_hidden_states = True
    
    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        test_dataset=predict_dataset if training_args.do_predict else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        data_args=data_args
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    return results


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
