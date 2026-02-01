import argparse
import random
import json
import os

import datasets
from tqdm import tqdm
from shutil import copyfile
from typing import Dict, Any
from pathlib import Path


import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup

import logging

logger = logging.getLogger(__name__)


def add_model_specific_args(parser):
    parser.add_argument("--name", type=str, default='samsum', help="Name of expt")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of dataloader workers")
    parser.add_argument("--max_output_len", type=int, default=128, help="Max output wordpieces")
    parser.add_argument("--limit_val_batches", default=1.0, type=float)
    parser.add_argument("--limit_test_batches", default=1.0, type=float)
    parser.add_argument("--limit_train_batches", default=1.0, type=float)

    # ✅ 本地模型路径
    parser.add_argument("--transformer_model", type=str, default='bart-base', help="Local model path")

    # ✅ 数据和输出路径
    parser.add_argument("--data_dir", type=str, default='dataset/samsum', help="Input data")
    parser.add_argument("--output_dir", type=str, default='output_dir/samsum/samsum_bart-base',
                        help="Output model save path")

    parser.add_argument("--predict_file_path", type=str,
                        default='dataset/DIALOGUE/Mask_phrase/run/test.jsonl',
                        help="Path for prediction data")
    parser.add_argument("--resume_checkpoint_dir", type=str, default="None")
    parser.add_argument("--resume_checkpoint_file", type=str, default="None")

    # ✅ 启用训练，禁用预测
    parser.add_argument("--do_train", default=True, type=bool, help="Enable training")
    parser.add_argument("--do_predict", default=False, type=bool)
    parser.add_argument("--val_every", type=float, default=1.0,
                        help="Validation check interval (fraction of epoch or steps)")

    # ✅ 模型相关参数
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--fp16", action='store_true', help="Enable mixed precision training (use 16-bit floats)")

    # ⚠️ 不要在 CPU 上开启 fp16
    # parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--grad_ckpt", action='store_true')

    return parser


SPECIAL_TOKEN_LIST = ["[MASK]"]

class SummarizationDataset(Dataset):
    def __init__(self, hf_arxiv_dataset, tokenizer, args):
        self.hf_arxiv_dataset = hf_arxiv_dataset
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(SPECIAL_TOKEN_LIST)
        self.args = args

    def __len__(self):
        return len(self.hf_arxiv_dataset)

    def __getitem__(self, idx):
        entry = self.hf_arxiv_dataset[idx]
        source = entry["masked_sent"] + " <SEP> " + entry["source"]
        target = entry["target"]

        input_ids = self.tokenizer.encode(source, truncation=True, max_length=self.args.max_input_len,
                                          padding='max_length')

        output_ids = self.tokenizer.encode(target, truncation=True, max_length=self.args.max_output_len,
                                           padding='max_length')

        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        pad_token_id = 1
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids

    def process_example(self, entry):
        source = entry["masked_sent"] + " <SEP> " + " ".join(entry['source_article_sentences'])

        target = entry["target"]

        input_ids = self.tokenizer.encode(source, truncation=True, max_length=self.args.max_input_len,
                                          padding='max_length')

        output_ids = self.tokenizer.encode(target, truncation=True, max_length=self.args.max_output_len,
                                           padding='max_length')

        return torch.tensor(input_ids), torch.tensor(output_ids)


class Summarizer(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params

        # ✅ 提前加载 tokenizer 并扩展特殊 token
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.transformer_model, use_fast=True)
        self.tokenizer.add_tokens(SPECIAL_TOKEN_LIST)

        # ✅ 加载 config
        config = AutoConfig.from_pretrained(self.args.transformer_model)
        config.gradient_checkpointing = self.args.grad_ckpt

        # ✅ 初始化模型
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.transformer_model, config=config)
        self.model.resize_token_embeddings(len(self.tokenizer))  # ⬅️ 调整词表大小到 50266

        # ✅ 加载 checkpoint（如果有）
        if self.args.resume_checkpoint_dir != "None":
            saved_model = torch.load(os.path.join(self.args.resume_checkpoint_dir, self.args.resume_checkpoint_file),
                                     map_location="cpu")
            renamed_state_dict = {}
            for k, v in saved_model["state_dict"].items():
                new_key = k.replace("model.model.", "model.")
                renamed_state_dict[new_key] = v

            # ✅ 安全加载权重，保留已有结构和词表
            missing_keys, unexpected_keys = self.model.load_state_dict(renamed_state_dict, strict=False)
            if missing_keys:
                print("⚠️ Missing keys:", missing_keys)
            if unexpected_keys:
                print("⚠️ Unexpected keys:", unexpected_keys)

        self.validation_step_outputs = []
        self.test_outputs = []
        self.rouge = datasets.load_metric('rouge')

    def forward(self, input_ids, output_ids):
        return self.model(input_ids,
                          attention_mask=(input_ids != self.tokenizer.pad_token_id),
                          labels=output_ids, use_cache=False)

    def training_step(self, batch, batch_nb):
        outputs = self.forward(*batch)
        epoch_num = self.current_epoch + 1
        self.log(f'train/train_step', batch_nb * epoch_num,
                 on_step=True, on_epoch=True)
        self.log('train/train_loss', outputs.loss, on_epoch=True)
        return {'loss': outputs.loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        dataset_size = len(self.hf_dataset['train'])

        # 防止除以 0，保证至少为 1（即使没有 GPU）
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            gpu_count = 1

        num_steps = dataset_size * self.args.epochs / gpu_count / self.args.grad_accum / self.args.batch_size

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup,
            num_training_steps=int(num_steps),  # 防止浮点数，转整数
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, split_name, is_train):
        dataset_split = self.hf_dataset[split_name]
        dataset = SummarizationDataset(hf_arxiv_dataset=dataset_split, tokenizer=self.tokenizer, args=self.args)

        # 判断是否使用分布式训练
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
            shuffle = False  # 使用分布式采样时不要再打乱
        else:
            sampler = None
            shuffle = is_train

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            sampler=sampler,
            collate_fn=SummarizationDataset.collate_fn,
        )

    def train_dataloader(self):
        return self._get_dataloader('train', is_train=True)

    def val_dataloader(self):
        return self._get_dataloader('validation', is_train=False)

    def test_dataloader(self):
        dl = self._get_dataloader('test', is_train=False)
        return dl

    def _evaluation_step(self, split, batch):
        input_ids, output_ids = batch
        generated_ids = self.model.generate(input_ids=input_ids,
                                            attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                            use_cache=True, max_length=self.args.max_output_len, num_beams=1)

        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        print("预测:", predictions[0])
        print("目标:", references[0])
        # Compute rouge
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        results = self.rouge.compute(predictions=predictions, references=references)
        metrics = {}
        for metric_name in metric_names:
            metric_val = input_ids.new_zeros(1) + results[metric_name].mid.fmeasure
            metrics[f'{split}_{metric_name}'] = metric_val
        return metrics

    def _evaluation_epoch_end(self, split, step_outputs):
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        aggregated_metrics = {}
        for metric_name in metric_names:
            aggregated_metrics[f'{split}_{metric_name}'] = []

        for pred in step_outputs:
            for key, value in pred.items():
                aggregated_metrics[key].append(value)

        for key, value in aggregated_metrics.items():
            aggregated_metrics[key] = torch.mean(torch.stack(value, dim=0), dim=0, keepdim=False)
            self.log(f'{split}/{key}_epoch', aggregated_metrics[key], on_step=False, on_epoch=True, prog_bar=True,
                     sync_dist=True)
        return aggregated_metrics

    def validation_step(self, batch, batch_idx):
        output = self._evaluation_step('val', batch)
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        fp = open(self.args.output_dir + "/val_metrics.txt", "a+")
        aggregated_metrics = self._evaluation_epoch_end('val', self.validation_step_outputs)
        for key, value in aggregated_metrics.items():
            fp.write(f'{key}_epoch: {value}\n')
        fp.write("\n")
        fp.close()
        self.validation_step_outputs.clear()

    def _test_evaluation_step(self, split, batch):
        input_ids, output_ids = batch
        generated_ids = self.model.generate(input_ids=input_ids,
                                            attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                            use_cache=True, max_length=self.args.max_output_len, num_beams=1)

        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)

        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        results = self.rouge.compute(predictions=predictions, references=references)
        metrics = {}
        for metric_name in metric_names:
            metric_val = input_ids.new_zeros(1) + results[metric_name].mid.fmeasure
            metrics[f'{split}_{metric_name}'] = metric_val
        return metrics

    def test_step(self, batch, batch_idx):
        output = self._evaluation_step('test', batch)
        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        aggregated_metrics = self._evaluation_epoch_end('test', self.test_outputs)

        # 将 tensor 转成普通数字，整理为 dict
        metrics_to_write = {key: value.cpu().item() for key, value in aggregated_metrics.items()}

        # 写入 JSON 文件
        output_file = os.path.join(self.args.output_dir, "metrics.json")
        with open(output_file, "w") as fp:
            json.dump(metrics_to_write, fp, indent=2)

        # 清空缓存，防止影响后续 epoch
        self.test_outputs.clear()

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = Path(self.args.output_dir + "/hf_checkpoints/").joinpath(
            f"best_tfmr_step={self.trainer.global_step}")
        save_path.mkdir(exist_ok=True, parents=True)
        self.model.config.save_step = self.trainer.global_step
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = add_model_specific_args(main_arg_parser)
    args = parser.parse_args()

    set_seed(args.seed)

    summarizer = Summarizer(args)


    # 构建 Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val/val_rougeLsum_epoch',
        dirpath=args.output_dir,
        filename='tw-{epoch:02d}-{step}-val_rougeLsum_epoch{val/val_rougeLsum_epoch:.4f}',
        save_top_k=3,
        mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        devices=1,
        accelerator="cpu",
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        precision=16 if args.fp16 else 32,
        accumulate_grad_batches=args.grad_accum,
        limit_val_batches=args.limit_val_batches,
        limit_train_batches=args.limit_train_batches,
        limit_test_batches=args.limit_test_batches,
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=args.val_every,
    )

    # ========== 训练流程 ==========
    if args.do_train:
        import pandas as pd
        from datasets import Dataset, DatasetDict

        train_file = os.path.join(args.data_dir, "1.jsonl")
        val_file = os.path.join(args.data_dir, "infill_valid.jsonl")
        test_file = os.path.join(args.data_dir, "infill_test.jsonl")

        train_dataset = Dataset.from_pandas(pd.read_json(train_file, lines=True))
        val_dataset = Dataset.from_pandas(pd.read_json(val_file, lines=True))
        test_dataset = Dataset.from_pandas(pd.read_json(test_file, lines=True))

        summarizer.hf_dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })

        # 开始训练
        ckpt_path = None
        if args.resume_checkpoint_dir and args.resume_checkpoint_dir != "None":
            if args.resume_checkpoint_file and args.resume_checkpoint_file != "None":
                ckpt_path = os.path.join(args.resume_checkpoint_dir, args.resume_checkpoint_file)

        trainer.fit(summarizer, ckpt_path=ckpt_path)

        # 训练完保存 best 模型
        best_ckpt_path = os.path.join(args.output_dir, "best.ckpt")
        copyfile(checkpoint_callback.best_model_path, best_ckpt_path)

        # 测试
        result = trainer.test(summarizer, ckpt_path=best_ckpt_path)
        print(result)

    if args.do_predict:
        print("Test on reference")

        import jsonlines
        from datasets import Dataset

        data = []
        with jsonlines.open(args.predict_file_path, mode='r') as reader:
            for obj in reader:
                data.append(obj)

        dataset_split = Dataset.from_list(data)

        orig_test_dataset = SummarizationDataset(hf_arxiv_dataset=dataset_split, tokenizer=summarizer.tokenizer, args=summarizer.args)
        output_test_preds_file = os.path.join(args.output_dir+"/mask_cands/", args.predict_file_path.split("/")[-1])

        if torch.cuda.is_available():
            summarizer.model = summarizer.model.to(device=torch.device('cuda'))
        with open(output_test_preds_file, "w") as writer:
            for idx, entry in tqdm(enumerate(dataset_split)):
                if idx % 100 == 0:
                    print("Processed "+str(idx)+" samples")
                input_ids, output_ids = orig_test_dataset.process_example(entry)
                input_ids = input_ids.unsqueeze(dim=0)

                input_ids = input_ids.to(summarizer.model.device)
                outputs = summarizer.model.generate(input_ids=input_ids,
                                                    attention_mask=(input_ids != summarizer.tokenizer.pad_token_id),
                                                    use_cache=False, max_length=args.max_output_len, num_beams=15, num_return_sequences=15,
                                                    return_dict_in_generate=True, output_hidden_states=True, early_stopping=True)
                generated_ids = outputs["sequences"]

                predictions = summarizer.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
                target_words = entry["target"].split()
                cand_preds = [x for x in predictions[5:] if not (x in entry["target"] or entry["target"] in x or len(list(set(entry["target"].split()) & set(x.split())))>=min(len(target_words)/2, len(x.split())/2) )]
                for pid, pred in enumerate(cand_preds):
                    replaced_summary_sent = entry["masked_sent"].replace("[MASK]", pred)
                    label = 0
                    err_span = pred
                    corr_span = entry["target"]
                    err_type = "Mask Model Cand"
                    error_prob = random.random()
                    if error_prob < 0.4:
                        replaced_summary_sent = entry["original_sent"]
                        label  = 1
                        err_type = "None"
                        err_span = "None"
                        corr_span = "None"

                    error_summary_sents = []
                    chosen_sent_idx = None
                    for sid, x in enumerate(entry["original_summary_sentences"]):
                        if x == entry["original_sent"]:
                            error_summary_sents.append(replaced_summary_sent)
                            chosen_sent_idx = sid
                        else:
                            error_summary_sents.append(x)
                    data = {"source_article_sentences": entry["source_article_sentences"],  # str
                            "original_summary_sentences": entry["original_summary_sentences"],  # List
                            "generated_summary_sentences": error_summary_sents,  # List
                            "incorrect_sent_idx": chosen_sent_idx,
                            "original_summary_sent": entry["original_sent"],
                            "generated_summary_sent": replaced_summary_sent,
                            "label": label,
                            "error_type": err_type,
                            "err_span": err_span,
                            "corr_span": corr_span}
                    writer.write(json.dumps(data) + "\n")
