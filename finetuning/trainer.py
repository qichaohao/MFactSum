import os
import math
import time
import json
import logging
import numpy as np
import random

from typing import Optional, List, Dict, Union, Callable, Tuple, Any
logger = logging.getLogger(__name__)
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback
from transformers.trainer_utils import speed_metrics
from apex import amp
from accelerate import Accelerator

def _get_neg_fields(inputs):
    neg_fields = []
    neg_idx = 0
    while f'neg{neg_idx}_input_ids' in inputs:
        neg_fields.append(f'neg{neg_idx}_input_ids')
        neg_fields.append(f'neg{neg_idx}_attention_mask')
        neg_idx += 1
    return neg_fields


def _prepare_inputs_for_generate(inputs, use_event_factuality=False):
    generate_inputs = inputs.copy()
    for k in ('pos_input_ids', 'pos_attention_mask'):
        if k in generate_inputs:
            del generate_inputs[k]
    for field in _get_neg_fields(generate_inputs):
        if field in generate_inputs:
            del generate_inputs[field]
    if not use_event_factuality:
        for k in ('event_input_ids', 'event_attention_mask'):
            if k in generate_inputs:
                del generate_inputs[k]
    return generate_inputs


class CustomTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        test_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        data_args: Optional[Any] = None,
        accelerator: Optional[Any] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.test_dataset = test_dataset
        self.data_args = data_args
        if accelerator is not None:
            self.accelerator = accelerator
        else:
            if getattr(self.args, "bf16", False):
                mp = "bf16"
            elif getattr(self.args, "fp16", False):
                mp = "fp16"
            else:
                mp = "no"
            try:
                self.accelerator = Accelerator(mixed_precision=mp)
            except Exception as e:
                logger.warning(f"Failed to create Accelerator automatically: {e}. Falling back to None (but this may break fp16 flows).")
                self.accelerator = None

        self.total_cross_entropy_loss = 0.0
        self.total_contrastive_loss = 0.0
        self.loss_count = 0

        # Override self.model.generation_config if a GenerationConfig is specified in args.
        if getattr(self.args, "generation_config", None) is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config
        if hasattr(self, 'data_args') and getattr(self.data_args, 'do_contrastive_learning', False):
            if hasattr(model, 'config') and hasattr(model.config, 'd_model'):
                hidden_size = model.config.d_model
            else:
                hidden_size = 768
            projection_dim = 256
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_size, projection_dim),
                nn.GELU(),
                nn.LayerNorm(projection_dim)
            )

            for module in self.projection_head.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    if module.bias is not None:
                        module.bias.data.zero_()

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        use_event_factuality = getattr(self.data_args, 'use_event_factuality', False)
        generate_inputs = _prepare_inputs_for_generate(inputs, use_event_factuality)
        return super().prediction_step(
            model=model,
            inputs=generate_inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys
        )

    def log(self, logs: Dict[str, float]) -> None:
        if hasattr(self, '_current_step_losses'):
            for key, value in self._current_step_losses.items():
                if key not in logs:
                    logs[key] = value

        if "cross_entropy_loss" not in logs:
            logs["cross_entropy_loss"] = 0.0
        if "contrastive_loss" not in logs:
            logs["contrastive_loss"] = 0.0

        super().log(logs)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        if hasattr(self.data_args, 'use_event_factuality'):
            inputs['use_event_factuality'] = self.data_args.use_event_factuality

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if getattr(self.args, "n_gpu", 1) > 1:
            loss = loss.mean()  # average on multi-gpu parallel training
        has_deepspeed = hasattr(self, 'deepspeed') and self.deepspeed is not None
        if getattr(self.args, 'gradient_accumulation_steps', 1) > 1 and not has_deepspeed:
            loss = loss / self.args.gradient_accumulation_steps
        try:
            cross_entropy_loss = loss.item()
        except Exception:
            cross_entropy_loss = float(loss.detach())
        contrastive_loss = 0.0
        contrastive_loss_value = 0.0
        
        if hasattr(self, 'data_args') and getattr(self.data_args, 'do_contrastive_learning', False):
            decoder_hidden = getattr(outputs, 'decoder_hidden_states', None)
            encoder_last_hidden = getattr(outputs, 'encoder_last_hidden_state', None)
            if decoder_hidden is not None and encoder_last_hidden is not None:
                decoder_last_hidden = decoder_hidden[-1]  # [batch_size, seq_len, hidden_size]
                with torch.cuda.amp.autocast(enabled=False):
                    decoder_hidden_fp32 = decoder_last_hidden.float()
                    if 'decoder_attention_mask' in inputs:
                        mask = inputs['decoder_attention_mask'].unsqueeze(-1).to(decoder_hidden_fp32.dtype)
                    else:
                        mask = torch.ones(decoder_hidden_fp32.size()[:-1], device=decoder_hidden_fp32.device).unsqueeze(-1).to(decoder_hidden_fp32.dtype)
                    denom = mask.sum(dim=1)
                    denom = denom + 1e-8
                    main_latent = (decoder_hidden_fp32 * mask).sum(dim=1) / denom
                pos_input_ids = inputs.get('pos_input_ids')
                pos_attention_mask = inputs.get('pos_attention_mask')
                if pos_input_ids is not None and pos_attention_mask is not None:
                    pos_encoder = model.get_encoder()
                    pos_encoder_outputs = pos_encoder(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
                    from modeling_bart import shift_tokens_right
                    pos_decoder_input_ids = shift_tokens_right(
                        pos_input_ids, 
                        model.config.pad_token_id, 
                        model.config.decoder_start_token_id
                    )
                    pos_decoder = model.get_decoder()
                    pos_decoder_outputs = pos_decoder(
                        input_ids=pos_decoder_input_ids,
                        attention_mask=None,  
                        encoder_hidden_states=pos_encoder_outputs.last_hidden_state,
                        encoder_attention_mask=pos_attention_mask,
                        output_hidden_states=True,
                    )
                    pos_decoder_last_hidden = pos_decoder_outputs.hidden_states[-1] if hasattr(pos_decoder_outputs, 'hidden_states') and pos_decoder_outputs.hidden_states else pos_decoder_outputs.last_hidden_state
                    with torch.cuda.amp.autocast(enabled=False):
                        pos_decoder_fp32 = pos_decoder_last_hidden.float()
                        pos_decoder_mask = (pos_decoder_input_ids != model.config.pad_token_id).unsqueeze(-1).to(pos_decoder_fp32.dtype)
                        pos_denom = pos_decoder_mask.sum(dim=1) + 1e-8
                        pos_latent = (pos_decoder_fp32 * pos_decoder_mask).sum(dim=1) / pos_denom
                    neg_latents = []
                    all_neg_inputs_list = []
                    all_neg_masks_list = []
                    neg_idx = 0
                    while f'neg{neg_idx}_input_ids' in inputs:
                        all_neg_inputs_list.append(inputs[f'neg{neg_idx}_input_ids'])
                        all_neg_masks_list.append(inputs[f'neg{neg_idx}_attention_mask'])
                        neg_idx += 1
                    num_available_negs = len(all_neg_inputs_list)
                    num_to_use = getattr(self.data_args, 'num_contrastive_negatives', 3)
                    num_to_use = min(num_to_use, num_available_negs)
                    if len(neg_inputs_list) > 0:
                        batch_size = neg_inputs_list[0].size(0)
                        max_neg_len = max(t.size(1) for t in neg_inputs_list)
                        padded_neg_ids = []
                        padded_neg_masks = []
                        for neg_ids, neg_mask in zip(neg_inputs_list, neg_masks_list):
                            current_len = neg_ids.size(1)
                            if current_len < max_neg_len:
                                pad_len = max_neg_len - current_len
                                neg_ids = torch.nn.functional.pad(neg_ids, (0, pad_len), value=model.config.pad_token_id)
                                neg_mask = torch.nn.functional.pad(neg_mask, (0, pad_len), value=0)
                            padded_neg_ids.append(neg_ids)
                            padded_neg_masks.append(neg_mask)
                        all_neg_ids = torch.cat(padded_neg_ids, dim=0)  # [batch_size * num_negs, seq_len]
                        all_neg_masks = torch.cat(padded_neg_masks, dim=0)
                        neg_encoder = model.get_encoder()
                        all_neg_encoder_outputs = neg_encoder(input_ids=all_neg_ids, attention_mask=all_neg_masks)
                        from modeling_bart import shift_tokens_right
                        all_neg_decoder_input_ids = shift_tokens_right(
                            all_neg_ids,
                            model.config.pad_token_id,
                            model.config.decoder_start_token_id
                        )
                        
                        neg_decoder = model.get_decoder()
                        all_neg_decoder_outputs = neg_decoder(
                            input_ids=all_neg_decoder_input_ids,
                            attention_mask=None,
                            encoder_hidden_states=all_neg_encoder_outputs.last_hidden_state,
                            encoder_attention_mask=all_neg_masks,
                            output_hidden_states=True,
                        )
                        all_neg_decoder_last_hidden = all_neg_decoder_outputs.hidden_states[-1] if hasattr(all_neg_decoder_outputs, 'hidden_states') and all_neg_decoder_outputs.hidden_states else all_neg_decoder_outputs.last_hidden_state
                        with torch.cuda.amp.autocast(enabled=False):
                            all_neg_decoder_fp32 = all_neg_decoder_last_hidden.float()
                            neg_decoder_states_split = all_neg_decoder_fp32.split(batch_size, dim=0)
                            neg_decoder_ids_split = all_neg_decoder_input_ids.split(batch_size, dim=0)
                            
                            for neg_decoder_hidden, neg_decoder_ids in zip(neg_decoder_states_split, neg_decoder_ids_split):
                                neg_decoder_mask = (neg_decoder_ids != model.config.pad_token_id).unsqueeze(-1).to(neg_decoder_hidden.dtype)
                                neg_denom = neg_decoder_mask.sum(dim=1) + 1e-8
                                neg_latent = (neg_decoder_hidden * neg_decoder_mask).sum(dim=1) / neg_denom
                                neg_latents.append(neg_latent)

                    if len(neg_latents) > 0:
                        if hasattr(self, 'projection_head'):
                            try:
                                proj_device = next(self.model.parameters()).device
                                self.projection_head.to(proj_device)
                            except Exception:
                                pass
                            with torch.cuda.amp.autocast(enabled=False):
                                main_latent_proj = self.projection_head(main_latent.float())
                                pos_latent_proj = self.projection_head(pos_latent.float())
                                neg_latents_proj = [self.projection_head(nl.detach().float()) for nl in neg_latents]
                                main_latent_norm = F.normalize(main_latent_proj, p=2, dim=1)
                                pos_latent_norm = F.normalize(pos_latent_proj, p=2, dim=1)
                                neg_latents_norm = [F.normalize(x, p=2, dim=1) for x in neg_latents_proj]
                                pos_sim = torch.sum(main_latent_norm * pos_latent_norm, dim=1)  # [B]
                                neg_sims = [torch.sum(main_latent_norm * neg_latent_norm, dim=1) for neg_latent_norm in neg_latents_norm]  # List of [B]
                                neg_sim_matrix = torch.stack(neg_sims, dim=1)  # [B, num_negatives]
                                margin = getattr(self.data_args, 'contrastive_margin', 0.2)
                                # pos_sim.unsqueeze(1): [B, 1], neg_sim_matrix: [B, N]
                                pairwise_hinge = F.relu(margin - pos_sim.unsqueeze(1) + neg_sim_matrix)  # [B, N]
                                contrastive_loss = pairwise_hinge.mean()
                                contrastive_loss_value = float(contrastive_loss.detach())
                        else:
                            main_latent_proj = main_latent
                            pos_latent_proj = pos_latent
                            neg_latents_proj = neg_latents
                            main_latent_norm = F.normalize(main_latent_proj, p=2, dim=1)
                            pos_latent_norm = F.normalize(pos_latent_proj, p=2, dim=1)
                            neg_latents_norm = [F.normalize(x, p=2, dim=1) for x in neg_latents_proj]
                            pos_sim = torch.sum(main_latent_norm * pos_latent_norm, dim=1)  # [B]
                            neg_sims = [torch.sum(main_latent_norm * neg_latent_norm, dim=1) for neg_latent_norm in neg_latents_norm]  # List of [B]
                            neg_sim_matrix = torch.stack(neg_sims, dim=1)  # [B, num_negatives]
                            margin = getattr(self.data_args, 'contrastive_margin', 0.2)
                            pairwise_hinge = F.relu(margin - pos_sim.unsqueeze(1) + neg_sim_matrix)  # [B, N]
                            contrastive_loss = pairwise_hinge.mean()
                            contrastive_loss_value = float(contrastive_loss.detach())

                        weight = getattr(self.data_args, 'contrastive_loss_weight', 0.1)
                        loss = loss + weight * contrastive_loss
        self.total_cross_entropy_loss += cross_entropy_loss
        self.total_contrastive_loss += contrastive_loss_value
        self._current_step_losses = {
            'cross_entropy_loss': cross_entropy_loss,
            'contrastive_loss': contrastive_loss_value,
        }
        if getattr(self, 'accelerator', None) is None:
            loss.backward()
        else:
            self.accelerator.backward(loss)
        return loss.detach()
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs,
    ) -> Dict[str, float]:

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * getattr(self.args, "world_size", 1)
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_steps / total_batch_size) if hasattr(output, 'num_steps') else math.ceil(
                    output.num_samples / total_batch_size),
            )
        )
        if output.predictions is not None and hasattr(output, 'label_ids') and output.label_ids is not None:
            try:
                from rouge_score import rouge_scorer
                preds = output.predictions
                labels = output.label_ids
                tokenizer = self.tokenizer
                preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds = [pred.strip() for pred in decoded_preds]
                decoded_labels = [label.strip() for label in decoded_labels]
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                rouge1_scores = []
                rouge2_scores = []
                rougeL_scores = []
                
                for pred, label in zip(decoded_preds, decoded_labels):
                    scores = scorer.score(label, pred)  
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                output.metrics[f"{metric_key_prefix}_rouge1"] = round(np.mean(rouge1_scores) * 100, 4)
                output.metrics[f"{metric_key_prefix}_rouge2"] = round(np.mean(rouge2_scores) * 100, 4)
                output.metrics[f"{metric_key_prefix}_rougeL"] = round(np.mean(rougeL_scores) * 100, 4)
                
            except Exception as e:
                logger.warning(f"Failed to compute ROUGE scores: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        self.log(output.metrics)
        print(f"\n{'='*70}")
        print(f"***** {metric_key_prefix.upper()} Results at Step {self.state.global_step} *****")
        if f"{metric_key_prefix}_rouge1" in output.metrics:
            print(f"  ROUGE-1: {output.metrics[f'{metric_key_prefix}_rouge1']:.4f}")
            print(f"  ROUGE-2: {output.metrics[f'{metric_key_prefix}_rouge2']:.4f}")
            print(f"  ROUGE-L: {output.metrics[f'{metric_key_prefix}_rougeL']:.4f}")
        if f"{metric_key_prefix}_loss" in output.metrics:
            print(f"  Loss: {output.metrics[f'{metric_key_prefix}_loss']:.4f}")
        print(f"{'='*70}\n")
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        if output.predictions is not None:
            step = getattr(self.state, "global_step", 0)
            summary_file = os.path.join(self.args.output_dir, f"STEP-{step}_generated_summaries.json")
            preds = output.predictions
            tokenizer = self.tokenizer
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_preds = [pred.strip().replace("\n", " ") for pred in decoded_preds]
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(summary_file, "w", encoding='utf-8') as file:
                json.dump(decoded_preds, file, indent=4, ensure_ascii=False)

        return output.metrics

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler
        if hasattr(self, 'data_args') and getattr(self.data_args, 'do_contrastive_learning', False) and hasattr(self, 'projection_head'):
            optimizer.add_param_group({
                'params': list(self.projection_head.parameters()),
                'lr': getattr(self.args, 'learning_rate', 5e-5)
            })
            try:
                model_device = next(self.model.parameters()).device
                self.projection_head.to(model_device)
            except Exception:
                logger.warning("Failed to move projection_head to model device in create_optimizer_and_scheduler(); ensure it's on correct device before training.")

        return optimizer, scheduler

    def _remove_unused_columns(
            self, dataset: Dataset, description: Optional[str] = None
    ) -> Dataset:
        import inspect
        signature = inspect.signature(self.model.forward)
        signature_columns = list(signature.parameters.keys())
        signature_columns += ['pos_input_ids', 'pos_attention_mask']
        signature_columns += ['event_input_ids', 'event_attention_mask']
        if len(dataset) > 0:
            first_item = dataset[0]
            neg_idx = 0
            while f'neg{neg_idx}_input_ids' in first_item:
                signature_columns.append(f'neg{neg_idx}_input_ids')
                signature_columns.append(f'neg{neg_idx}_attention_mask')
                neg_idx += 1
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        if ignored_columns and getattr(self.args, "verbose", False):
            logger.info(f" {ignored_columns}")
        return dataset.remove_columns(ignored_columns)
