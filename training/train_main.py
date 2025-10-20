import os
os.environ["TORCH_FORCE_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_START_METHOD"] = "thread"

import argparse, random, csv, json
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    AutoTokenizer
)


from training.utils import (
    load_tokenizer, load_model, load_dataset,
    compute_metrics, load_align_dict_tsv,
    LANG2ID, make_sentence_pairs, load_golden_pairs
)
import training

from training.normalization import normalize_text

from training.custom_collator import CustomMultiTaskCollator

from training.losses import load_loss
from training.losses.simcse_loss import simcse_batch_golden_loss
from models.fine_tune_mbert import FineTuneMBertWrapper
from models.egyptian_bert import EgyptianBertWrapper
from models.egyptian_transformer import EgyptianTransformer

from importlib import import_module
sl = import_module("training.losses.simcse_loss")




from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()



import wandb
from transformers import TrainerCallback

project_root = Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def pad_to_len(seq, pad_id, L):
    return seq + [pad_id] * (L - len(seq))



MAX_TOTAL = 128
HALF1, HALF2 = (MAX_TOTAL - 1) // 2, (MAX_TOTAL - 1) - (MAX_TOTAL - 1) // 2

def build_collate_fn(tokenizer_dict, args, align_dict):

    def collate_fn(batch):
        loss_type = args.loss_type
        input_ids, attn_mask, lid = [], [], []
        align_pairs, sent_flags   = [], []
        text_en_list              = []
        raw_text_list             = []
        offset_mapping_list = []

        for sample in batch:
            lang = sample["lang"]
            tok  = tokenizer_dict[lang]


            orig, en = make_sentence_pairs(sample["text"])
            has_en   = (en.strip() != "")

            offsets = sample.get("offset_mapping", [])




            if args.with_translation:
                orig_ids = tok(orig, truncation=True, padding=False, max_length=HALF1)["input_ids"]
                en_ids   = tok(en if has_en else orig, truncation=True, padding=False, max_length=HALF2)["input_ids"]
                ids  = orig_ids + [tok.sep_token_id] + en_ids
                attn = [1]*len(ids)
                ids  = pad_to_len(ids, tok.pad_token_id, MAX_TOTAL)
                attn = attn + [0]*(MAX_TOTAL-len(attn))

                orig_ids_len = len(tok(orig, truncation=True, padding=False, max_length=HALF1)["input_ids"])
                orig_offsets = offsets[:orig_ids_len]
                padded_offsets = orig_offsets + [(0, 0)] * (MAX_TOTAL - len(orig_offsets))
                offset_mapping_list.append(padded_offsets)

            else:
                enc  = tok(orig, truncation=True, padding="max_length", max_length=MAX_TOTAL)
                ids, attn = enc["input_ids"], enc["attention_mask"]
                offset_mapping_list.append(offsets)

            input_ids.append(ids)
            attn_mask.append(attn)
            lid.append(LANG2ID[lang])
            text_en_list.append(en if has_en else "")
            raw_text_list.append(orig)

        max_len = max(len(o) for o in offset_mapping_list)
        padded_offset_mapping_list = [o + [(0, 0)] * (max_len - len(o)) for o in offset_mapping_list]

        batch_out = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn_mask),
            "lang_id": torch.tensor(lid),
            "raw_text": raw_text_list,
            "offset_mapping": torch.tensor(padded_offset_mapping_list, dtype=torch.long),
            "align_pairs": torch.tensor(align_pairs) if align_pairs else None,
            "sent_flags": torch.tensor(sent_flags) if args.use_sent_cl else None,
            "text_en": text_en_list,
        }

        if loss_type == "mlm":
            batch_out["labels"] = batch_out["input_ids"].clone()

        return batch_out

    return collate_fn




class ClearCudaMemoryCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):


        self.model_size = kwargs.pop("model_size", "tiny")
        self.tokenizer_dict = kwargs.pop("tokenizer_dict", None)

        self.loss_fn = kwargs.pop("loss_fn", None)
        self.loss_type = kwargs.pop("loss_type", "mlm")
        self.mlm_weight = kwargs.pop("mlm_weight", 0.0)
        self.tlm_weight = kwargs.pop("tlm_weight", 0.0)
        self.translation_weight = kwargs.pop("translation_weight", 0.0)
        self.pos_weight = kwargs.pop("pos_weight", 0.0)
        self.consistency_lambda = kwargs.pop("consistency_lambda", 0.0)

        self.contrastive_sampler = kwargs.pop("contrastive_sampler", None)
        self.contrastive_weight = kwargs.pop("contrastive_weight", 0.0)
        self.contrastive_batch_size = kwargs.pop("contrastive_batch_size", 64)
        self.ccl_weight = kwargs.pop("ccl_weight", 0.0)
        self.fusion_mode = kwargs.pop("fusion_mode", "none")

        super().__init__(*args, **kwargs)


        if self.model_size == "bert_all":
            print("\n💡 [bert_all] Initializing learnable parameters for uncertainty weighting.")


            self.task_weights = {
                'mlm': self.mlm_weight,
                'tlm': self.tlm_weight,
                'translation': self.translation_weight,
                'pos': self.pos_weight,

            }


            self.task_log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(()))
                for name, weight in self.task_weights.items() if weight > 0
            }).to(self.args.device)

            if self.optimizer is not None:
                self.optimizer.add_param_group({"params": self.task_log_vars.parameters()})

            print(f"   - Active main tasks: {list(self.task_log_vars.keys())}")

            print(f"   - Consistency Lambda (fixed weight): {self.consistency_lambda}")
            print(f"   - ⭐ Fusion Mode: {self.fusion_mode}")



    def _log_fusion_diagnostics(self, inputs, outputs):

        try:

            fusion_mode = getattr(self, "fusion_mode", "none")
            if fusion_mode != "alpha":
                return


            log_every = max(1, getattr(self.args, "logging_steps", 200))
            if (self.state.global_step or 0) % log_every != 0:
                return

            import torch

            attn = inputs.get("attention_mask", None)
            norm_attn = inputs.get("norm_attention_mask", None)


            if attn is None or norm_attn is None:
                return

            if isinstance(attn, torch.Tensor):
                max_len_orig = int(attn.sum(dim=1).max().item())
                sum_len_orig = float(attn.sum().item())
            else:
                return


            max_len_norm = int(norm_attn.sum(dim=1).max().item())
            sum_len_norm = float(norm_attn.sum().item())


            fused_mask = torch.clamp(attn + norm_attn, max=1)
            max_len_fused = int(fused_mask.sum(dim=1).max().item())


            denom = max(1.0, sum_len_orig)
            norm_coverage = float(sum_len_norm / denom)


            alpha_val = None
            if isinstance(outputs, dict) and ("alpha" in outputs) and (outputs["alpha"] is not None):
                try:
                    if hasattr(outputs["alpha"], "item"):
                        alpha_val = float(outputs["alpha"].item())
                    else:
                        alpha_val = float(outputs["alpha"])
                except Exception:
                    alpha_val = None
            if alpha_val is None and hasattr(self.model, "alpha_logit"):
                import math, torch
                alpha_val = float(torch.sigmoid(self.model.alpha_logit).item())


            payload = {
                "fusion/alpha": alpha_val if alpha_val is not None else -1.0,
                "fusion/max_len_orig": max_len_orig,
                "fusion/max_len_norm": max_len_norm,
                "fusion/max_len_fused": max_len_fused,
                "fusion/norm_coverage": norm_coverage,
            }
            self.log(payload)


            alpha_str = f"{alpha_val:.4f}" if alpha_val is not None else "N/A"
            print(
                f"[FusionDiag @ step {self.state.global_step}] "
                f"alpha={alpha_str} | "
                f"len(orig/norm/fused)=[{max_len_orig}/{max_len_norm}/{max_len_fused}] | "
                f"coverage={norm_coverage:.3f}"
            )

        except Exception as e:

            print(f"[FusionDiag] skip due to error: {e}")
            return



    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        from pathlib import Path
        import numpy as np


        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        all_losses, mlm_losses, tlm_losses, translation_losses, pos_losses, contrastive_losses, ccl_losses, consistency_losses = [], [], [], [], [], [], [], []

        eval_dataloader = self.get_eval_dataloader(eval_ds)
        self.model.eval()

        for inputs in eval_dataloader:
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                if self.model_size == "bert_all":
                    is_tlm_flags = inputs.pop("is_tlm", None)
                    inputs["fusion_mode"] = getattr(self, "fusion_mode", "none")
                    outputs = self.model(**inputs)
                    loss_dict = outputs.get("loss_dict", {})




                    final_loss, outputs_after = self.compute_loss(self.model, {**inputs, "is_tlm": is_tlm_flags},
                                                                  return_outputs=True)
                    if final_loss is not None:
                        all_losses.append(final_loss.item())


                    if self.mlm_weight > 0 and "mlm_loss" in loss_dict and is_tlm_flags is not None and (~is_tlm_flags).sum() > 0:
                        mlm_losses.append(loss_dict["mlm_loss"].item())

                    if self.tlm_weight > 0 and "tlm_loss" in loss_dict and is_tlm_flags is not None and is_tlm_flags.sum() > 0:
                        tlm_losses.append(loss_dict["tlm_loss"].item())

                    if self.translation_weight > 0 and "translation_loss" in loss_dict:
                        translation_losses.append(loss_dict["translation_loss"].item())
                    if self.pos_weight > 0 and "pos_loss" in loss_dict:
                        pos_losses.append(loss_dict["pos_loss"].item())
                    if self.consistency_lambda > 0 and "consistency_loss" in outputs_after:
                        consistency_losses.append(outputs_after["consistency_loss"].item())

                    if self.fusion_mode == "none" and self.consistency_lambda > 0 and "consistency_loss" in outputs_after:
                        consistency_losses.append(outputs_after["consistency_loss"].item())


                    continue

                if self.loss_type == "transformer_multitask":

                    if self.contrastive_sampler is not None and self.contrastive_weight > 0:
                        try:
                            indices = next(self.contrastive_sampler['dataloader_iter'])[0]
                        except StopIteration:
                            self.contrastive_sampler['dataloader_iter'] = iter(self.contrastive_sampler['dataloader'])
                            indices = next(self.contrastive_sampler['dataloader_iter'])[0]

                        sampled_entries = [self.contrastive_sampler['entries'][i] for i in indices]
                        src_forms = [e['form'] for e in sampled_entries]
                        tgt_forms = [random.choice(e['english']) if isinstance(e['english'], list) else e['english'] for
                                     e in sampled_entries]

                        src_tokenized = self.tokenizer(src_forms, padding=True, truncation=True,
                                                       return_tensors="pt").to(self.args.device)
                        tgt_tokenized = self.tokenizer(tgt_forms, padding=True, truncation=True,
                                                       return_tensors="pt").to(self.args.device)

                        inputs['contrastive_src_input_ids'] = src_tokenized['input_ids']
                        inputs['contrastive_src_attention_mask'] = src_tokenized['attention_mask']
                        inputs['contrastive_tgt_input_ids'] = tgt_tokenized['input_ids']
                        inputs['contrastive_tgt_attention_mask'] = tgt_tokenized['attention_mask']



                    def _ensure_2d_long_tensor(x, pad_value, device):
                        if x is None:
                            return None
                        if isinstance(x, torch.Tensor):
                            return x.to(device)
                        if isinstance(x, (list, tuple)):
                            if len(x) == 0:
                                return None
                            if isinstance(x[0], torch.Tensor):
                                try:
                                    return torch.stack([t.to(device) for t in x], dim=0)
                                except RuntimeError:
                                    return pad_sequence([t.to(device) for t in x], batch_first=True,
                                                        padding_value=pad_value).long()
                            if isinstance(x[0], (list, tuple)):
                                seqs = [torch.tensor(s, dtype=torch.long) for s in x]
                                return pad_sequence(seqs, batch_first=True, padding_value=pad_value).to(device)
                            return torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
                        return torch.tensor(x, dtype=torch.long, device=device)

                    device = self.args.device
                    dec_pad_id = getattr(getattr(self.model, "decoder_tokenizer", None), "pad_token_id", 0)

                    if "translation_input_ids" in inputs:
                        inputs["translation_input_ids"] = _ensure_2d_long_tensor(inputs["translation_input_ids"],
                                                                                 dec_pad_id, device)

                    if "translation_labels" in inputs:
                        tl = _ensure_2d_long_tensor(inputs["translation_labels"], dec_pad_id, device)
                        if tl is not None and "translation_input_ids" in inputs and inputs[
                            "translation_input_ids"] is not None:
                            pad_mask = (inputs["translation_input_ids"] == dec_pad_id)
                            tl = tl.masked_fill(pad_mask, -100)
                        inputs["translation_labels"] = tl



                    outputs = self.model(**inputs)
                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        all_losses.append(outputs.loss.item())
                    if hasattr(outputs, "mlm_loss") and outputs.mlm_loss is not None:
                        mlm_losses.append(outputs.mlm_loss.item())
                    if hasattr(outputs, "translation_loss") and outputs.translation_loss is not None:
                        translation_losses.append(outputs.translation_loss.item())
                    if hasattr(outputs, "pos_loss") and outputs.pos_loss is not None:
                        pos_losses.append(outputs.pos_loss.item())
                    if hasattr(outputs, "contrastive_loss") and outputs.contrastive_loss is not None:
                        contrastive_losses.append(outputs.contrastive_loss.item())
                    if hasattr(outputs, "ccl_loss") and outputs.ccl_loss is not None:
                        ccl_losses.append(outputs.ccl_loss.item())



        metrics = {}

        if all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.mean(all_losses)

        print("\n--- Evaluation Metrics Summary ---")
        log_path_loss = Path(self.args.output_dir) / "evaluation_log.txt"

        with open(log_path_loss, "a", encoding="utf-8") as f:
            f.write(f"\n=== Eval at Step: {self.state.global_step}, Epoch: {self.state.epoch:.2f} ===\n")

            if f"{metric_key_prefix}_loss" in metrics:
                log_str = f"  - Total Eval Loss:          {metrics[f'{metric_key_prefix}_loss']:.4f}\n"
                print(log_str, end="");
                f.write(log_str)


            if mlm_losses:
                mean_loss = np.mean(mlm_losses);
                metrics[f"{metric_key_prefix}_mlm_loss"] = mean_loss
                log_str = f"  - Mean Eval MLM Loss:         {mean_loss:.4f}\n";
                print(log_str, end="");
                f.write(log_str)

            if tlm_losses:
                mean_loss = np.mean(tlm_losses);
                metrics[f"{metric_key_prefix}_tlm_loss"] = mean_loss
                log_str = f"  - Mean Eval TLM Loss:         {mean_loss:.4f}\n";
                print(log_str, end="");
                f.write(log_str)

            if translation_losses:
                mean_loss = np.mean(translation_losses);
                metrics[f"{metric_key_prefix}_translation_loss"] = mean_loss
                log_str = f"  - Mean Eval Translation Loss: {mean_loss:.4f}\n";
                print(log_str, end="");
                f.write(log_str)
            if pos_losses:
                mean_loss = np.mean(pos_losses);
                metrics[f"{metric_key_prefix}_pos_loss"] = mean_loss
                log_str = f"  - Mean Eval POS Loss:         {mean_loss:.4f}\n";
                print(log_str, end="");
                f.write(log_str)

            if self.fusion_mode == "none" and consistency_losses:
                mean_loss = np.mean(consistency_losses);
                metrics[f"{metric_key_prefix}_consistency_loss"] = mean_loss
                log_str = f"  - Mean Eval Consistency Loss (KL): {mean_loss:.4f}\n"
                print(log_str, end="");
                f.write(log_str)

            if self.model_size == "bert_all" and hasattr(self, "task_log_vars"):
                log_str = "  - Learned Task Weights (log_var):\n"
                print(log_str, end="");
                f.write(log_str)
                for name, log_var in self.task_log_vars.items():
                    metrics[f"{metric_key_prefix}_{name}_log_var"] = log_var.item()
                    log_str = f"    - {name}: {log_var.item():.4f}\n"
                    print(log_str, end="");
                    f.write(log_str)







            if self.model_size == "bert_all" and hasattr(self, "task_log_vars"):
                log_str = "  - Learned Task Weights:\n"
                print(log_str, end="");
                f.write(log_str)
                header = f"    {'Task':<15} | {'Manual W.':<12} | {'Log_Var':<10} | {'Effective W.':<15}\n"
                separator = "    " + "-" * 60 + "\n"
                print(header, end="");
                f.write(header)
                print(separator, end="");
                f.write(separator)
                for name, log_var in self.task_log_vars.items():
                    manual_weight = self.task_weights.get(name, 0.0)
                    effective_weight = manual_weight * torch.exp(-log_var).item()
                    metrics[f"{metric_key_prefix}_{name}_log_var"] = log_var.item()
                    metrics[f"{metric_key_prefix}_{name}_effective_weight"] = effective_weight
                    log_str = f"    {name:<15} | {manual_weight:<12.2f} | {log_var.item():<10.4f} | {effective_weight:<15.4f}\n"
                    print(log_str, end="");
                    f.write(log_str)
        try:
            if hasattr(self.model, "alpha_logit"):
                alpha = torch.sigmoid(self.model.alpha_logit).item()
                metrics[f"{metric_key_prefix}_fusion_alpha"] = alpha
                print(f"  - Fusion alpha (sigmoid):   {alpha:.4f}")
        except Exception:
            pass


        if self.loss_type == "transformer_multitask" and self.translation_weight > 0:
            print("\n--- Translation Quality (on 100 fixed samples) ---")
            from training.utils import evaluate_translation_quality
            translation_metrics = evaluate_translation_quality(model=self.model, eval_dataset=eval_ds,
                                                               tokenizer_dict=self.tokenizer_dict,
                                                               device=self.args.device, max_samples=200,
                                                               debug_log_path=str(Path(
                                                                   self.args.output_dir) / "translation_debug_samples.txt"),
                                                               global_step=self.state.global_step)
            metrics.update({f"{metric_key_prefix}_translation_{k}": v for k, v in translation_metrics.items()})
            print(f"  - Translation BLEU:      {translation_metrics.get('bleu', 0.0):.4f}")
            print(f"  - Distinct-1:            {translation_metrics.get('distinct_1', 0.0):.4f}")
            print(f"  - Distinct-2:            {translation_metrics.get('distinct_2', 0.0):.4f}")
            print(f"  - Repetition Rate:       {translation_metrics.get('repetition_rate', 0.0):.4f}")

        if self.loss_type == "transformer_multitask" and self.pos_weight > 0:
            print("\n--- POS Quality (on 200 fixed samples) ---")
            from training.utils import evaluate_pos_quality

            pos_metrics = evaluate_pos_quality(
                model=self.model,
                eval_dataset=eval_ds,
                tokenizer_dict=self.tokenizer_dict,
                device=self.args.device,
                max_samples=200,
                debug_log_path=str(Path(self.args.output_dir) / "pos_debug_samples.txt"),
                global_step=self.state.global_step
            )
            metrics.update({f"{metric_key_prefix}_{k}": v for k, v in pos_metrics.items()})
            print(f"  - POS F1:                {pos_metrics['pos_f1']:.4f}")

        print("=" * 50)
        self.log(metrics)
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        import torch
        from torch.nn.utils.rnn import pad_sequence

        def _ensure_2d_long_tensor(x, pad_value, device):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    return None

                if isinstance(x[0], torch.Tensor):
                    try:
                        return torch.stack([t.to(device) for t in x], dim=0)
                    except RuntimeError:
                        return pad_sequence([t.to(device) for t in x], batch_first=True, padding_value=pad_value).long()

                if isinstance(x[0], (list, tuple)):
                    seqs = [torch.tensor(s, dtype=torch.long) for s in x]
                    return pad_sequence(seqs, batch_first=True, padding_value=pad_value).to(device)

                return torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)

            return torch.tensor(x, dtype=torch.long, device=device)


        device = inputs["input_ids"].device
        dec_pad_id = getattr(getattr(self.model, "decoder_tokenizer", None), "pad_token_id", 0)

        if "translation_input_ids" in inputs:
            inputs["translation_input_ids"] = _ensure_2d_long_tensor(inputs["translation_input_ids"], dec_pad_id,
                                                                     device)

        if "translation_labels" in inputs:
            tl = _ensure_2d_long_tensor(inputs["translation_labels"], dec_pad_id, device)

            if tl is not None and "translation_input_ids" in inputs:
                pad_mask = (inputs["translation_input_ids"] == dec_pad_id)
                tl = tl.masked_fill(pad_mask, -100)
            inputs["translation_labels"] = tl

        tl = inputs.get("translation_labels", None)

        if model.training and self.contrastive_sampler is not None and self.contrastive_weight > 0:
            try:
                indices = next(self.contrastive_sampler['dataloader_iter'])[0]
            except StopIteration:
                self.contrastive_sampler['dataloader_iter'] = iter(self.contrastive_sampler['dataloader'])
                indices = next(self.contrastive_sampler['dataloader_iter'])[0]

            sampled_entries = [self.contrastive_sampler['entries'][i] for i in indices]
            src_forms = [e['form'] for e in sampled_entries]
            tgt_forms = [random.choice(e['english']) if isinstance(e['english'], list) else e['english'] for e in
                         sampled_entries]


            if self.state.global_step == 0:
                print("\n--- Contrastive Learning Sampled Batch (First Step Sample) ---")
                for i in range(min(3, len(src_forms))):
                    print(f"  - Pair {i + 1}: '{src_forms[i]}' <---> '{tgt_forms[i]}'")
                print("----------------------------------------------------------------")

            src_tokenized = self.tokenizer(src_forms, padding=True, truncation=True, return_tensors="pt").to(
                self.args.device)
            tgt_tokenized = self.tokenizer(tgt_forms, padding=True, truncation=True, return_tensors="pt").to(
                self.args.device)

            inputs['contrastive_src_input_ids'] = src_tokenized['input_ids']
            inputs['contrastive_src_attention_mask'] = src_tokenized['attention_mask']
            inputs['contrastive_tgt_input_ids'] = tgt_tokenized['input_ids']
            inputs['contrastive_tgt_attention_mask'] = tgt_tokenized['attention_mask']

        if self.state.global_step == 0:
            print("\n=== Compute Loss Debug ===")



        if self.model_size == "bert_all":


            is_tlm_flags = inputs.pop("is_tlm", None)
            inputs["fusion_mode"] = getattr(self, "fusion_mode", "none")

            outputs = model(**inputs)
            loss_dict = outputs.get("loss_dict", {})


            main_task_loss = torch.tensor(0.0, device=self.args.device)


            if self.mlm_weight > 0 and 'mlm_loss' in loss_dict:
                if is_tlm_flags is not None and (~is_tlm_flags).sum() > 0:
                    raw_loss = loss_dict['mlm_loss']
                    log_var = self.task_log_vars['mlm']
                    dynamic_loss = torch.exp(-log_var) * raw_loss + log_var
                    manual_weight = self.mlm_weight * ((~is_tlm_flags).sum() / len(is_tlm_flags))
                    main_task_loss += manual_weight * dynamic_loss


            if self.tlm_weight > 0 and 'tlm_loss' in loss_dict:
                if is_tlm_flags is not None and is_tlm_flags.sum() > 0:
                    raw_loss = loss_dict['tlm_loss']
                    log_var = self.task_log_vars['tlm']
                    dynamic_loss = torch.exp(-log_var) * raw_loss + log_var
                    manual_weight = self.tlm_weight * (is_tlm_flags.sum() / len(is_tlm_flags))
                    main_task_loss += manual_weight * dynamic_loss


            if self.translation_weight > 0 and 'translation_loss' in loss_dict:
                raw_loss = loss_dict['translation_loss']
                log_var = self.task_log_vars['translation']
                dynamic_loss = torch.exp(-log_var) * raw_loss + log_var
                main_task_loss += self.translation_weight * dynamic_loss


            if self.pos_weight > 0 and 'pos_loss' in loss_dict:
                raw_loss = loss_dict['pos_loss']
                log_var = self.task_log_vars['pos']
                dynamic_loss = torch.exp(-log_var) * raw_loss + log_var
                main_task_loss += self.pos_weight * dynamic_loss


            consistency_loss = torch.tensor(0.0, device=self.args.device)
            if self.fusion_mode == "none" and self.consistency_lambda > 0 and outputs.get("logits_orig") is not None and outputs.get(
                    "logits_norm") is not None:
                logits_orig = outputs["logits_orig"]
                logits_norm = outputs["logits_norm"]

                attention_mask = inputs.get("attention_mask")

                if logits_orig is not None and logits_norm is not None:


                    def calculate_kl(logits_p, logits_q, mask):
                        log_p = F.log_softmax(logits_p.detach(), dim=-1)
                        q = F.softmax(logits_q, dim=-1)
                        kl_loss = F.kl_div(log_p, q, reduction='none').sum(dim=-1)
                        if mask is not None:
                            kl_loss = (kl_loss * mask).sum() / mask.sum()
                        else:
                            kl_loss = kl_loss.mean()
                        return kl_loss


                    kl_orig_to_norm = calculate_kl(logits_orig, logits_norm, attention_mask)
                    kl_norm_to_orig = calculate_kl(logits_norm, logits_orig, attention_mask)

                    consistency_loss = (kl_orig_to_norm + kl_norm_to_orig) / 2.0

            total_loss = main_task_loss + self.consistency_lambda * consistency_loss


            outputs["loss"] = total_loss
            outputs["main_task_loss"] = main_task_loss
            outputs["consistency_loss"] = consistency_loss
            if is_tlm_flags is not None:
                outputs["is_tlm"] = is_tlm_flags

            return (total_loss, outputs) if return_outputs else total_loss


        if self.loss_type == "transformer_multitask":

            lang_ids = inputs.get("lang_id", None)


            inputs_for_model = {k: v for k, v in inputs.items() if k != "lang_id"}


            mlm_labels = inputs_for_model.pop("labels", None)
            if mlm_labels is not None:
                inputs_for_model["labels"] = mlm_labels


            translation_labels = inputs.get("translation_labels")


            if translation_labels is not None:
                inputs_for_model["translation_labels"] = translation_labels


            pos_labels = inputs_for_model.pop("pos_labels", None)
            if pos_labels is not None:
                inputs_for_model["pos_labels"] = pos_labels





            outputs = model(**inputs_for_model, lang_id=lang_ids)


            if outputs.loss is None:
                outputs.loss = torch.zeros(1, device=self.args.device, requires_grad=True)




            if return_outputs:
                return outputs.loss, outputs
            return outputs.loss


        if self.loss_type == "mlm":
            print("Processing MLM loss")

            lang_ids = inputs.pop("lang_id", None)
            align_pairs = inputs.pop("align_pairs", None)



            if "labels" in inputs:
                print(f"Labels shape: {inputs['labels'].shape}")
                print(f"Non-masked positions: {(inputs['labels'] != -100).sum().item()}")
            else:
                print("Warning: No labels found in inputs!")

            outputs = model(**inputs, lang_id=lang_ids)

            if isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs.loss

            print(f"MLM loss computed: {loss.item():.4f}")
            return (loss, outputs) if return_outputs else loss


        labels = inputs.pop("labels", None)
        lang_ids = inputs.pop("lang_id", None)
        align_pairs = inputs.pop("align_pairs", None)

        outputs = model(**inputs, lang_id=lang_ids)

        if self.loss_type == "simcse":
            emb = outputs.hidden_states[-1][:, 0]
            loss = self.loss_fn(emb)

        elif self.loss_type == "early_fusion":
            logits = outputs.logits
            emb = outputs.hidden_states[-1][:, 0]
            loss = self.loss_fn(logits, labels, emb)

        elif self.loss_type == "english_anchor":
            loss = self.loss_fn(inputs)
            loss = loss.to(self.args.device)
            return (loss, None) if return_outputs else loss

        else:
            logits = outputs.logits
            loss = self.loss_fn(logits, labels)

            if align_pairs is not None:
                emb = outputs.hidden_states[-1]
                align_loss = 0.0
                count = 0
                for (b1, i1, j1, b2, i2, j2) in align_pairs:
                    if (
                            b1 < emb.size(0) and j1 <= emb.size(1)
                            and b2 < emb.size(0) and j2 <= emb.size(1)
                    ):
                        emb1 = emb[b1, i1:j1].mean(dim=0)
                        emb2 = emb[b2, i2:j2].mean(dim=0)
                        align_loss += torch.nn.functional.mse_loss(emb1, emb2)
                        count += 1
                if count > 0:
                    align_loss = align_loss / count
                    print(f"✅ Align pairs used: {count}, align loss: {align_loss.item():.4f}")
                    loss = loss + self.args.align_lambda * align_loss

        if torch.isnan(loss):
            print("⚠️ NaN loss detected!")

        print("=== End Compute Loss ===")
        return (loss, outputs) if return_outputs else loss


    def _save(self, output_dir: str, _internal_call: bool = False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "config"):
            inner.config.save_pretrained(output_dir)

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",            required=True)
    parser.add_argument("--tokenizer_type",  default="bpe",   choices=["bpe","unigram","word","distinct_bpe", "shared_bpe", "xlmr_adapted"])
    parser.add_argument("--model_size",      default="tiny",  choices=["tiny","small","mbert", "ancient", "egyptian", "transformer", "xlmr", "xlmr_warm", "adapted_xlmr", "bert_all"])
    parser.add_argument("--symmetric_encoder_path", type=str, default=None,
                        help="Path to the pre-trained encoder checkpoint to be used for BOTH the encoder and decoder in a symmetric setup (triggers with --model_size adapted_xlmr).")
    parser.add_argument("--loss_type",       default="mlm",   choices=["mlm","simcse","early_fusion", "english_anchor", "simcse_batchgolden", "transformer_multitask"])
    parser.add_argument("--output_dir",      default="checkpoints")
    parser.add_argument("--use_wandb",       action="store_true")
    parser.add_argument("--peft_method",     default=None,    choices=["None","lora","adapter"])
    parser.add_argument("--lora_r",          type=int,        default=8)
    parser.add_argument("--batch_size",      type=int,        default=32)
    parser.add_argument("--eval_batch_size", type=int,        default=32)
    parser.add_argument("--epochs",          type=int,        default=10)
    parser.add_argument("--lr",              type=float,      default=3e-5)
    parser.add_argument("--eval_strategy", default="no", choices=["no","steps","epoch"])
    parser.add_argument("--eval_steps",      type=int,        default=500, help="Number of steps between evaluations when evaluation_strategy=steps")
    parser.add_argument("--eval_accumulation_steps", type=int, default=10)
    parser.add_argument("--multi",           action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16",            action="store_true")
    parser.add_argument("--with_translation", action="store_true")
    parser.add_argument("--lr_scheduler_type", default="linear", choices=["linear","cosine","cosine_with_restarts","polynomial", "constant_with_warmup"])
    parser.add_argument("--warmup_steps",    type=int,        default=0)
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument("--mlm_weight",      type=float,      default=0.0)
    parser.add_argument("--tlm_weight", type=float, default=0.0,
                        help="Weight for the Translation Language Modeling task. 0=off.")
    parser.add_argument("--translation_weight", type=float,   default=0.0)
    parser.add_argument("--pos_weight",      type=float,      default=0.0)
    parser.add_argument("--consistency_lambda", type=float, default=0.0,
                        help="Weight for the KL-divergence consistency loss between original and normalized views. 0=off.")
    parser.add_argument("--normalization_mode", type=str, default="NONE", choices=["NONE", "LATIN", "IPA"],
                        help="The normalization strategy to apply for the augmented view in consistency training.")
    parser.add_argument("--fusion_mode", type=str, default="none", choices=["none", "alpha"],
                        help="Path choice: 'none' = dual-channel with optional KL; 'alpha' = early fusion single-encoder (no KL).")


    parser.add_argument("--num_masks", type=int, default=1,
                        help="Number of different masked versions to create for each sentence in MLM.")
    parser.add_argument("--weight_decay",    type=float, default=0.0,  help="Weight decay for regularization")
    parser.add_argument("--dropout",         type=float, default=0.0,  help="Dropout rate")
    parser.add_argument("--attention_dropout", type=float, default=0.0, help="Attention dropout rate")

    parser.add_argument("--mask_strategy", 
                       choices=["token", "word"], 
                       default="token",
                       help="Masking strategy: token-level or word-level masking")

    return parser.parse_args()

def _format_lang_tag(lang_str: str) -> str:

    return "multi" if "," in lang_str else lang_str



def get_stage_output_dir(base_dir: Path, args) -> Path:

    stage_suffix = f"_stage_{args.training_stage}"
    if args.continue_from_stage:
        stage_suffix += f"_from_{args.continue_from_stage}"
    

    weights_info = f"_mlm{args.mlm_weight}_trans{args.translation_weight}_pos{args.pos_weight}"
    
    
    return base_dir / (_format_lang_tag(base_dir.name) + stage_suffix + weights_info)



class TransformerCollator:

    
    def __init__(self, tokenizer, training_stage="all", mlm=True, mlm_probability=0.15, num_masks=1):

        self.tokenizer = tokenizer
        self.is_dict_tokenizer = isinstance(tokenizer, dict)
        self.training_stage = training_stage
        self.mlm = mlm and training_stage in ["all", "mlm_only"]
        self.mlm_probability = mlm_probability
        self.num_masks = num_masks
        self._printed_mlm = False

        if self.is_dict_tokenizer:
            first_tok = next(iter(tokenizer.values()))
            self.pad_token_id = first_tok.pad_token_id
            self.mask_token_id = first_tok.mask_token_id
            print(f"\nInitialized with dictionary tokenizer. Languages: {list(tokenizer.keys())}")
            print(f"Mask token IDs: {[(lang, tok.mask_token_id) for lang, tok in tokenizer.items()]}")
        else:
            self.pad_token_id = tokenizer.pad_token_id
            self.mask_token_id = tokenizer.mask_token_id
            print(f"\nInitialized with single tokenizer. Mask token ID: {self.mask_token_id}")

    def __call__(self, examples):

        lang = examples[0].get("lang", None) if examples and isinstance(examples[0], dict) else None

        current_tokenizer = self.tokenizer.get(lang, self.tokenizer) if self.is_dict_tokenizer else self.tokenizer

        original_offsets = [e.get('offset_mapping', []) for e in examples]

        base_batch = {
            'input_ids': torch.tensor([e['input_ids'] for e in examples], dtype=torch.long),
            'attention_mask': torch.tensor([e['attention_mask'] for e in examples], dtype=torch.long),
            'lang_id': torch.tensor([e.get('lang_id', 0) for e in examples], dtype=torch.long),
            'lang': lang,
            'raw_text': [e['text'] for e in examples],
        }


        if self.mlm and self.training_stage in ["all", "mlm_only"]:
            if not self._printed_mlm:
                print(f"\nProcessing MLM in collator for language: {lang}")

            mlm_processor = DataCollatorForLanguageModeling(
                tokenizer=current_tokenizer,
                mlm=True,
                mlm_probability=self.mlm_probability
            )


            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            all_lang_ids = []

            raw_text_list = base_batch["raw_text"]

            for _ in range(self.num_masks):

                mlm_outputs = mlm_processor([
                    {'input_ids': ids.tolist()} for ids in base_batch['input_ids']
                ])
                
                all_input_ids.append(torch.tensor(mlm_outputs['input_ids'], dtype=torch.long))
                all_attention_masks.append(base_batch['attention_mask'].clone())
                all_labels.append(torch.tensor(mlm_outputs['labels'], dtype=torch.long))
                all_lang_ids.append(base_batch['lang_id'].clone())
            

            batch = {
                'input_ids': torch.cat(all_input_ids),
                'attention_mask': torch.cat(all_attention_masks),
                'labels': torch.cat(all_labels),
                'lang_id': torch.cat(all_lang_ids),
                'lang': lang,
                'raw_text': raw_text_list * self.num_masks
            }
            

            n_tokens = batch['attention_mask'].sum().item()
            n_masked = (batch['labels'] != -100).sum().item()
            if not self._printed_mlm:
                print(f"Total tokens: {n_tokens}, Masked tokens: {n_masked}")
                print(f"Masking ratio: {n_masked/n_tokens:.2%}")
                print(f"Using tokenizer for language: {lang}")
                print(f"Mask token ID: {current_tokenizer.mask_token_id}")
                print(f"Generated {self.num_masks} mask versions for each sentence")
                self._printed_mlm = True
        else:
            batch = base_batch
            batch['labels'] = None

        if self.training_stage in ["all", "translation_only"]:

            if any('translation_labels' in e for e in examples):
                tl = torch.tensor([e['translation_labels'] for e in examples], dtype=torch.long)
                if self.mlm and self.training_stage in ["all", "mlm_only"]:
                    tl = tl.repeat(self.num_masks, 1)
                batch['translation_labels'] = tl
            else:
                batch['translation_labels'] = None


            batch['decoder_input_ids'] = None
            batch['decoder_attention_mask'] = None

        if self.training_stage in ["all", "pos_only"]:
            if any('pos_labels' in e for e in examples):
                pos_labels = torch.tensor([e['pos_labels'] for e in examples], dtype=torch.long)
                if self.mlm and self.training_stage in ["all", "mlm_only"]:

                    pos_labels = pos_labels.repeat(self.num_masks, 1)
                batch['pos_labels'] = pos_labels
            else:
                batch['pos_labels'] = None

                if self.mlm and self.training_stage in ["all", "mlm_only"]:
                    offset_mapping_list = original_offsets * self.num_masks
                else:
                    offset_mapping_list = original_offsets


                max_len = batch['input_ids'].size(1)
                padded_offsets = [o + [(0, 0)] * (max_len - len(o)) for o in offset_mapping_list]
                batch['offset_mapping'] = torch.tensor(padded_offsets, dtype=torch.long)

        return batch





class WordLevelMLMCollator:
    def __init__(self, tokenizer, mlm_probability=0.15, num_masks=1):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.num_masks = num_masks
        self._printed = 0

    def __call__(self, examples):
        batch = []
        for example in examples:
            text = example["text"]

            words = text.split()
            

            word_to_tokens = []
            current_pos = 1
            for word in words:
                word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
                word_to_tokens.append((current_pos, current_pos + len(word_tokens), word))
                current_pos += len(word_tokens)
            

            num_words = len(words)
            num_to_mask = max(1, int(num_words * self.mlm_probability))


            for _ in range(self.num_masks):

                token_ids = self.tokenizer.encode(text, 
                    truncation=True, 
                    max_length=512, 
                    padding="max_length"
                )
                labels = [-100] * len(token_ids)
                

                mask_word_indices = random.sample(range(num_words), num_to_mask)
                for word_idx in mask_word_indices:
                    start, end = word_to_tokens[word_idx][0:2]

                    for pos in range(start, end):
                        labels[pos] = token_ids[pos]
                        token_ids[pos] = self.tokenizer.mask_token_id
                

                if self._printed < 1:
                    print("\n=== Masking Debug ===")
                    print(f"Original text: {text}")
                    print(f"Words to mask: {[words[idx] for idx in mask_word_indices]}")


                    print("\nMasked text:")
                    print(self.tokenizer.decode(token_ids))
                    print("Labels:", [self.tokenizer.decode([label]) if label != -100 else "_" for label in labels])
                    self._printed += 1

                batch.append({
                    "input_ids": token_ids,
                    "attention_mask": [1] * len(token_ids),
                    "labels": labels
                })

        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch]),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch])
        }


from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling
import random


class BertAllMultiTaskCollator:
    def __init__(self, tokenizer, normalization_mode="NONE", mlm_probability=0.15, mlm_weight=0.0, tlm_weight=0.0,
                 translation_weight=0.0, pos_weight=0.0):

        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_probability
        )
        from training.utils import TAG_BY_LANG, LANG2ID
        self.TAG_BY_LANG, self.LANG2ID = TAG_BY_LANG, LANG2ID
        self.do_mlm, self.do_tlm = mlm_weight > 0, tlm_weight > 0
        if self.do_mlm and self.do_tlm:
            self.tlm_ratio = tlm_weight / (mlm_weight + tlm_weight)
        elif self.do_tlm:
            self.tlm_ratio = 1.0
        else:
            self.tlm_ratio = 0.0
        self.normalization_mode = normalization_mode
        self.translation_weight = translation_weight
        self.pos_weight = pos_weight

    def _process_view(self, examples: List[dict], decisions: List[dict], use_normalized_text: bool) -> dict:

        processed_examples = []
        MAX_LEN, HALF_LEN = 768, (768 - 5) // 2
        for i, ex in enumerate(examples):
            lang, egyptian_text, english_text = ex["lang"], ex["text"], ex["translation"]
            if use_normalized_text and self.normalization_mode != "NONE" and lang != "english":
                egyptian_text = normalize_text(egyptian_text, lang, self.normalization_mode)
            lang_tag_egyptian = self.TAG_BY_LANG.get(lang, "")
            if decisions[i]["is_tlm"]:
                lang_tag_english = "<eng>"
                egyptian_tokens = self.tokenizer.tokenize(egyptian_text, add_special_tokens=False)[:HALF_LEN]
                english_tokens = self.tokenizer.tokenize(english_text, add_special_tokens=False)[:HALF_LEN]
                egyptian_processed = self.tokenizer.convert_tokens_to_string(egyptian_tokens)
                english_processed = self.tokenizer.convert_tokens_to_string(english_tokens)
                encoder_input_text = f"{lang_tag_egyptian} {egyptian_processed} [SEP] <eng> {english_processed}"
            else:
                if decisions[i]["use_egyptian"] and egyptian_text:
                    encoder_input_text = f"{lang_tag_egyptian} {egyptian_text}"
                else:
                    encoder_input_text = f"<eng> {english_text}"
            processed_examples.append({"encoder_input": encoder_input_text})
        encoder_texts = [p_ex["encoder_input"] for p_ex in processed_examples]
        encoder_inputs_no_mask = self.tokenizer(
            encoder_texts, truncation=True, padding="longest", max_length=MAX_LEN, return_tensors=None
        )
        mlm_input_features = [{"input_ids": ids} for ids in encoder_inputs_no_mask["input_ids"]]
        mlm_result = self.mlm_collator(mlm_input_features)
        is_tlm_tensor = torch.tensor([d["is_tlm"] for d in decisions], dtype=torch.bool)
        mixed_labels = mlm_result["labels"];
        mlm_labels = mixed_labels.clone();
        tlm_labels = mixed_labels.clone()
        tlm_mask = is_tlm_tensor.unsqueeze(1).expand_as(mlm_labels)
        mlm_labels[tlm_mask] = -100;
        tlm_labels[~tlm_mask] = -100
        return {
            "input_ids": mlm_result["input_ids"],
            "attention_mask": torch.tensor(encoder_inputs_no_mask["attention_mask"], dtype=torch.long),

            "raw_input_ids_no_mlm": encoder_inputs_no_mask["input_ids"],
            "labels": mlm_result["labels"]
        }

    def __call__(self, examples: List[dict]) -> dict:
        decisions = [{"is_tlm": random.random() < self.tlm_ratio, "use_egyptian": random.random() < 0.5} for _ in
                     examples]


        original_view = self._process_view(examples, decisions, use_normalized_text=False)

        batch = {
            "input_ids": original_view["input_ids"],
            "attention_mask": original_view["attention_mask"],
            "lang_id": torch.tensor([self.LANG2ID.get(ex["lang"], -1) for ex in examples], dtype=torch.long),
        }


        is_tlm_tensor = torch.tensor([d["is_tlm"] for d in decisions], dtype=torch.bool)
        mixed_labels = original_view["labels"]
        mlm_labels = mixed_labels.clone();
        tlm_labels = mixed_labels.clone()
        tlm_mask = is_tlm_tensor.unsqueeze(1).expand_as(mlm_labels)
        mlm_labels[tlm_mask] = -100;
        tlm_labels[~tlm_mask] = -100
        batch["mlm_labels"] = mlm_labels;
        batch["tlm_labels"] = tlm_labels
        batch["is_tlm"] = is_tlm_tensor


        if self.normalization_mode != "NONE":
            normalized_view = self._process_view(examples, decisions, use_normalized_text=True)

            len_orig = original_view["input_ids"].shape[1]
            len_norm = normalized_view["input_ids"].shape[1]
            target_len = max(len_orig, len_norm)

            def pad_tensor(tensor, length, pad_value):
                if tensor.shape[1] >= length:
                    return tensor[:, :length]
                padding = torch.full((tensor.shape[0], length - tensor.shape[1]), pad_value, dtype=tensor.dtype)
                return torch.cat([tensor, padding], dim=1)


            batch["input_ids"] = pad_tensor(batch["input_ids"], target_len, self.tokenizer.pad_token_id)
            batch["attention_mask"] = pad_tensor(batch["attention_mask"], target_len, 0)
            batch["mlm_labels"] = pad_tensor(batch["mlm_labels"], target_len, -100)
            batch["tlm_labels"] = pad_tensor(batch["tlm_labels"], target_len, -100)


            norm_attn = pad_tensor(normalized_view["attention_mask"], target_len, 0)


            sep_id = self.tokenizer.sep_token_id
            raw_lists = normalized_view["raw_input_ids_no_mlm"]

            for i, is_tlm in enumerate(is_tlm_tensor.tolist()):
                if not is_tlm:
                    continue
                raw_ids = raw_lists[i]

                sep_idx = None
                for j, tid in enumerate(raw_ids):
                    if tid == sep_id:
                        sep_idx = j
                        break
                if sep_idx is None:
                    continue


                cur_len = min(len(raw_ids), target_len)

                norm_attn[i, sep_idx:cur_len] = 0


            batch["norm_input_ids"] = pad_tensor(normalized_view["input_ids"], target_len, self.tokenizer.pad_token_id)
            batch["norm_attention_mask"] = norm_attn


        if self.translation_weight > 0:
            translation_targets = [ex["translation"] for ex in examples]
            decoder_inputs = self.tokenizer(
                translation_targets, truncation=True, padding="longest",
                max_length=768, return_tensors="pt"
            )
            labels = decoder_inputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            batch["translation_labels"] = labels
            decoder_input_ids = torch.full_like(labels, self.tokenizer.pad_token_id)
            decoder_input_ids[:, 1:] = labels[:, :-1].clone()
            decoder_input_ids[:, 0] = self.tokenizer.cls_token_id
            decoder_input_ids[labels == -100] = self.tokenizer.pad_token_id
            batch["decoder_input_ids"] = decoder_input_ids
            batch["decoder_attention_mask"] = (decoder_input_ids != self.tokenizer.pad_token_id).long()

        if self.pos_weight > 0 and examples[0].get("pos_ids") is not None:
            pos_labels_list = []
            max_len = batch["input_ids"].shape[1]
            for ex in examples:
                pos_ids = ex["pos_ids"] if ex["pos_ids"] else []
                padded_pos = pos_ids + [-100] * (max_len - len(pos_ids))
                pos_labels_list.append(padded_pos[:max_len])
            batch["pos_labels"] = torch.tensor(pos_labels_list, dtype=torch.long)

        return batch







def main():
    args = parse_args()
    set_seed(42)

    print("\n=== Loading tokenizers ===")

    tokenizer_dict = {
        lang: load_tokenizer(
            lang=lang,
            tokenizer_type=args.tokenizer_type,
            with_translation=args.loss_type == "transformer_multitask",
            model_size=args.model_size
        )
        for lang in args.lang.split(",")
    }
    vocab_size = next(iter(tokenizer_dict.values())).vocab_size
    
    print(f"Loaded tokenizers for languages: {list(tokenizer_dict.keys())}")




    align_dict = None
    if args.use_token_align:
        align_dict = load_align_dict_tsv(
            project_root / "resources" / "align_dict.tsv"
        )
        print(f"[align] token-align dictionary loaded: {len(align_dict)} language pairs")


    mlm_probability = 0.25 if args.training_stage == "mlm_only" else 0.15
    print(f"\nMLM probability set to: {mlm_probability}")

    num_masks = args.num_masks
    print(f"Number of mask versions per sentence: {num_masks}")


    print(f"Number of mask versions per sentence: {num_masks}")

    data_collator = None

    data_collator = None
    if args.model_size == 'bert_all':
        print(f"\n💡 ！！！！[bert_all] Initializing BertAllMultiTaskCollator with normalization_mode='{args.normalization_mode}'")
        unified_tokenizer = next(iter(tokenizer_dict.values()))
        data_collator = BertAllMultiTaskCollator(
            tokenizer=unified_tokenizer,
            mlm_probability=0.15,
            mlm_weight=args.mlm_weight,
            tlm_weight=args.tlm_weight,
            translation_weight=args.translation_weight,
            pos_weight=args.pos_weight,
            normalization_mode=args.normalization_mode,
        )


    else:

        data_collator = build_collate_fn(tokenizer_dict, args, align_dict)
    




    if args.use_wandb:
        run_name = (
            f"{_format_lang_tag(args.lang)}_{args.model_size}_{args.loss_type}_{args.peft_method or 'full'}_"
            f"{args.tokenizer_type}"
            f"_T{int(args.with_translation)}"
            f"_SC{int(args.use_sent_cl)}"
            f"_TA{int(args.use_token_align)}"
            f"_M{int(args.multi)}"
            f"_epoch{args.epochs}_lr{args.lr:.0e}_b{args.batch_size}"
        )
        if args.peft_method == "lora":
            run_name += f"-r{args.lora_r}"

        wandb.init(project="semantic-shift", name=run_name, config=vars(args), mode="offline")


    tokenizer_for_collator = tokenizer_dict[args.lang.split(",")[0]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("✅ Device:", device)



    model = load_model(
        args.model_size,
        vocab_size,
        loss_type=args.loss_type,
        peft_method=args.peft_method,
        lora_r=args.lora_r,
        temperature_strategy=args.temperature_strategy,
        mlm_weight=args.mlm_weight,
        translation_weight=args.translation_weight,
        pos_weight=args.pos_weight,

        contrastive_weight=args.contrastive_weight,
        ccl_weight=args.ccl_weight,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        checkpoint_path=args.previous_stage_path
    ).to(device)

    if args.model_size == 'bert_all':
        print("💡 [bert_all] Binding the unified tokenizer as 'decoder_tokenizer' for evaluation purposes.")

        model.decoder_tokenizer = next(iter(tokenizer_dict.values()))




    dataset = load_dataset(
        args.lang.split(","), tokenizer_dict,
        with_translation=args.with_translation,
        multi=args.multi,
        model_size=args.model_size,
        pos_weight=args.pos_weight,

        ccl_weight=args.ccl_weight
    )





    out_dir = Path(args.output_dir) / (
        f"{_format_lang_tag(args.lang)}_{args.model_size}_{args.loss_type}_{args.peft_method or 'full'}_"
        f"{args.tokenizer_type}"
        f"_T{int(args.with_translation)}_SC{int(args.use_sent_cl)}_TA{int(args.use_token_align)}_M{int(args.multi)}"
        f"_epoch{args.epochs}_lr{args.lr:.0e}_b{args.batch_size}"
    )




    custom_trainer_args = {
        "model_size": args.model_size,
        "loss_fn": loss_fn,
        "loss_type": args.loss_type,
        "tokenizer_dict": tokenizer_dict,


        "mlm_weight": args.mlm_weight,
        "tlm_weight": args.tlm_weight,
        "translation_weight": args.translation_weight,
        "pos_weight": args.pos_weight,


        "fusion_mode": args.fusion_mode,
    }

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        fp16=args.fp16,
        seed=42,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy="epoch",
        save_total_limit=31,
        save_safetensors=False,
        logging_dir="logs",
        load_best_model_at_end=False,
        metric_for_best_model=None,
        remove_unused_columns=False,
        prediction_loss_only = True,
        dataloader_drop_last=True,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        max_grad_norm=1,
        report_to=["wandb"] if args.use_wandb else [],
        eval_accumulation_steps=args.eval_accumulation_steps,

        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=False,
    )


    print("\n=== Training Configuration ===")
    print(f"Training Stage: {args.training_stage}")
    print(f"Loss Type: {args.loss_type}")
    print(f"Loss Type: {args.loss_type}")
    print(f"MLM Weight: {args.mlm_weight}")
    print(f"TLM Weight: {args.tlm_weight}")
    print(f"Translation Weight: {args.translation_weight}")
    print(f"POS Weight: {args.pos_weight}")

    print(f"Static Contrastive Weight: {args.contrastive_weight}")

    print(f"Contextual Contrastive (CCL) Weight: {args.ccl_weight}")

    print(f"With Translation: {args.with_translation}")
    if args.loss_type == "transformer_multitask":
        print(f"Label Smoothing: 0.1 (enabled)")
    print("============================\n")
    

    training_args.training_stage = args.training_stage


    optimizers = (None, None)




    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer_for_collator,
        optimizers=optimizers,

        **custom_trainer_args,
        callbacks=[
            ClearCudaMemoryCallback()
        ]
    )

    import torch as _T
    from torch.utils.data import DataLoader as _DataLoader



    if args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
    for lg, tok in tokenizer_dict.items():
        tok_dir = out_dir / f"{lg}_tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(tok_dir)

    trainer.evaluate(dataset["test"])


    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
    print("⚠️ TORCH_FORCE_DISABLE_FLASH_ATTENTION =", os.environ.get("TORCH_FORCE_DISABLE_FLASH_ATTENTION"))
