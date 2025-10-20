
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict

import csv
import json
import os
import random
import re
import torch
import torch.nn as nn # ⭐
from tokenizers import Tokenizer
from transformers import (
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModel, # ⭐ [新增] 添加这一行来解决 NameError
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM
)
from transformers.modeling_outputs import BaseModelOutput

from datasets import Dataset, DatasetDict, concatenate_datasets

from models.tiny_transformer import build_tiny_transformer
from models.fine_tune_mbert import FineTuneMBertWrapper
from models.ancient_bert import AncientBertWrapper
from models.egyptian_bert import EgyptianBertWrapper
from training.losses import load_loss
from tokenizers.processors import TemplateProcessing

from transformers import AutoModelForMaskedLM # 确保这个被导入



def _preprocess_egyptian_text(text: str) -> str:

    text = re.sub(r"\.", "", text) # <- 不再全局删除点
    text = text.replace(' •', '')  # 注意这里是 "空格+•"，如果也可能没有空格，可以分两步
    text = text.replace('•', '')  # 再加一步确保所有•都被删除
    text = text.replace('[gap]', '?')



    return text

# ---------- 语言 ↔ id 映射 ----------
LANG2ID: Dict[str, int] = {
    "hieroglyphic": 0,
    "demotic":       1,
    "bohairic":      2,
    "sahidic":       3,
    "en":           4,
}
TAG_BY_LANG = {
    "hieroglyphic": "<hiero>",
    "demotic":      "<dem>",
    "bohairic":     "<boh>",
        "sahidic":      "<sah>",
}

def load_align_dict_tsv(
    tsv_path: str | os.PathLike,
    langs: Tuple[str, str, str] = ("hieroglyphic", "demotic", "sahidic"),
) -> Dict[Tuple[str, str], Dict[str, str]]:

    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"align TSV not found: {path}")

    align_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < len(langs):
                continue
            words = {
                lang: w.strip()
                for lang, w in zip(langs, row)
                if w.strip() and w.strip() != "<empty>"
            }

            langs_with_word = list(words.items())
            for i in range(len(langs_with_word)):
                for j in range(i + 1, len(langs_with_word)):
                    (l1, w1), (l2, w2) = langs_with_word[i], langs_with_word[j]
                    key = tuple(sorted((l1, l2)))
                    align_map.setdefault(key, {})[w1] = w2
                    align_map[key][w2] = w1
    return align_map



def make_sentence_pairs(text_with_en_sep: str, sep_token: str = "[SEP]") -> Tuple[str, str]:
    if sep_token in text_with_en_sep:
        a, b = text_with_en_sep.split(sep_token, 1)
        return a.strip(), b.strip()
    return text_with_en_sep, text_with_en_sep


def load_tokenizer(
    lang: str,
    tokenizer_type: str,
    with_translation: bool = False,
    model_size: str = "tiny",
):

    if model_size == "bert_all":
        tokenizer_path = Path(__file__).parent.parent / "project_tokenizers" / "bert_all" / "tokenizer.json"
        print(f"💡 [bert_all] Loading our custom trained tokenizer from: {tokenizer_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"bert_all tokenizer not found at {tokenizer_path}.")

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))

        tokenizer.pad_token = "[PAD]"
        tokenizer.unk_token = "[UNK]"
        tokenizer.cls_token = "[CLS]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.mask_token = "[MASK]"

        return tokenizer


    else:
        tok_name = (
            f"{lang}_{tokenizer_type}_joint_tokenizer.json"
            if with_translation and model_size in ("tiny", "small")
            else f"{lang}_{tokenizer_type}_tokenizer.json"
        )
        tok_path = (Path(__file__).parent.parent / "project_tokenizers" / tok_name).resolve()

    tokenizer = Tokenizer.from_file(str(tok_path))
    if tokenizer.token_to_id("[UNK]") is None:
        tokenizer.add_special_tokens(["[UNK]"])

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    hf_tok.add_special_tokens({"additional_special_tokens": ["[gap]"]})

    if model_size == "ancient":
        hf_tok.add_special_tokens(
            {"additional_special_tokens": ["<boh>", "<sah>", "<dem>", "<hiero>"]}
        )

    return hf_tok


def load_model(
    model_size: str,
    vocab_size: int,
    loss_type: str = "mlm",
    peft_method: str | None = None,
    lora_r: int = 8,
    checkpoint_path: str | None = None,
    temperature_strategy: str = "fixed",
    tlm_weight: float = 0.0,
    translation_weight: float = 0.0,
    pos_weight: float = 0.0,

    contrastive_weight: float = 0.0,
    ccl_weight: float = 0.0,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
):

    if model_size == "bert_all":
        from models.multi_task_bert_encoder_decoder import MultiTaskBertEncoderDecoder, MultiTaskBertConfig
        print(f"💡 [bert_all] Initializing our custom MultiTaskBertEncoderDecoder model.")


        config = MultiTaskBertConfig(
            vocab_size=vocab_size,
            num_hidden_layers=6,
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            mlm_weight=mlm_weight,
            tlm_weight=tlm_weight,
            translation_weight=translation_weight,
            pos_weight=pos_weight,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_pos_labels=17
        )


        model = MultiTaskBertEncoderDecoder(config=config)


        def _freeze_encoder_layers(self, k: int):
            layers = self.encoder.encoder.layer
            who = "self.encoder.encoder.layer"

            tot = len(layers)
            k = max(0, min(k, tot))
            for idx, layer in enumerate(layers):
                if idx < k:
                    for p in layer.parameters():
                        p.requires_grad_(False)
            print(f"[freeze] 已冻结底部 {k}/{tot} 层 ({who})")
            setattr(self, "frozen_encoder_n", k)

        setattr(model, "freeze_encoder", _freeze_encoder_layers.__get__(model, type(model)))


        return model






from transformers import AutoTokenizer



def load_dataset(
    lang_list,
    tokenizer_dict,
    with_translation: bool = False,
    multi: bool = False,
    model_size: str = "tiny",
    pos_weight: float = 0.0,
    ccl_weight: float = 0.0,

    task: str = "mlm"
):
    if isinstance(lang_list, str):
        lang_list = [lang_list]



    base_dir = Path(__file__).resolve().parent.parent / "data"
    if model_size == "bert_all":

        jsonl_dir = base_dir / "processed_jsonl"
        print(f"\n💡 [bert_all] Loading data from clean source: {jsonl_dir.name}")
    else:

        jsonl_dir = base_dir / "final_for_CCL"
        print(f"\n💡 Loading data from historical source: {jsonl_dir.name}")

    decoder_tok = None



    random.seed(42)

    def load_one_lang_jsonl(lang):

        if model_size == "bert_all":

            path = jsonl_dir / f"{lang}_rev.jsonl"


            
        with path.open(encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        random.shuffle(lines)
        total = len(lines)


        print(f"\n   [Debug] Loaded {total} total samples for language: {lang}")

        train_split = lines[: int(0.8 * total)]
        valid_split = lines[int(0.8 * total): int(0.9 * total)]
        test_split = lines[int(0.9 * total):]


        print(f"     └─ Train: {len(train_split)}, Validation: {len(valid_split)}, Test: {len(test_split)}")

        return {
            "train": train_split,
            "validation": valid_split,
            "test": test_split,
        }


    datasets_by_split = {"train": [], "validation": [], "test": []}

    for lang in lang_list:
        tok  = tokenizer_dict[lang]
        data = load_one_lang_jsonl(lang)

        for split in ("train", "validation", "test"):
            entries = []

            for i, item in enumerate(data[split]):


                if model_size == "bert_all":


                    entry = {
                        "lang": lang,
                        "text": item.get('text', ''),
                        "translation": item.get("translation", ""),
                    }


                    if pos_weight > 0 and "UPOS" in item:

                        alias_map = {"_": "X", "N": "NOUN", "V": "VERB"}
                        raw_tags = item["UPOS"].split()
                        pos_tags = [alias_map.get(t, t) for t in raw_tags]

                        pos_label_map = {
                            "NOUN": 0, "VERB": 1, "ADJ": 2, "ADV": 3, "PRON": 4,
                            "DET": 5, "ADP": 6, "NUM": 7, "CONJ": 8, "PART": 9,
                            "INTJ": 10, "PUNCT": 11, "SYM": 12, "X": 13, "PROPN": 14,
                            "AUX": 15, "CCONJ": 16
                        }

                        entry["pos_ids"] = [pos_label_map.get(t, pos_label_map["X"]) for t in pos_tags]

                    entries.append(entry)
                    continue

                raw_text = item.get('text', '')
                processed_text = raw_text

                tag_tok = TAG_BY_LANG.get(lang)
                source_with_tag = f"{tag_tok} {processed_text}" if tag_tok else processed_text
                entry = {
                    "lang": lang,
                    "text": source_with_tag,
                    "translation": item.get("translation", ""),
                }


                max_length = 128 if model_size == "transformer" else 128
                encoded = tok(
                    source_with_tag,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_offsets_mapping=True
                )
                offsets = list(encoded["offset_mapping"])
                entry.update({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "lang_id": LANG2ID[lang],
                    "offset_mapping": offsets
                })

                if model_size in ["transformer", "xlmr", "xlmr_warm"]:
                    entry["labels"] = encoded["input_ids"].copy()

                if pos_weight > 0 and "UPOS" in item:

                    alias_map = {
                        "_": "X",
                        "N": "NOUN",
                        "V": "VERB",
                    }
                    raw_tags = item["UPOS"].split()
                    pos_tags = [alias_map.get(t, t) for t in raw_tags]


                    pos_label_map = {
                        "NOUN": 0, "VERB": 1, "ADJ": 2, "ADV": 3, "PRON": 4,
                        "DET": 5, "ADP": 6, "NUM": 7, "CONJ": 8, "PART": 9,
                        "INTJ": 10, "PUNCT": 11, "SYM": 12, "X": 13, "PROPN": 14,
                        "AUX": 15, "CCONJ": 16
                    }
                    pos_ids = [pos_label_map.get(t, pos_label_map["X"]) for t in pos_tags]

                    pos_ids.insert(0, -100)
                    pos_ids.insert(0, -100)


                    seq_len = max_length
                    if len(pos_ids) >= seq_len:
                        pos_ids = pos_ids[:seq_len]
                    else:
                        pos_ids += [-100] * (seq_len - len(pos_ids))

                    entry["pos_labels"] = pos_ids


                elif with_translation and not model_size == "transformer":
                    max_total = 128
                    half = (max_total - 1) // 2
                    sep_id = tok.convert_tokens_to_ids("[SEP]")

                    orig_ids = tok(
                        source_with_tag,
                        truncation=True,
                        padding=False,
                        max_length=half
                    )["input_ids"]
                    try:
                        trans_ids = tok(
                            item.get("translation", ""),
                            truncation=True,
                            padding=False,
                            max_length=half
                        )["input_ids"]
                    except Exception:
                        trans_ids = [tok.unk_token_id] * min(half, 5)

                    input_ids = orig_ids + [sep_id] + trans_ids
                    attn_mask = [1] * len(input_ids)
                    pad_len = max_total - len(input_ids)
                    input_ids += [tok.pad_token_id] * pad_len
                    attn_mask += [0] * pad_len

                    entry.update({
                        "input_ids": input_ids,
                        "attention_mask": attn_mask,
                    })

                if os.environ.get("LOSS_MODE") == "english_anchor":
                    entry["text_en"] = item.get("translation", "")



                entries.append(entry)

            datasets_by_split[split].append(Dataset.from_list(entries))

    out = DatasetDict()
    for split in ("train", "validation", "test"):
        out[split] = (
            concatenate_datasets(datasets_by_split[split]) if multi else datasets_by_split[split][0]
        )


    if "train" in out:
        final_train_count = len(out['train'])
        final_valid_count = len(out['validation'])
        final_test_count = len(out['test'])
        final_total_count = final_train_count + final_valid_count + final_test_count

        print("\n" + "=" * 50)
        print("=== Final Aggregated Dataset Statistics ===")
        print(f"  - Total training samples:   {final_train_count}")
        print(f"  - Total validation samples: {final_valid_count}")
        print(f"  - Total test samples:       {final_test_count}")
        print(f"  - Grand total all samples:  {final_total_count}")
        print("=" * 50 + "\n")
    else:
        print("\n[WARN] No data was loaded, final statistics are empty.")

    return out



def compute_metrics(eval_pred):

    if hasattr(eval_pred, "loss"):
        return {"eval_loss": eval_pred.loss}
    return {}



def load_loss_function(loss_type: str):
    return load_loss(loss_type)


import re
from collections import defaultdict

def remove_parentheses(s):

    return re.sub(r"\(.*?\)", "", s).replace(" ", "").strip()

def load_golden_pairs(json_path: str,
                      check_filter: str = "true"
                      ) -> Dict[int, Dict[str, set]]:

    golden_map = defaultdict(lambda: defaultdict(set))

    with open(json_path, encoding="utf‑8") as f:
        data = json.load(f)


    want = {"true"} if check_filter == "true" else \
           {"unsure"} if check_filter == "unsure" else \
           {"true", "unsure"} if check_filter == "both" else \
           {"true", "unsure", "false", ""}

    for item in data:
        if str(item.get("check", "")).lower() not in want:
            continue

        forms = item.get("forms", {})


        for src_lang in ("sahidic", "bohairic", "demotic", "hieroglyphic"):
            src_vals = forms.get(src_lang, [])
            src_vals = [src_vals] if isinstance(src_vals, str) else src_vals

            for src_raw in src_vals:
                src_w = remove_parentheses(src_raw)
                if not src_w:
                    continue

                for tgt_lang in ("sahidic", "bohairic", "demotic", "hieroglyphic"):
                    tgt_vals = forms.get(tgt_lang, [])
                    tgt_vals = [tgt_vals] if isinstance(tgt_vals, str) else tgt_vals

                    for tgt_raw in tgt_vals:
                        tgt_w = remove_parentheses(tgt_raw)
                        if not tgt_w:
                            continue

                        src_id = LANG2ID[src_lang]
                        tgt_id = LANG2ID[tgt_lang]


                        golden_map[src_id][src_w].add((tgt_w, tgt_id))

                        golden_map[tgt_id][tgt_w].add((src_w, src_id))

    return golden_map


def sample_contexts(text: str, target_word: str, window_size: int = 5) -> List[str]:

    contexts = []
    words = text.split()
    for i, word in enumerate(words):
        if remove_parentheses(word) == target_word:
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context = words[start:end]
            contexts.append((context, i - start))
    return contexts

def get_context_batch(
    batch_texts: List[str],
    target_words: List[str],
    window_size: int = 5
) -> Dict[str, List[Tuple[List[str], int]]]:

    context_dict = defaultdict(list)
    for text, word in zip(batch_texts, target_words):
        contexts = sample_contexts(text, word, window_size)
        if contexts:
            context_dict[word].extend(contexts)
    return context_dict


import collections
from typing import List

def compute_bleu_score(predictions: List[str], references: List[str]) -> float:

    def get_ngrams(tokens, n):
        return collections.Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    
    def bleu_n(pred_tokens, ref_tokens, n):
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if len(pred_ngrams) == 0:
            return 0.0
            
        overlap = sum((pred_ngrams & ref_ngrams).values())
        return overlap / len(pred_ngrams)
    
    if not predictions or not references:
        return 0.0
    
    total_bleu = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        if len(pred_tokens) == 0:
            continue
            

        bleu_scores = []
        for n in range(1, 5):
            bleu_scores.append(bleu_n(pred_tokens, ref_tokens, n))

        if all(score > 0 for score in bleu_scores):
            bleu_4 = (bleu_scores[0] * bleu_scores[1] * bleu_scores[2] * bleu_scores[3]) ** 0.25
        else:
            bleu_4 = 0.0
            
        total_bleu += bleu_4
    
    return total_bleu / len(predictions) if predictions else 0.0

def compute_diversity_metrics(texts: List[str]) -> Dict[str, float]:

    all_tokens = []
    all_bigrams = []
    repetition_count = 0
    
    for text in texts:
        tokens = text.split()
        all_tokens.extend(tokens)
        

        for i in range(len(tokens)-1):
            all_bigrams.append((tokens[i], tokens[i+1]))
        

        if len(tokens) >= 3:
            for i in range(len(tokens)-2):
                if tokens[i] == tokens[i+1] == tokens[i+2]:
                    repetition_count += 1
                    break
    
    if not all_tokens:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_rate": 0.0}
    

    distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
    

    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
    

    repetition_rate = repetition_count / len(texts) if texts else 0.0
    
    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2, 
        "repetition_rate": repetition_rate
    }


import collections

import sacrebleu


import torch

import sacrebleu
import random
import torch
from pathlib import Path


def evaluate_translation_quality(
        model,
        eval_dataset,
        tokenizer_dict,
        device,
        max_samples: int | None = 100,
        debug_log_path: str | None = None,
        global_step: int | None = None,
):

    model.eval()


    if not hasattr(model, 'decoder_tokenizer'):
        print("[WARN] Model does not have a 'decoder_tokenizer'. Skipping BLEU evaluation.")
        return {"bleu": 0.0, "num_samples": 0}
    eng_tok = model.decoder_tokenizer

    predictions, references = [], []


    subset_seed = 42
    rng = random.Random(subset_seed)
    if max_samples is not None and len(eval_dataset) > max_samples:
        all_indices = list(range(len(eval_dataset)))
        fixed_indices = rng.sample(all_indices, max_samples)
        eval_subset = eval_dataset.select(fixed_indices)
    else:
        eval_subset = eval_dataset
    print(f"   - Evaluating translation on a FIXED subset of {len(eval_subset)} samples (seed={subset_seed}).")

    debug_lines = []
    skipped_count = 0

    with torch.no_grad():
        for idx, samp in enumerate(eval_subset):
            lang = samp["lang"]
            source_with_tag = samp["text"]


            if "translation_labels" not in samp:
                skipped_count += 1
                continue


            lbl_ids = [t for t in samp["translation_labels"] if t != -100 and t != eng_tok.pad_token_id]
            reference = eng_tok.decode(lbl_ids, skip_special_tokens=True).strip()

            if not reference:
                skipped_count += 1
                continue

            src_tok = tokenizer_dict[lang]
            enc = src_tok(source_with_tag, return_tensors="pt", max_length=128, truncation=True).to(device)

            generated_ids = model.generate(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                lang_id=torch.tensor([LANG2ID[lang]], device=device),
                max_length=50,
                num_beams=4,
                early_stopping=True
            )


            prediction = eng_tok.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            predictions.append(prediction)
            references.append(reference)

            if idx < 8 and debug_log_path:
                original_egyptian_text = source_with_tag.replace(TAG_BY_LANG.get(lang, ""), "").strip()
                debug_lines.extend([
                    "\n" + "=" * 50,
                    f"=== Translation Debug Sample #{idx + 1} ===",
                    f"Language: {lang}",
                    f"Source Text (Original): {original_egyptian_text[:120]}",
                    f"Reference (Ground Truth): {reference}",
                    f"Prediction (Model Output): {prediction}",
                ])


    if skipped_count > 0:
        print(f"  [INFO] Total skipped samples in translation evaluation: {skipped_count}")

    if not predictions:

        return {"bleu": 0.0, "distinct_1": 0.0, "distinct_2": 0.0,
                "repetition_rate": 0.0, "num_samples": 0}


    bleu_score = sacrebleu.corpus_bleu(predictions, [references])
    diversity_metrics = compute_diversity_metrics(predictions)


    if debug_log_path:
        try:
            with open(debug_log_path, "a", encoding="utf-8") as f:
                if global_step is not None:
                    f.write(f"\n\n{'=' * 20} EVALUATION AT STEP {global_step} {'=' * 20}\n")
                if debug_lines:
                    f.write("\n".join(debug_lines) + "\n")
                f.write("\n" + "-" * 50)
                f.write("\n--- Translation Metrics Summary ---\n")
                f.write(f"BLEU-4:      {bleu_score.score:.4f}\n")
                f.write(f"Distinct-1:  {diversity_metrics['distinct_1']:.4f}\n")
                f.write(f"Distinct-2:  {diversity_metrics['distinct_2']:.4f}\n")
                f.write(f"Repetition:  {diversity_metrics['repetition_rate']:.4f}\n")
                f.write(f"Samples:     {len(predictions)}\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            print(f"[Warn] Debug log write fail: {e}")


    return {
        "bleu": bleu_score.score,
        **diversity_metrics,
        "num_samples": len(predictions),
    }



from sklearn.metrics import precision_recall_fscore_support

def evaluate_pos_quality(model, eval_dataset, tokenizer_dict,
                         device, max_samples=100,
                         debug_log_path=None, global_step=None):

    model.eval()
    all_pred, all_gold = [], []
    dbg = []
    with torch.no_grad():
        for i, samp in enumerate(eval_dataset):
            if i >= max_samples: break
            if "pos_labels" not in samp: continue
            gold = samp["pos_labels"]
            if all(g == -100 for g in gold): continue

            tok = tokenizer_dict[samp["lang"]]
            enc = torch.tensor([samp["input_ids"]], device=device)
            att = torch.tensor([samp["attention_mask"]], device=device)
            out = model(input_ids=enc, attention_mask=att,
                        lang_id=torch.tensor([LANG2ID[samp["lang"]]], device=device),
                        return_dict=True)
            logits = out.pos_logits.squeeze(0)[:len(gold)]
            pred   = logits.argmax(-1).cpu().tolist()

            for p, g in zip(pred, gold):
                if g != -100:
                    all_pred.append(p); all_gold.append(g)

            if debug_log_path and i < 5:
                sent = tok.decode(samp["input_ids"], skip_special_tokens=True)
                dbg.extend([
                    "\n=== POS Debug Sample ===",
                    f"Sentence: {sent}",
                    f"Gold : {gold}",
                    f"Pred : {pred}",
                    "="*40,
                ])

    p,r,f1,_ = precision_recall_fscore_support(
        all_gold, all_pred, average="micro", zero_division=0.0)

    if debug_log_path and dbg:
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(dbg))
            f.write(f"\n=== POS Metrics Summary ===\n"
                    f"F1: {f1:.4f}  P: {p:.4f}  R: {r:.4f}\n")

    return {"pos_f1": f1, "pos_precision": p, "pos_recall": r,
            "pos_samples": len(all_gold)}

