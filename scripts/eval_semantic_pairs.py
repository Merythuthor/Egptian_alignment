#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: lrec_eval_bert_all_3.0.py (v3.0, safe, no file-reading changes)
# Description: Final evaluation script for bert_all, based on your original logic.
# Minimal changes: word-boundary regex, symmetric sampling, v2 caches/plots/csv, safety checks.

import os
import json
import random
from collections import defaultdict
from pathlib import Path
import sys
import re
import csv

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from sklearn.metrics import roc_auc_score

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# 高质量导出（文字矢量、不糊）
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['pdf.fonttype'] = 42      # TrueType，避免文字被转曲
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'  # SVG/ PDF 中保留文本


# --- 导入我们的新模型定义（按你的工程路径） ---
sys.path.append(str(Path(".").resolve()))
from models.multi_task_bert_encoder_decoder import MultiTaskBertEncoderDecoder, MultiTaskBertConfig

# —— 与论文一致：无衬线（TeX Gyre Heros / Helvetica 系）——
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['TeX Gyre Heros', 'Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False  # 负号不乱码

# —— 导出更清晰（矢量文字）——
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'

# =================================================================
#               ⭐⭐⭐ 核心配置区 START ⭐⭐⭐
# =================================================================

MODELS_TO_EVALUATE = {
    "BertAll_Exp1_Baseline (ep10)": {
        "path_template": "checkpoints/bert_all_exp3_baseline_new/multi_bert_all_mlm_full_shared_bpe_T0_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoints": [55500],  # 示例步数，请修改为你实际训练结束的步数
        "type": "bert_all"
    },
    "BertAll_Exp_MLM_TLM (ep10)": {
        "path_template": "checkpoints/bert_all_exp2_MLM_TLM/multi_bert_all_mlm_full_shared_bpe_T0_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoints": [55500],  # 示例步数，请修改为你实际训练结束的步数
        "type": "bert_all"
    },
    "BertAll_Exp_MLM_Translation (ep10)": {
        "path_template": "checkpoints/bert_all_exp2_MLM_Translation/multi_bert_all_mlm_full_shared_bpe_T1_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoints": [55500],  # 示例步数，请修改为你实际训练结束的步数
        "type": "bert_all"
    },
    "BertAll_Exp_MLM_TLM_Translation (ep10)": {
        "path_template": "checkpoints/bert_all_exp2_MLM_TLM_Translation/multi_bert_all_mlm_full_shared_bpe_T1_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoints": [55500],  # 示例步数，请修改为你实际训练结束的步数
        "type": "bert_all"
    },
    "BertAll_Exp1_MLM_TLM_Translation_POS (ep10)": {
        "path_template": "checkpoints/bert_all_exp2_balanced_new/multi_bert_all_mlm_full_shared_bpe_T1_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoints": [55500],  # 示例步数，请修改为你实际训练结束的步数
        "type": "bert_all"
    },

    "BertAll_Exp_MLM_Fusion_Alpha_Latin_new (ep10)": {
        "path_template": "checkpoints/bert_all_exp_MLM_fusion_alpha_latin_new_2/multi_bert_all_mlm_full_shared_bpe_T0_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoints": [55500],  # 示例步数，请修改为你实际训练结束的步数
        "type": "bert_all"
    },
    "BertAll_Exp_MLM_KL_Latin_latest (ep10)": {
        "path_template": "checkpoints/bert_all_exp3_MLM_KL_Latin_latest/multi_bert_all_mlm_full_shared_bpe_T0_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoints": [55500],  # 示例步数，请修改为你实际训练结束的步数
        "type": "bert_all"
    },

    "BertAll_Exp_MLM_Fusion_Alpha_IPA (ep10)": {
        "path_template": "checkpoints/bert_all_exp_MLM_fusion_alpha_IPA/multi_bert_all_mlm_full_shared_bpe_T0_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoints": [55500],  # 示例步数，请修改为你实际训练结束的步数
        "type": "bert_all"
    },
    "BertAll_Exp_MLM_KL_IPA (ep10)": {
        "path_template": "checkpoints/bert_all_exp_MLM_KL_ipa/multi_bert_all_mlm_full_shared_bpe_T0_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoints": [55500],  # 示例步数，请修改为你实际训练结束的步数
        "type": "bert_all"
    },
}

# Tokenizer 路径（保持你的写法）
TOKENIZER_PATH = "project_tokenizers/bert_all/tokenizer.json"

# 语料与同源对（保持你的读取方式）
COGNATE_PAIRS_PATH = "resource_eval_new/egyptian_cognate_pairs_without_repetition.jsonl"
CORPUS_DIR = "data/processed_jsonl_UPOS"  # 使用你的 *_rev.jsonl 逻辑

# 结果与缓存（v2 名称，避免覆盖）
RESULTS_CSV_PATH = "results/bert_all_evaluation_final_v3.csv"
CACHE_DIR = "evaluation_cache_bert_all_v_without_repetition"

TOTAL_EVAL_SAMPLES = 20000
CACHE_BUILD_BATCH_SIZE = 32
MAX_SEQ_LEN = 768

# =================================================================
#               ⭐⭐⭐ 核心配置区 END ⭐⭐⭐
# =================================================================

LANGUAGES = ["hieroglyphic", "demotic", "sahidic", "bohairic"]
EGYPTIAN_FAMILY = {"hieroglyphic", "demotic"}
COPTIC_FAMILY = {"sahidic", "bohairic"}

# 导入语言标签，如果失败则使用硬编码（保持你的逻辑）
try:
    from aaai.training.utils import TAG_BY_LANG
except ImportError:
    TAG_BY_LANG = {"hieroglyphic": "<hiero>", "demotic": "<dem>", "bohairic": "<boh>", "sahidic": "<sah>"}

LANG_TAGS = TAG_BY_LANG.copy()
LANG_TAGS["english"] = "<eng>"


# ---------------------------- 工具函数（不改你的数据读取） ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hardcoded_preprocess_text(text: str) -> str:
    if not text: return ""
    text = text.replace('[gap]', '?')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------- 上下文池（保持你的 *_rev.jsonl 逻辑） ----------------------------
def build_context_pool(corpus_dir):
    print("Building context pool from probing corpus (validation and test sets).")
    context_pool = defaultdict(list)
    cache_path = Path(CACHE_DIR) / "context_pool_v2.pt"   # <- v2 避免覆盖
    if cache_path.exists():
        print(f"   - Loading context pool from cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    for lang in LANGUAGES:
        file_path = Path(corpus_dir) / f"{lang}_rev.jsonl"
        if not file_path.exists():
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]
        val_start_index = int(0.8 * len(lines))
        held_out_lines = lines[val_start_index:]
        for item in held_out_lines:
            processed_text = hardcoded_preprocess_text(item.get("text", ""))
            if processed_text:
                for word in set(processed_text.split(' ')):
                    if word:
                        context_pool[(word, lang)].append(processed_text)
            en_text = hardcoded_preprocess_text(item.get("translation", ""))
            if en_text:
                for word in set(en_text.split(' ')):
                    if word:
                        context_pool[(word, "english")].append(en_text)

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(context_pool, cache_path)
    print(f"   - Context pool contains {len(context_pool)} keys. Saved to cache.")
    return context_pool


# ---------------------------- E–E 评测分组（保持你的规则） ----------------------------
def get_pair_group(lang1, lang2):
    if (lang1 in EGYPTIAN_FAMILY and lang2 in COPTIC_FAMILY) or \
       (lang1 in COPTIC_FAMILY and lang2 in EGYPTIAN_FAMILY):
        return "Cross-Branches"
    if (lang1 in EGYPTIAN_FAMILY and lang2 in EGYPTIAN_FAMILY) or \
       (lang1 in COPTIC_FAMILY and lang2 in COPTIC_FAMILY):
        return "Within-Branches"
    return "Other"


def load_and_stratify_pairs(cognate_path):
    print("Loading and stratifying all cognate pairs.")
    with open(cognate_path, 'r', encoding='utf-8') as f:
        all_pairs = [json.loads(line) for line in f]
    stratified_pairs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in all_pairs:
        eval_group = get_pair_group(p['lang1'], p['lang2'])
        form_type = "Homograph" if p['word1'] == p['word2'] else "Heterograph"
        sorted_langs = tuple(sorted((p['lang1'], p['lang2'])))
        lang_pair_key = f"{sorted_langs[0].capitalize()}-{sorted_langs[1].capitalize()}"
        stratified_pairs[eval_group][form_type][lang_pair_key].append(p)
    return stratified_pairs


# ---------------------------- E–EN 评测辅助（保持你的读取/切分方式） ----------------------------
def _iter_split_lines(corpus_dir, split="train"):
    lang2lines = {}
    for lang in LANGUAGES:
        fp = Path(corpus_dir) / f"{lang}_rev.jsonl"
        if not fp.exists():
            continue
        with open(fp, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        N = len(lines)
        k = int(0.8 * N)
        lang2lines[lang] = lines[:k] if split == "train" else lines[k:]
    return lang2lines


def build_seen_set_egyptian_english(corpus_dir, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    seen_cache = cache_dir / "seen_pairs_train_E2EN_v2.json"  # <- v2 避免覆盖
    if seen_cache.exists():
        with open(seen_cache, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        seen = set((eg, en) for eg, en in pairs)
        print(f"   - Loaded E-EN seen-set from cache: {seen_cache.name} (size={len(seen)})")
        return seen
    train_lines = _iter_split_lines(corpus_dir, split="train")
    seen = set()
    for lang, lines in train_lines.items():
        for ex in lines:
            text_words = set(hardcoded_preprocess_text(ex.get("text", "")).split())
            trans_words = set(hardcoded_preprocess_text(ex.get("translation", "")).split())
            for eg_word in text_words:
                for en_word in trans_words:
                    if eg_word and en_word:
                        seen.add((eg_word, en_word))
    with open(seen_cache, "w", encoding="utf-8") as f:
        json.dump(list(seen), f, ensure_ascii=False)
    print(f"   - Saved seen-set to cache: {seen_cache.name} (size={len(seen)})")
    return seen


def build_e2en_eval_pairs(corpus_dir, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs_cache = cache_dir / "e2en_eval_pairs_valtest_v2.json"  # <- v2 避免覆盖
    if pairs_cache.exists():
        with open(pairs_cache, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"   - Loaded E–EN val/test pair lists from cache: {pairs_cache.name}")
        return data, None
    seen_set = build_seen_set_egyptian_english(corpus_dir, cache_dir)
    heldout_lines = _iter_split_lines(corpus_dir, split="heldout")
    lang2english_vocab = defaultdict(set)
    for lang, lines in heldout_lines.items():
        for ex in lines:
            for w in hardcoded_preprocess_text(ex.get("translation", "")).split():
                if len(w) > 1:
                    lang2english_vocab[lang].add(w)
    result = {}
    for lang, lines in heldout_lines.items():
        true_seen, true_unseen, false_pairs = [], [], []
        english_list = sorted(list(lang2english_vocab[lang]))
        if len(english_list) < 2:
            continue
        for ex in lines:
            text_words = set(hardcoded_preprocess_text(ex.get("text", "")).split())
            trans_words = set(hardcoded_preprocess_text(ex.get("translation", "")).split())
            if not text_words or not trans_words:
                continue
            eg_word, en_word = random.choice(list(text_words)), random.choice(list(trans_words))
            rec = {"word1": eg_word, "lang1": lang, "word2": en_word, "lang2": "english"}
            if (eg_word, en_word) in seen_set:
                true_seen.append(rec)
            else:
                true_unseen.append(rec)
            neg_en = en_word
            for _ in range(5):
                alt = random.choice(english_list)
                if alt != en_word:
                    neg_en = alt
                    break
            false_pairs.append({"word1": eg_word, "lang1": lang, "word2": neg_en, "lang2": "english"})
        result[lang] = {"true_seen": true_seen, "true_unseen": true_unseen, "false_pairs": false_pairs}
    with open(pairs_cache, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"   - Saved E–EN val/test pair lists to cache: {pairs_cache.name}")
    return result, seen_set


# ---------------------------- 批量向量池缓存（只改正则与缓存名） ----------------------------
@torch.no_grad()
def build_vector_pool_cache_batched(model, tokenizer, device, context_pool, words_to_cache):
    vector_pool = defaultdict(list)
    print(f"   - Building vector pool cache for {len(words_to_cache)} required words.")

    encoder = model.get_encoder()
    encoder.eval()

    for (word, lang) in tqdm(words_to_cache, desc="     Caching required words"):
        contexts = context_pool.get((word, lang), [])
        if not contexts:
            continue

        lang_tag = LANG_TAGS.get(lang, "")
        if not lang_tag:
            continue

        tag_prefix = f"{lang_tag} "
        tag_char_len = len(tag_prefix)

        for i in range(0, len(contexts), CACHE_BUILD_BATCH_SIZE):
            raw_batch_texts = contexts[i: i + CACHE_BUILD_BATCH_SIZE]
            tagged_batch_texts = [tag_prefix + text for text in raw_batch_texts]

            encoding = tokenizer(
                tagged_batch_texts, return_tensors="pt", return_offsets_mapping=True,
                truncation=True, padding="longest", max_length=MAX_SEQ_LEN
            ).to(device)

            try:
                outputs = encoder(
                    input_ids=encoding.input_ids,
                    attention_mask=encoding.attention_mask,
                    return_dict=True
                )
                last_hidden_states = outputs.last_hidden_state.detach().cpu()
                offsets = encoding.offset_mapping.cpu()

                for j, raw_text in enumerate(raw_batch_texts):
                    sent_offsets = offsets[j]
                    sent_hidden = last_hidden_states[j]

                    try:
                        # ★ 修改：加入词边界，避免子串误命中
                        for match in re.finditer(r"\b{}\b".format(re.escape(word)), raw_text):
                            char_start, char_end = match.span()
                            adj_start, adj_end = char_start + tag_char_len, char_end + tag_char_len

                            token_indices = [
                                tok_idx for tok_idx, (off_start, off_end) in enumerate(sent_offsets)
                                if off_start != off_end and max(off_start, adj_start) < min(off_end, adj_end)
                            ]
                            if token_indices:
                                vector_pool[(word, lang)].append(sent_hidden[token_indices].mean(dim=0))
                                break
                    except Exception:
                        continue
            except RuntimeError as e:
                print(f"  | [Warn] Skipping batch due to error: {e}")
                torch.cuda.empty_cache()

    final_pool = {}
    for key, vecs in vector_pool.items():
        if vecs:
            final_pool[key] = torch.stack(vecs).mean(dim=0)
    return final_pool


# ---------------------------- 评测逻辑（只改为对称采样） ----------------------------
def evaluate_with_instance_sampling(true_pairs, false_pairs, vector_pool):
    def generate_scores(pair_list):
        scores, valid_count = [], 0
        for p in pair_list:
            if (p['word1'], p['lang1']) in vector_pool and (p['word2'], p['lang2']) in vector_pool:
                valid_count += 1
                v1 = vector_pool[(p['word1'], p['lang1'])]
                v2 = vector_pool[(p['word2'], p['lang2'])]
                scores.append(F.cosine_similarity(v1, v2, dim=0).item())
        return scores, valid_count

    # ★ 对称随机采样（仍保留上限）
    n = min(TOTAL_EVAL_SAMPLES, len(true_pairs), len(false_pairs))
    if n <= 0:
        return {"AUC-ROC": 0.5, "Triplet Accuracy": 0.5, "Valid Pairs": 0}

    sampled_true = random.sample(true_pairs, n)
    sampled_false = random.sample(false_pairs, n)

    sims_true, num_valid_true = generate_scores(sampled_true)
    sims_false, num_valid_false = generate_scores(sampled_false)
    valid_pairs = min(num_valid_true, num_valid_false)

    if not sims_true or not sims_false:
        return {"AUC-ROC": 0.5, "Triplet Accuracy": 0.5, "Valid Pairs": valid_pairs}

    y_true_auc = [1] * len(sims_true) + [0] * len(sims_false)
    y_scores_auc = sims_true + sims_false
    try:
        auc_roc = roc_auc_score(y_true_auc, y_scores_auc)
    except ValueError:
        auc_roc = 0.5

    m = min(len(sims_true), len(sims_false))
    triplet_accuracy = sum(1 for i in range(m) if sims_true[i] > sims_false[i]) / m if m > 0 else 0.0

    return {"AUC-ROC": float(auc_roc), "Triplet Accuracy": float(triplet_accuracy), "Valid Pairs": int(valid_pairs)}


# =================================================================
#                               主流程
# =================================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1) Tokenizer（保持你的加载方式）
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token = "[PAD]", "[CLS]", "[SEP]"
    vocab_size = len(tokenizer)

    # 2) 数据（保持你的读取逻辑）
    context_pool = build_context_pool(CORPUS_DIR)
    stratified_true_ee = load_and_stratify_pairs(COGNATE_PAIRS_PATH)
    e2en_pairs_by_lang, _ = build_e2en_eval_pairs(CORPUS_DIR, CACHE_DIR)

    # 3) 为 E-E 构造负例（保持你的旧逻辑）
    stratified_false_ee = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for group, forms in stratified_true_ee.items():
        for form, lang_pairs in forms.items():
            for lp, pairs in lang_pairs.items():
                all_w2 = [p['word2'] for p in pairs]
                for p in pairs:
                    if len(all_w2) > 1:
                        neg_w2 = p['word2']
                        while neg_w2 == p['word2']:
                            neg_w2 = random.choice(all_w2)
                        stratified_false_ee[group][form][lp].append(
                            {"word1": p['word1'], "lang1": p['lang1'], "word2": neg_w2, "lang2": p['lang2']}
                        )

    # 4) 收集需要缓存的词（保持逻辑）
    all_words_needed = set()
    for groups in (stratified_true_ee, stratified_false_ee):
        for g in groups.values():
            for f in g.values():
                for pairs in f.values():
                    for p in pairs:
                        all_words_needed.add((p['word1'], p['lang1']))
                        all_words_needed.add((p['word2'], p['lang2']))
    for lang, d in e2en_pairs_by_lang.items():
        for k in d:
            for p in d[k]:
                all_words_needed.add((p['word1'], p['lang1']))
                all_words_needed.add((p['word2'], p['lang2']))
    print(f"\nFound {len(all_words_needed)} unique (word, lang) keys required for evaluation.")

    # 5) 输出 CSV（v2）
    Path(RESULTS_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(RESULTS_CSV_PATH, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "model_name", "checkpoint", "evaluation_group", "form_type",
        "language_pair", "auc_roc", "triplet_accuracy", "valid_pairs"
    ])

    # 6) 遍历模型评测
    for model_name, config in MODELS_TO_EVALUATE.items():
        for checkpoint in config["checkpoints"]:
            ckpt_path = Path(config["path_template"].format(checkpoint))
            if not (ckpt_path / "pytorch_model.bin").exists():
                print(f"[Skip] Checkpoint not found: {ckpt_path / 'pytorch_model.bin'}")
                continue

            print(f"\n{'=' * 80}\n--- Evaluating Model: {model_name} | Checkpoint: {checkpoint} ---")

            # 模型（保持你的构造方式 + 安全检查）
            model_config = MultiTaskBertConfig(vocab_size=vocab_size, max_position_embeddings=MAX_SEQ_LEN)
            model = MultiTaskBertEncoderDecoder(model_config)
            state_dict = torch.load(ckpt_path / "pytorch_model.bin", map_location="cpu")

            emb_key = "encoder.embeddings.word_embeddings.weight"
            if emb_key in state_dict:
                ckpt_vocab = state_dict[emb_key].shape[0]
                if ckpt_vocab != vocab_size:
                    raise ValueError(f"[Vocab mismatch] ckpt={ckpt_vocab}, tokenizer={vocab_size}. "
                                     f"Ensure the tokenizer matches training-time vocab.")

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[WARN] load_state_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}")

            model.to(device)

            # 向量池缓存（v2 文件名，避免覆盖）
            sane_model_name = model_name.replace(' ', '_').replace(':', '')
            sig = f"{'-'.join(sorted(LANGUAGES))}_{Path(CORPUS_DIR).name}"
            cache_file = Path(CACHE_DIR) / f"{sane_model_name}_{checkpoint}_{sig}_vecs_v2.pt"

            if cache_file.exists():
                print(f"Loading vector pool from cache: {cache_file}")
                vector_pool = torch.load(cache_file, weights_only=False)
            else:
                vector_pool = build_vector_pool_cache_batched(
                    model, tokenizer, device, context_pool, list(all_words_needed)
                )
                Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
                torch.save(vector_pool, cache_file)
                print(f"Vector pool saved to: {cache_file}")

            del model
            torch.cuda.empty_cache()

            # ---- E-E 评估
            print("\n--- Running E-E Evaluation ---")
            for eval_group, form_types in sorted(stratified_true_ee.items()):
                for form_type, lang_pairs in sorted(form_types.items()):
                    for lang_pair, true_pairs in sorted(lang_pairs.items()):
                        false_pairs = stratified_false_ee.get(eval_group, {}).get(form_type, {}).get(lang_pair, [])
                        if not true_pairs or not false_pairs:
                            continue

                        print(f"   - Evaluating group: [{eval_group} / {form_type} / {lang_pair}].")
                        results = evaluate_with_instance_sampling(true_pairs, false_pairs, vector_pool)
                        print(f"     -> AUC: {results['AUC-ROC']:.4f}, Acc: {results['Triplet Accuracy']:.4f}, "
                              f"Valid Pairs: {results['Valid Pairs']}")
                        csv_writer.writerow([
                            model_name, checkpoint, eval_group, form_type, lang_pair,
                            results['AUC-ROC'], results['Triplet Accuracy'], results['Valid Pairs']
                        ])
                        csv_file.flush()

            # ---- E-EN 评估
            print("\n--- Running E-EN Evaluation ---")
            for lang in LANGUAGES:
                if lang not in e2en_pairs_by_lang:
                    continue
                buckets = e2en_pairs_by_lang[lang]
                for flag in ["seen", "unseen"]:
                    true_pairs = buckets.get(f"true_{flag}", [])
                    false_pairs = buckets.get("false_pairs", [])
                    if not true_pairs or not false_pairs:
                        continue

                    group, form, lp = "E2EN-Alignment", flag.capitalize(), f"{lang.capitalize()}-English"
                    print(f"   - Evaluating E-EN: [{lp} / {form}]")
                    results = evaluate_with_instance_sampling(true_pairs, false_pairs, vector_pool)
                    print(f"     -> AUC: {results['AUC-ROC']:.4f}, Acc: {results['Triplet Accuracy']:.4f}, "
                          f"Valid Pairs: {results['Valid Pairs']}")
                    csv_writer.writerow([
                        model_name, checkpoint, group, form, lp,
                        results['AUC-ROC'], results['Triplet Accuracy'], results['Valid Pairs']
                    ])
                    csv_file.flush()

    csv_file.close()
    print(f"\n✅ All evaluation results have been saved to {RESULTS_CSV_PATH}")

    # 绘图（折线图版本）
    _plot_results(RESULTS_CSV_PATH)


def _plot_results(csv_path: str, out_dir: str = "results"):
    """
    并排两面板热力图（左保留 y 轴，右隐藏 y 轴；正方格）。
    - 颜色映射：AUC
    - 单元格文字：显示 "AUC" 和 "(Acc)" 两行
    - 行内按 AUC 的最大值：数字加粗 + 同色白底高亮
    输出：
      - EE_heatmap_groups_auc_acc.png
      - EEN_heatmap_groups_auc_acc.png
    """
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    CMAP_NAME = 'YlGnBu'

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # ---------------- 固定两组模型 & 顺序 ----------------
    group1_models = [
        "BertAll_Exp1_Baseline (ep10)",
        "BertAll_Exp_MLM_TLM (ep10)",
        "BertAll_Exp_MLM_Translation (ep10)",
        "BertAll_Exp_MLM_TLM_Translation (ep10)",
        "BertAll_Exp1_MLM_TLM_Translation_POS (ep10)",
    ]
    group2_models = [
        "BertAll_Exp1_Baseline (ep10)",
        "BertAll_Exp_MLM_Fusion_Alpha_Latin_new (ep10)",
        "BertAll_Exp_MLM_KL_Latin_latest (ep10)",
        "BertAll_Exp_MLM_Fusion_Alpha_IPA (ep10)",
        "BertAll_Exp_MLM_KL_IPA (ep10)",
    ]
    short_label = {
        "BertAll_Exp1_Baseline (ep10)": "baseline MLM",
        "BertAll_Exp_MLM_TLM (ep10)": "MLM+TLM",
        "BertAll_Exp_MLM_Translation (ep10)": "MLM+Trans",
        "BertAll_Exp_MLM_TLM_Translation (ep10)": "MLM+TLM+Trans",
        "BertAll_Exp1_MLM_TLM_Translation_POS (ep10)": "MLM+TLM+Trans+POS",
        "BertAll_Exp_MLM_Fusion_Alpha_Latin_new (ep10)": "MLM Fusion Latin",
        "BertAll_Exp_MLM_KL_Latin_latest (ep10)": "MLM KL Latin",
        "BertAll_Exp_MLM_Fusion_Alpha_IPA (ep10)": "MLM Fusion IPA",
        "BertAll_Exp_MLM_KL_IPA (ep10)": "MLM KL IPA",
    }
    order_g1 = ["baseline MLM","MLM+TLM","MLM+Trans","MLM+TLM+Trans","MLM+TLM+Trans+POS"]
    order_g2 = ["baseline MLM","MLM Fusion Latin","MLM KL Latin","MLM Fusion IPA","MLM KL IPA"]

    # ---------------- 紧凑缩写（English->E，Seen->S，Unseen->UnS） ----------------
    def _shorten_label(s: str) -> str:
        s = str(s)
        repl = [
            ("cross-branches","C"), ("within-branches","I"),
            ("heterograph","Ht"),  ("homograph","Ho"),
            ("demotic","D"), ("hieroglyphic","H"),
            ("sahidic","S"), ("bohairic","B"),
            ("english","E"), ("English","E"),
            ("Seen","S"), ("seen","S"),
            ("Unseen","UnS"), ("unseen","UnS"),
            ("Cross-Branches","C"), ("Within-Branches","I"),
            ("Heterograph","Ht"),  ("Homograph","Ho"),
            ("Demotic","D"), ("Hieroglyphic","H"),
            ("Sahidic","S"), ("Bohairic","B"),
            (" / ", "/"),
        ]
        for a,b in repl:
            s = s.replace(a, b)
        return s

    # ---------------- 面板数据：行=评估项，列=模型；支持传入自定义行顺序 ----------------
    def _prep_panel(frame: pd.DataFrame, model_keys: list, is_ee: bool, model_order: list, row_order: list | None = None):
        f = frame[frame["model_name"].isin(model_keys)].copy()
        f["model_key"] = f["model_name"].map(lambda x: short_label.get(x, x))
        if is_ee:
            f["row_key"] = "[" + f["evaluation_group"].astype(str) + "/" \
                             + f["form_type"].astype(str) + "/" \
                             + f["language_pair"].astype(str) + "]"
        else:
            f["row_key"] = f["language_pair"].astype(str) + "/" + f["form_type"].astype(str)
        f["row_key"] = f["row_key"].apply(_shorten_label)

        piv_auc = f.pivot_table(index="row_key", columns="model_key", values="auc_roc", aggfunc="max")
        piv_acc = f.pivot_table(index="row_key", columns="model_key", values="triplet_accuracy", aggfunc="max")

        # 固定列顺序
        piv_auc = piv_auc.reindex(columns=model_order)
        piv_acc = piv_acc.reindex(columns=model_order)

        # 固定行顺序：优先使用 row_order（精确匹配），缺的自动忽略；未列出的其余行附在末尾
        if row_order is not None:
            existing = [r for r in row_order if r in piv_auc.index]
            remaining = [r for r in piv_auc.index if r not in existing]
            final_rows = existing + remaining
        else:
            final_rows = sorted(set(piv_auc.index.tolist()) | set(piv_acc.index.tolist()))

        piv_auc = piv_auc.reindex(index=final_rows)
        piv_acc = piv_acc.reindex(index=final_rows)
        return piv_auc, piv_acc

    # ---------------- 画并排两面板（左保留 y，右隐藏 y） ----------------
    def _draw_two_panels(fig_title: str,
                         left_auc: pd.DataFrame,  right_auc: pd.DataFrame,
                         left_acc: pd.DataFrame,  right_acc: pd.DataFrame,
                         left_xlabel: str, right_xlabel: str, outfile: str):
        vmin, vmax = 0.4, 1.0
        cell = 0.65
        n_rows = max(left_auc.shape[0], right_auc.shape[0])
        n_cols = max(left_auc.shape[1], right_auc.shape[1])
        width = cell * (n_cols * 2 + 3.0)
        height = cell * (n_rows + 3)

        fig = plt.figure(figsize=(width, height), dpi=300)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)

        def _panel(ax, M_auc: pd.DataFrame, M_acc: pd.DataFrame, title: str, xlabel: str, show_y: bool):
            M_acc = M_acc.reindex(index=M_auc.index, columns=M_auc.columns)
            data = M_auc.values.astype(float)
            im = ax.imshow(
                data,
                cmap=plt.cm.get_cmap(CMAP_NAME),
                vmin=vmin, vmax=vmax,
                interpolation="nearest", aspect="equal"
            )

            for i in range(M_auc.shape[0]):
                row_auc = data[i, :]
                if np.all(np.isnan(row_auc)):
                    continue
                row_max = np.nanmax(row_auc)
                for j in range(M_auc.shape[1]):
                    auc_val = M_auc.iloc[i, j]
                    acc_val = M_acc.iloc[i, j]
                    auc_txt = "--" if pd.isna(auc_val) else f"{auc_val:.2f}"
                    acc_txt = "--" if pd.isna(acc_val) else f"{acc_val:.2f}"

                    is_max = (not pd.isna(auc_val)) and np.isclose(auc_val, row_max, atol=1e-9)

                    use_white_auc = (not pd.isna(auc_val)) and (auc_val >= 0.80)
                    auc_text_color = "white" if use_white_auc else "black"
                    if is_max:
                        auc_text_color = "black"

                    auc_kwargs = dict(
                        ha="center", va="center",
                        fontsize=11.0,
                        fontweight="bold" if is_max else "normal",
                        color=auc_text_color
                    )
                    if is_max:
                        auc_kwargs["bbox"] = dict(
                            boxstyle="round,pad=0.12,rounding_size=0.06",
                            facecolor="white", edgecolor="black", linewidth=0.9
                        )
                    ax.text(j, i - 0.10, auc_txt, **auc_kwargs)

                    use_white_acc = (not pd.isna(auc_val)) and (auc_val >= 0.80)
                    acc_text_color = "white" if use_white_acc else "black"
                    ax.text(j, i + 0.20, f"({acc_txt})",
                            ha="center", va="center",
                            fontsize=10.0, color=acc_text_color)

            ax.set_title(title, fontsize=17, fontweight="semibold", pad=10)
            ax.set_xticks(np.arange(M_auc.shape[1]))
            ax.set_xticklabels(M_auc.columns.tolist(), rotation=15, ha="right", fontsize=11)

            ax.set_yticks(np.arange(M_auc.shape[0]))
            if show_y:
                ax.set_yticklabels(M_auc.index.tolist(), fontsize=11)
                ax.yaxis.set_visible(True)
                ax.tick_params(axis="y", which="both", left=True, labelleft=True)
            else:
                ax.set_yticklabels([])
                ax.yaxis.set_visible(False)
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)

            ax.set_xlabel(xlabel, fontsize=10)
            ax.grid(False)
            return im

        axL = plt.subplot(gs[0])
        imL = _panel(axL, left_auc, left_acc, "Task Ablation", left_xlabel, show_y=True)

        axR = plt.subplot(gs[1])
        imR = _panel(axR, right_auc, right_acc, "Normalization Ablation", right_xlabel, show_y=False)
        axR.set_ylim(axL.get_ylim())
        axR.set_yticks(axL.get_yticks())

        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cb = fig.colorbar(imL, cax=cbar_ax)
        cb.set_label("AUC (0.4–1.0)", fontsize=10)

        _base = Path(out_dir) / Path(outfile).with_suffix('').name
        plt.savefig(f"{_base}.pdf", bbox_inches="tight")
        plt.savefig(f"{_base}.png", dpi=600, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {os.path.join(out_dir, outfile)}")

    # ===================== E–E（带方括号的行键！）=====================
    df_ee = df[~df["evaluation_group"].str.contains("E2EN", na=False)].copy()
    row_order_ee = [
        "[I/Ho/D-H]", "[I/Ho/B-S]", "[I/Ht/D-H]", "[I/Ht/B-S]",
        "[C/Ht/H-S]", "[C/Ht/D-S]", "[C/Ht/B-D]", "[C/Ht/B-H]"
    ]
    left_auc_ee,  left_acc_ee  = _prep_panel(df_ee,  group1_models, is_ee=True,  model_order=order_g1, row_order=row_order_ee)
    right_auc_ee, right_acc_ee = _prep_panel(df_ee,  group2_models, is_ee=True,  model_order=order_g2, row_order=row_order_ee)
    _draw_two_panels("Egyptian–Egyptian (AUC / Acc)",
                     left_auc_ee, right_auc_ee, left_acc_ee, right_acc_ee,
                     left_xlabel="",
                     right_xlabel="",
                     outfile="EE_heatmap_groups_auc_acc.png")

    # ===================== E–EN ====================
    df_een = df[df["evaluation_group"].str.contains("E2EN", na=False)].copy()
    row_order_een = [
        "H-E/UnS", "H-E/S",
        "D-E/UnS", "D-E/S",
        "S-E/UnS", "S-E/S",
        "B-E/UnS", "B-E/S"
    ]
    left_auc_een,  left_acc_een  = _prep_panel(df_een,  group1_models, is_ee=False, model_order=order_g1, row_order=row_order_een)
    right_auc_een, right_acc_een = _prep_panel(df_een,  group2_models, is_ee=False, model_order=order_g2, row_order=row_order_een)
    _draw_two_panels("Egyptian–English (AUC / Acc)",
                     left_auc_een, right_auc_een, left_acc_een, right_acc_een,
                     left_xlabel="",
                     right_xlabel="",
                     outfile="EEN_heatmap_groups_auc_acc.png")


if __name__ == "__main__":
    main()
