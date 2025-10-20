

import os
import json
import random
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
from sklearn.manifold import TSNE
from transformers import PreTrainedTokenizerFast
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]

LANGUAGES_EGYPT = ["hieroglyphic", "demotic", "sahidic", "bohairic"]
LANGUAGES_WITH_EN = ["hieroglyphic", "demotic", "sahidic", "bohairic", "english"]

FIG_A_EGYPT_ONLY = {
    "MLM baseline": {
        "path_template": "BASE_DIR/checkpoints/bert_all_exp3_baseline_new/multi_bert_all_mlm_full_shared_bpe_T0_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoint": 55500,
    },
    "MLM+TLM+Trans+POS": {
        "path_template": "BASE_DIR/checkpoints/bert_all_exp2_balanced_new/multi_bert_all_mlm_full_shared_bpe_T1_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoint": 55500,
    },

}

FIG_B_WITH_ENGLISH = {
    "MLM baseline": {
        "path_template": "BASE_DIR/checkpoints/bert_all_exp3_baseline_new/multi_bert_all_mlm_full_shared_bpe_T0_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoint": 55500,
    },
    "KL-Latin": {
        "path_template": "BASE_DIR/checkpoints/bert_all_exp3_MLM_KL_Latin_latest/multi_bert_all_mlm_full_shared_bpe_T0_SC0_TA0_M1_epoch10_lr5e-05_b16/checkpoint-{}",
        "checkpoint": 55500,
    },
}

TOKENIZER_PATH = "BASE_DIR/project_tokenizers/bert_all/tokenizer.json"
CORPUS_DIR = "BASE_DIR/data/processed_jsonl"
CACHE_DIR = "BASE_DIR/evaluation_cache_bert_all"
TSNE_OUT_DIR = Path("BASE_DIR/results/tsne"); TSNE_OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 768
CACHE_BUILD_BATCH_SIZE = 32
RANDOM_SEED = 42


PER_LANG_WORDS_EGYPT = 500
PER_LANG_WORDS_WITH_EN = 400
MIN_OCCUR_HELDOUT = 3


TSNE_PERPLEXITY = 30
TSNE_N_ITER = 2000
TSNE_INIT = "pca"

try:
    from training.utils import TAG_BY_LANG
except ImportError:
    TAG_BY_LANG = {"hieroglyphic": "<hiero>", "demotic": "<dem>", "bohairic": "<boh>", "sahidic": "<sah>"}
LANG_TAGS = TAG_BY_LANG.copy()
LANG_TAGS["english"] = "<eng>"


import sys
sys.path.append(str(Path("..").resolve()))
from models.multi_task_bert_encoder_decoder import MultiTaskBertEncoderDecoder, MultiTaskBertConfig



def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess(text: str) -> str:
    if not text: return ""
    text = text.replace("[gap]", "?")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def split_idx(n: int):
    n_train = int(0.8 * n); n_val = int(0.1 * n)
    return 0, n_train, n_train, n_train + n_val, n_train + n_val, n

def sane_name(s: str) -> str:
    return s.replace(" ", "_").replace(":", "").replace("/", "_").replace("\\", "_")


def build_context_pool_heldout(corpus_dir: str, include_english: bool):

    print(f"💾 Building held-out context pool (include_english={include_english}).")
    suffix = "_withEN" if include_english else "_woEN"
    cache_path = Path(CACHE_DIR) / f"context_pool_heldout_v2{suffix}.pt"
    if cache_path.exists():
        print(f"   - Loading from cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    pool = defaultdict(list)
    for lang in LANGUAGES_EGYPT:
        fp = Path(corpus_dir) / f"{lang}_rev.jsonl"
        if not fp.exists():
            print(f"[Warn] Missing {fp}")
            continue
        lines = list(load_jsonl(fp))
        n = len(lines)
        s1,e1,s2,e2,s3,e3 = split_idx(n)
        heldout = lines[e1:e3]

        for item in heldout:

            text = preprocess(item.get("text", ""))
            if text:
                for w in set(text.split()):
                    pool[(w, lang)].append(text)


            if include_english:
                en = preprocess(item.get("translation", ""))
                if en:
                    for w in set(en.split()):
                        pool[(w, "english")].append(en)

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(pool, cache_path)
    print(f"   - Held-out context keys: {len(pool)}  (saved)")
    return pool


@torch.no_grad()
def build_vector_pool_cache_batched(model, tokenizer, device, context_pool, words_to_cache):
    vector_pool = defaultdict(list)
    print(f"🧠 Building vector pool for {len(words_to_cache)} words ...")

    encoder = model.get_encoder()
    encoder.eval()

    for (word, lang) in tqdm(words_to_cache, desc="   Caching word vectors"):
        contexts = context_pool.get((word, lang), [])
        if not contexts: continue

        lang_tag = LANG_TAGS.get(lang, "")
        if not lang_tag: continue
        tag_prefix = f"{lang_tag} "
        tag_char_len = len(tag_prefix)

        for i in range(0, len(contexts), CACHE_BUILD_BATCH_SIZE):
            raw_batch = contexts[i:i+CACHE_BUILD_BATCH_SIZE]
            tagged = [tag_prefix + t for t in raw_batch]

            enc = tokenizer(
                tagged, return_tensors="pt", return_offsets_mapping=True,
                truncation=True, padding="longest", max_length=MAX_SEQ_LEN
            ).to(device)

            out = encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask, return_dict=True)
            last_hidden = out.last_hidden_state.detach().cpu()
            offsets = enc.offset_mapping.cpu()

            for j, raw_text in enumerate(raw_batch):
                sent_offsets = offsets[j]
                sent_hidden = last_hidden[j]
                try:

                    for match in re.finditer(r"\b{}\b".format(re.escape(word)), raw_text):
                        cs, ce = match.span()
                        adj_s, adj_e = cs + tag_char_len, ce + tag_char_len
                        token_idx = [
                            t for t, (os, oe) in enumerate(sent_offsets)
                            if os != oe and max(os, adj_s) < min(oe, adj_e)
                        ]
                        if token_idx:
                            vector_pool[(word, lang)].append(sent_hidden[token_idx].mean(dim=0))
                            break
                except Exception:
                    continue

    final_pool = {}
    for key, vecs in vector_pool.items():
        if vecs:
            final_pool[key] = torch.stack(vecs).mean(dim=0)
    return final_pool



def sample_words_for_tsne(context_pool, languages, per_lang, min_occur, seed=RANDOM_SEED):
    rng = random.Random(seed)
    picks = []
    for lang in languages:
        cands = [(w, lang) for (w, l) in context_pool.keys()
                 if l == lang and len(context_pool[(w, l)]) >= min_occur]
        cands_sorted = sorted(cands, key=lambda k: len(context_pool[k]), reverse=True)
        if len(cands_sorted) > per_lang:
            head = cands_sorted[: per_lang * 10]
            picks_lang = rng.sample(head, per_lang)
        else:
            picks_lang = cands_sorted
        print(f"   - {lang}: candidates≥{min_occur} = {len(cands_sorted)}; picked = {len(picks_lang)}")
        picks.extend(picks_lang)
    return picks


def run_tsne(X, seed=RANDOM_SEED):

    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        init=TSNE_INIT,
        learning_rate="auto",
        n_iter=TSNE_N_ITER,
        verbose=1,
        random_state=seed,
        metric="euclidean"
    )
    return tsne.fit_transform(X)


def embed_for_model(model_title, path_template, checkpoint, tokenizer, device,
                    context_pool, picks, cache_sig):
    vocab_size = len(tokenizer)

    ckpt_dir = Path(path_template.format(checkpoint))
    ckpt_file = ckpt_dir / "pytorch_model.bin"
    if not ckpt_file.exists():
        print(f"[Skip] missing checkpoint: {ckpt_file}")
        return None, None, None

    print(f"\n=== {model_title} @ {checkpoint} ===")

    model_config = MultiTaskBertConfig(vocab_size=vocab_size, max_position_embeddings=MAX_SEQ_LEN)
    model = MultiTaskBertEncoderDecoder(model_config).to(device)
    state = torch.load(ckpt_file, map_location="cpu")

    emb_key = "encoder.embeddings.word_embeddings.weight"
    if emb_key in state and state[emb_key].shape[0] != vocab_size:
        raise ValueError(f"[Vocab mismatch] ckpt={state[emb_key].shape[0]} vs tokenizer={vocab_size}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()


    vec_cache = Path(CACHE_DIR) / f"{sane_name(model_title)}_{checkpoint}_{cache_sig}_tsne_vecs_v2.pt"
    if vec_cache.exists():
        print(f"   - Loading vectors from cache: {vec_cache}")
        vector_pool = torch.load(vec_cache, weights_only=False)
    else:
        vector_pool = build_vector_pool_cache_batched(model, tokenizer, device, context_pool, picks)
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        torch.save(vector_pool, vec_cache)
        print(f"   - Saved vectors to: {vec_cache}")


    X, y = [], []
    kept = Counter()
    for key in picks:
        if key in vector_pool:
            v = vector_pool[key].numpy()
            X.append(v); y.append(key[1]); kept[key[1]] += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(X) < 10:
        print("[Abort] Too few vectors collected.")
        return None, None, None

    X = np.stack(X, axis=0)
    X2 = run_tsne(X)
    print("   - Final kept per language:", dict(kept))
    print(f"   - Total points = {len(X)}")
    return X2, y, kept



COLOR_MAP = {
    "hieroglyphic": "#1f77b4",
    "demotic": "#ff7f0e",
    "sahidic": "#2ca02c",
    "bohairic": "#d62728",
    "english": "#7f7f7f",
}

def scatter_ax(ax, X2, y, languages, title):
    for lang in languages:
        idx = [i for i, l in enumerate(y) if l == lang]
        if not idx: continue

        ax.scatter(X2[idx, 0], X2[idx, 1], s=9, alpha=0.70, linewidths=0,
                   label=lang.capitalize(), c=COLOR_MAP.get(lang, None),
                   rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])

    ax.set_title(title, fontsize=17, fontweight="semibold", pad=12)

    ax.legend(frameon=False, markerscale=2.8, fontsize=15, labelspacing=0.4, borderaxespad=0.6)


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    print(f"🔤 Loading tokenizer: {TOKENIZER_PATH}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.pad_token or "[PAD]"
    tokenizer.cls_token = tokenizer.cls_token or "[CLS]"
    tokenizer.sep_token = tokenizer.sep_token or "[SEP]"


    print("\n===== Figure A: Egypt-only (4 langs) =====")
    ctx_pool_A = build_context_pool_heldout(CORPUS_DIR, include_english=False)
    picks_A = sample_words_for_tsne(
        ctx_pool_A, LANGUAGES_EGYPT, per_lang=PER_LANG_WORDS_EGYPT, min_occur=MIN_OCCUR_HELDOUT, seed=RANDOM_SEED
    )
    cache_sig_A = f"{'-'.join(sorted(LANGUAGES_EGYPT))}_heldout_woEN"


    figA, axesA = plt.subplots(1, 2, figsize=(11.5, 4.8))
    titlesA = []
    for i, (model_title, cfg) in enumerate(FIG_A_EGYPT_ONLY.items()):
        X2, y, kept = embed_for_model(
            model_title, cfg["path_template"], cfg["checkpoint"], tokenizer, device,
            ctx_pool_A, picks_A, cache_sig_A
        )
        if X2 is None: continue
        scatter_ax(axesA[i], X2, y, LANGUAGES_EGYPT, model_title)
        titlesA.append(model_title)

    figA.tight_layout()
    outA = TSNE_OUT_DIR / "tsne_egypt_only"
    figA.savefig(f"{outA}.pdf", bbox_inches="tight")
    figA.savefig(f"{outA}.png", dpi=600, bbox_inches="tight")
    plt.close(figA)
    print(f"📈 Saved Figure A: {outA}")

    print("\n===== Figure B: With-English (5 langs) =====")
    ctx_pool_B = build_context_pool_heldout(CORPUS_DIR, include_english=True)
    picks_B = sample_words_for_tsne(
        ctx_pool_B, LANGUAGES_WITH_EN, per_lang=PER_LANG_WORDS_WITH_EN, min_occur=MIN_OCCUR_HELDOUT, seed=RANDOM_SEED
    )
    cache_sig_B = f"{'-'.join(sorted(LANGUAGES_WITH_EN))}_heldout_withEN"

    figB, axesB = plt.subplots(1, 2, figsize=(12.6, 5.4))
    for ax in axesB: ax.set_aspect('equal', adjustable='datalim')
    for i, (model_title, cfg) in enumerate(FIG_B_WITH_ENGLISH.items()):
        X2, y, kept = embed_for_model(
            model_title, cfg["path_template"], cfg["checkpoint"], tokenizer, device,
            ctx_pool_B, picks_B, cache_sig_B
        )
        if X2 is None: continue
        scatter_ax(axesB[i], X2, y, LANGUAGES_WITH_EN, model_title)

    figB.tight_layout()
    outB = TSNE_OUT_DIR / "tsne_with_english"
    figB.savefig(f"{outB}.pdf", bbox_inches="tight")
    figB.savefig(f"{outB}.png", dpi=600, bbox_inches="tight")
    plt.close(figB)
    print(f"📈 Saved Figure B: {outB}")

    print("\n✅ Done. Figures are under:", TSNE_OUT_DIR.resolve())


if __name__ == "__main__":
    main()
