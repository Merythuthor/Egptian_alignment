
from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.normalizers import NFKC


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CORPUS_DIR = PROJECT_ROOT / "data" / "processed_jsonl"
OUTPUT_DIR = SCRIPT_DIR


LANGS = ["hieroglyphic", "demotic", "bohairic", "sahidic"]


VOCAB_SIZE = 32000
MIN_FREQ = 2


SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "<hiero>", "<dem>", "<boh>", "<sah>", "<eng>", "[gap]",
]




def iter_corpus() -> Iterable[str]:

    print(f"INFO: Reading corpus from {CORPUS_DIR}")
    for lang in LANGS:
        filepath = CORPUS_DIR / f"{lang}_rev.jsonl"
        if not filepath.exists():
            print(f"WARNING: File not found, skipping: {filepath}")
            continue

        print(f"INFO: Processing language file: {filepath.name}")
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)

                    if "text" in data and data["text"]:
                        yield data["text"]

                    if "translation" in data and data["translation"]:
                        yield data["translation"]
                except json.JSONDecodeError:
                    print(f"WARNING: Could not parse JSON line in {filepath.name}: {line.strip()}")



def main():

    print("\n" + "=" * 50)
    print("🚀 Starting Byte-Level BPE Tokenizer Training (v3.0) 🚀")
    print("=" * 50)
    print(f"Source Corpus Path : {CORPUS_DIR}")
    print(f"Languages          : {', '.join(LANGS)}")
    print(f"Vocab Size         : {VOCAB_SIZE}")
    print(f"Min Frequency      : {MIN_FREQ}")
    print(f"Special Tokens     : {SPECIAL_TOKENS}")
    print(f"Output Directory   : {OUTPUT_DIR}")
    print("=" * 50 + "\n")


    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))


    tokenizer.normalizer = NFKC()


    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()


    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )


    corpus_iterator = iter_corpus()
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "tokenizer.json"
    tokenizer.save(str(save_path))

    print("\n" + "=" * 50)
    print(f"✅ Training complete!")
    print(f"Tokenizer saved to: {save_path}")
    print(f"Final Vocab Size: {tokenizer.get_vocab_size()}")
    print("=" * 50 + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a shared Byte-Level BPE tokenizer for Ancient Egyptian and English."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=VOCAB_SIZE, help="The size of the vocabulary."
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=MIN_FREQ,
        help="The minimum frequency for a token to be included.",
    )
    args = parser.parse_args()


    VOCAB_SIZE = args.vocab_size
    MIN_FREQ = args.min_freq

    main()