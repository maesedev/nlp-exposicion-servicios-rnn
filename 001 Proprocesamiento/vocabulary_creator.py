#!/usr/bin/env python3
"""
Build BPE vocabulary from dinosaur names.

Each sequence is wrapped with <SOW> and <EOW> special tokens.
Sequences shorter than --min-len (T) are right-padded with <PAD>.

Outputs (inside --out-dir):
    vocabulary.txt  — one token per line; index = line number
    sequences.csv   — word, token sequence, length
    merges.csv      — BPE merge rules (paso, merge, token_resultante)

Usage:
    python vocabulary_creator.py "../000 data/Dinosours.csv"
    python vocabulary_creator.py "../000 data/Dinosours.csv" --merges 400 --min-len 6
"""

import argparse
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

SOW = "<SOW>"
EOW = "<EOW>"
PAD = "<PAD>"
UNK = "<UNK>"

SPECIAL_TOKENS = [PAD, SOW, EOW, UNK]


# ── Normalización ──────────────────────────────────────────────────────────────

def normalize(word: str) -> str:
    word = word.lower()
    word = unicodedata.normalize("NFD", word)
    word = "".join(c for c in word if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", word)


# ── BPE training ───────────────────────────────────────────────────────────────

def _get_pairs(vocab: dict[str, int]) -> Counter:
    pairs: Counter = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def _merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    bigram = " ".join(pair)
    replacement = "".join(pair)
    return {w.replace(bigram, replacement): f for w, f in vocab.items()}


def train_bpe(words: list[str], n_merges: int) -> list[tuple[str, str]]:
    """Train BPE on plain normalized words (no embedded markers)."""
    vocab: dict[str, int] = Counter(" ".join(list(w)) for w in words if w)
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        pairs = _get_pairs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = _merge_pair(best, vocab)
        merges.append(best)
    return merges


# ── Encoding ───────────────────────────────────────────────────────────────────

def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Apply BPE merges to a single normalized word."""
    tokens = list(word)
    for left, right in merges:
        i, merged = 0, []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                merged.append(left + right)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        tokens = merged
    return tokens


def build_sequence(word: str, merges: list[tuple[str, str]], min_len: int) -> list[str]:
    """Return [<SOW>, bpe_tokens..., <EOW>] right-padded to min_len with <PAD>."""
    seq = [SOW] + encode_word(word, merges) + [EOW]
    while len(seq) < min_len:
        seq.append(PAD)
    return seq


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build BPE vocabulary with SOW/EOW/PAD")
    parser.add_argument("data_csv", help="CSV file with dinosaur names (column 'nombre')")
    parser.add_argument(
        "--merges", type=int, default=300, metavar="N",
        help="Number of BPE merge operations (default: 300)",
    )
    parser.add_argument(
        "--min-len", type=int, default=4, dest="min_len", metavar="T",
        help="Minimum sequence length T; shorter sequences get <PAD> appended (default: 4)",
    )
    parser.add_argument(
        "--out-dir", default="tokens", dest="out_dir", metavar="DIR",
        help="Output directory (default: tokens/)",
    )
    args = parser.parse_args()

    # ── Load & normalize ──────────────────────────────────────────────────────
    words: list[str] = []
    with open(args.data_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = normalize(row["nombre"])
            if w:
                words.append(w)

    print(f"Words loaded    : {len(words)}")

    # ── Train BPE ─────────────────────────────────────────────────────────────
    merges = train_bpe(words, args.merges)
    print(f"BPE merges done : {len(merges)}")

    # ── Build sequences & collect all BPE tokens ──────────────────────────────
    bpe_tokens: set[str] = set()
    rows: list[dict] = []

    for word in words:
        seq = build_sequence(word, merges, args.min_len)
        rows.append({"word": word, "sequence": " ".join(seq), "length": len(seq)})
        bpe_tokens.update(t for t in seq if t not in SPECIAL_TOKENS)

    # ── Vocabulary: special tokens first, then sorted BPE tokens ─────────────
    vocab_list = SPECIAL_TOKENS + sorted(bpe_tokens)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # vocabulary.txt — index of each token = its line number (0-based)
    with open(out_dir / "vocabulary.txt", "w", encoding="utf-8") as f:
        for tok in vocab_list:
            f.write(tok + "\n")

    # sequences.csv
    with open(out_dir / "sequences.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "sequence", "length"])
        writer.writeheader()
        writer.writerows(rows)

    # merges.csv
    with open(out_dir / "merges.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paso", "merge", "token_resultante"])
        for i, (a, b) in enumerate(merges, 1):
            writer.writerow([i, f"{a} + {b}", a + b])

    padded = sum(1 for r in rows if r["length"] > len(r["sequence"].split()) - r["sequence"].count(PAD))
    padded = sum(1 for r in rows if PAD in r["sequence"])

    print(f"Vocabulary size : {len(vocab_list)}  ({len(SPECIAL_TOKENS)} special + {len(bpe_tokens)} BPE tokens)")
    print(f"Sequences saved : {len(rows)}  ({padded} padded to T={args.min_len})")
    print(f"Saved to        : {out_dir}/")
    print(f"  vocabulary.txt  ({len(vocab_list)} tokens, index = line number)")
    print(f"  sequences.csv   ({len(rows)} sequences)")
    print(f"  merges.csv      ({len(merges)} rules)")


if __name__ == "__main__":
    main()
