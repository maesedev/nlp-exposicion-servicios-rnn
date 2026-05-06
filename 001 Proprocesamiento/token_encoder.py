#!/usr/bin/env python3
"""
Preprocess and encode text using BPE vocabulary and merge rules.

Preprocessing: lowercase → strip accents → keep only [a-z0-9].
Use --raw to skip preprocessing and encode text as-is.

Usage:
    python token_encoder.py <vocabulary> <merges_file> <text>
    echo "Velociraptor" | python token_encoder.py <vocabulary> <merges_file>
    python token_encoder.py <vocabulary> <merges_file> "raw text" --raw
"""

import argparse
import csv
import re
import sys
import unicodedata

EOW = "<EOW>"
SOW = "<SOW>"
PAD = "<PAD>"
UNK = "<UNK>"


# ── Preprocessing ──────────────────────────────────────────────────────────────

def normalize(word: str) -> str:
    word = word.lower()
    word = unicodedata.normalize("NFD", word)
    word = "".join(c for c in word if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", word)


# ── Vocabulary & merges ────────────────────────────────────────────────────────

def load_merges(merges_file: str) -> list[tuple[str, str]]:
    merges = []
    with open(merges_file, newline="", encoding="utf-8") as f:
        for row in sorted(csv.DictReader(f), key=lambda r: int(r["paso"])):
            a, b = row["merge"].split(" + ", 1)
            merges.append((a, b))
    return merges


def load_vocab(vocabulary_file: str) -> dict[str, int]:
    """Return token → index mapping (index = line number, 0-based)."""
    vocab: dict[str, int] = {}
    with open(vocabulary_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tok = line.strip()
            if tok:
                vocab[tok] = idx
    return vocab


# ── Encoding ───────────────────────────────────────────────────────────────────

def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
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


def encode(
    text: str,
    merges: list[tuple[str, str]],
    vocab: dict[str, int] | None = None,
    raw: bool = False,
) -> list[str]:
    """
    Tokenize each word in text.
    Wraps each word with SOW/EOW when a vocabulary is provided.
    Unknown tokens are replaced with <UNK> when vocab is provided.
    """
    tokens: list[str] = []
    for word in text.split():
        word = word if raw else normalize(word)
        if not word:
            continue
        word_tokens = [SOW] + encode_word(word, merges) + [EOW] if vocab else encode_word(word, merges)
        if vocab:
            word_tokens = [t if t in vocab else UNK for t in word_tokens]
        tokens.extend(word_tokens)
    return tokens


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and encode text into BPE tokens")
    parser.add_argument("vocabulary", help="Vocabulary file (one token per line, index = line number)")
    parser.add_argument("merges_file", help="Merges CSV file")
    parser.add_argument("text", nargs="?", help="Text to encode (default: stdin)")
    parser.add_argument("--raw", action="store_true", help="Skip normalization preprocessing")
    parser.add_argument("--no-vocab", action="store_true", dest="no_vocab",
                        help="Skip vocabulary lookup (no SOW/EOW wrapping, no UNK replacement)")
    args = parser.parse_args()

    merges = load_merges(args.merges_file)
    vocab = None if args.no_vocab else load_vocab(args.vocabulary)
    text = args.text if args.text else sys.stdin.read().strip()
    tokens = encode(text, merges, vocab=vocab, raw=args.raw)
    print(" ".join(tokens))


if __name__ == "__main__":
    main()
