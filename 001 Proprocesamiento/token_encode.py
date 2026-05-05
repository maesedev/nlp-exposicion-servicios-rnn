#!/usr/bin/env python3
"""
Encode text using BPE vocabulary and merge rules.

Usage:
    python token_encode.py <tokens_file> <merges_file> <text>
    echo "velociraptor" | python token_encode.py <tokens_file> <merges_file>
"""

import argparse
import csv
import sys

EOW = "</w>"


def load_merges(merges_file: str) -> list[tuple[str, str]]:
    merges = []
    with open(merges_file, newline="", encoding="utf-8") as f:
        for row in sorted(csv.DictReader(f), key=lambda r: int(r["paso"])):
            a, b = row["merge"].split(" + ", 1)
            merges.append((a, b))
    return merges


def load_vocab(tokens_file: str) -> set[str]:
    with open(tokens_file, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def tokenize_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    tokens = list(word) + [EOW]
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


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    tokens = []
    for word in text.lower().split():
        tokens.extend(tokenize_word(word, merges))
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode text into BPE tokens")
    parser.add_argument("tokens_file", help="Vocabulary file (one token per line)")
    parser.add_argument("merges_file", help="Merges CSV file")
    parser.add_argument("text", nargs="?", help="Text to encode (default: stdin)")
    args = parser.parse_args()

    merges = load_merges(args.merges_file)
    text = args.text if args.text else sys.stdin.read().strip()
    tokens = encode(text, merges)
    print(" ".join(tokens))


if __name__ == "__main__":
    main()
