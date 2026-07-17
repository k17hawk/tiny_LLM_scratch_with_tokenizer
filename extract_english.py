#!/usr/bin/env python3
"""
One-time: pull English-dominant lines out of the blended 23.9 GB corpus into a
smaller file, so Latin retrains stream ~12 GB instead of ~24 GB and you can tune
LAT_BUDGET cheaply.

A line is kept if it has at least as many ASCII letters as Devanagari letters.
That keeps pure-English and English-dominant mixed lines, drops pure-Nepali.
Word-level Devanagari filtering still happens inside the trainer, so keeping a
few stray Devanagari words on mixed lines is harmless.

Run:
    python extract_english.py dataset_merged/whole_train_corpus.txt english_only.txt
"""
import sys


def dev_count(s: str) -> int:
    return sum(1 for c in s if "\u0900" <= c <= "\u097f")


def latin_count(s: str) -> int:
    return sum(1 for c in s if c.isascii() and c.isalpha())


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python extract_english.py <in_corpus> <out_english>",
              file=sys.stderr)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    kept = total = 0
    with open(src, "r", encoding="utf-8", errors="replace") as fin, \
         open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            lat = latin_count(line)
            # cheap gate: need at least a few Latin letters AND >= Devanagari
            if lat >= 3 and lat >= dev_count(line):
                fout.write(line)
                kept += 1
            if total % 1_000_000 == 0:
                print(f"  {total:,} lines scanned, {kept:,} kept", flush=True)
    print(f"done: kept {kept:,} / {total:,} lines -> {dst}", flush=True)


if __name__ == "__main__":
    main()