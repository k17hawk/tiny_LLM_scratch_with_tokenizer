#!/usr/bin/env python3
"""
Append new abbreviation seeds to a vocab TSV without retraining.

Reads an existing vocab TSV (id\tsurface), finds the max ID, and appends
new seeds that are not already present. The Rust load_vocab_tsv will
recompute seed_max_len on load, so atomic pre-scan will fire for these.

Usage:
    python append_seeds.py nepbpe_vocab_bilingual_v3.tsv nepbpe_vocab_bilingual_v4.tsv
"""
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: python append_seeds.py <input_tsv> <output_tsv>", file=sys.stderr)
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    # 1. Load existing vocab, find max ID, and track existing surfaces
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    surf_to_id = {}
    max_id = -1

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # TSV format: id\t surface  (surface might contain escaped chars, but we just need the surface)
        parts = line.split("\t", 1)
        if len(parts) == 2:
            try:
                idx = int(parts[0])
                surf = parts[1]
                if idx > max_id:
                    max_id = idx
                surf_to_id[surf] = idx
            except ValueError:
                continue

    # 2. Define the new seeds to add (from your fertility report's worst list)
    #    These are the most frequent fragments in your 596-line sample.
    new_seeds = [
        "कि.मी.",        # x70
        "कि.मि.",        # x64
        "वि.स.",         # x48
        "हुनुहुन्थ्यो",  # x27
        "हुन्थ्यो",      # x47 (appeared with danda, but seed without danda so it matches)
        "गराए।",         # x45 (with danda)
        "सम्भावना",      # x3 (appeared with danda, seed without)
        "आउँछ।",        # x27 (with danda)
    ]

    # 3. Filter out existing ones
    to_add = [s for s in new_seeds if s not in surf_to_id]

    if not to_add:
        print("No new seeds to add. All already exist.")
        print(f"Input: {in_path}, Output: {out_path} (unchanged)")
        sys.exit(0)

    # 4. Assign new contiguous IDs
    next_id = max_id + 1
    new_lines = []
    for seed in to_add:
        # No need to escape; these are pure Devanagari/periods/dandas.
        new_lines.append(f"{next_id}\t{seed}\n")
        next_id += 1

    # 5. Write the output
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)  # Keep original formatting
        f.writelines(new_lines)

    print(f"Added {len(to_add)} new seeds: {', '.join(to_add)}")
    print(f"New vocab size: {next_id - 1} tokens")
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    main()