"""
Append missing special tokens to a trained vocab.tsv IN PLACE (writes a
backup first). Safe to run even if some special tokens are already present
-- those are skipped.

Usage:
    python add_special_tokens_to_vocab.py vocab_nepbpe/vocab_v4.tsv
"""

import shutil
import sys

SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]


def unescape_tsv(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "t":
                out.append("\t"); i += 2; continue
            if nxt == "n":
                out.append("\n"); i += 2; continue
            if nxt == "\\":
                out.append("\\"); i += 2; continue
        out.append(c)
        i += 1
    return "".join(out)


def escape_tsv(s: str) -> str:
    # Mirror of the Rust escape_tsv, for symmetry with the loader.
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python add_special_tokens_to_vocab.py <vocab.tsv>")
        sys.exit(1)

    vocab_path = sys.argv[1]
    backup_path = vocab_path + ".bak"

    # 1. Read existing entries, tracking the highest id and which surfaces
    #    already exist (so re-running this script is a no-op).
    max_id = -1
    existing_surfaces = set()
    lines = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw:
                continue
            lines.append(raw)
            id_str, _, surf_raw = raw.partition("\t")
            try:
                tid = int(id_str)
            except ValueError:
                continue
            max_id = max(max_id, tid)
            existing_surfaces.add(unescape_tsv(surf_raw))

    to_add = [t for t in SPECIAL_TOKENS if t not in existing_surfaces]
    if not to_add:
        print("All special tokens already present -- nothing to do.")
        return

    # 2. Backup before touching anything.
    shutil.copy2(vocab_path, backup_path)
    print(f"[backup] wrote {backup_path}")

    # 3. Append new entries with fresh contiguous-from-the-top ids.
    #    (load_from_pairs sorts by id and reassigns positionally, so these
    #    just need to be numerically highest -- exact contiguity with gaps
    #    elsewhere in the file doesn't matter.)
    next_id = max_id + 1
    with open(vocab_path, "a", encoding="utf-8") as f:
        for tok in to_add:
            f.write(f"{next_id}\t{escape_tsv(tok)}\n")
            print(f"[add] id={next_id}  surface={tok!r}")
            next_id += 1

    print(f"\nAdded {len(to_add)} special token(s). New vocab size: {next_id}")


if __name__ == "__main__":
    main()