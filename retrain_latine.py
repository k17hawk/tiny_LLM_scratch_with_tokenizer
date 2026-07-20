#!/usr/bin/env python3
"""
Improve English WITHOUT retraining Devanagari.

Loads your existing (frozen) Devanagari vocab, then runs a single Latin-only BPE
pass over an English corpus, adding LAT_BUDGET new ▂-prefixed Latin tokens on top.
Script separation guarantees no Devanagari token is touched or dropped.

Prereq: rebuild the extension with the two-marker tokenizer.rs:
    maturin develop --release

Optional but recommended (do ONCE, then reuse english_only.txt):
    python extract_english.py dataset_merged/whole_train_corpus.txt english_only.txt

Then:
    python retrain_latin.py
"""
import sys
import time
from HimalayanTOK_Nepali_64K import PyHimalayanTOK_Nepali_64K
# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
IN_VOCAB   = "vocab_nepbpe/nepbpe_vocab_bilingual_new.tsv"   # frozen DEV vocab
OUT_VOCAB  = "vocab_nepbpe/nepbpe_vocab_bilingual_v9.tsv"    # DEV + new LAT



ENGLISH_PATH = "fileweb_wikipedia_english.txt"

# English saturates faster than Devanagari but 4k was too thin. 8-12k is the
# sweet spot; go higher only if the fertility report still shows heavy Latin
# fragmentation. Because Devanagari is frozen, spending more here costs nothing
# on the Devanagari side.
LAT_BUDGET = 24_000

# English types are few (small alphabet), so 1 is usually fine on memory here —
# unlike the Devanagari build. Raise if RAM is tight.
MIN_WORD_FREQ   = 1
PROGRESS_LINES  = 500_000
PROGRESS_MERGES = 1_000

# Folding rules are baked into the vocab; they don't affect the LAT pass but the
# tokenizer object still needs them for normalize().
FOLDING_RULES = [
    ("सङ्ग", "संग"),
    ("सँग", "संग"),
]


def _show(tok, ids):
    out = []
    for i in ids:
        s = tok.get_token_surface(i)
        out.append("·" if s in ("\u2581", "\u2582") else s)
    return " ".join(out)


def main() -> None:
    tok = PyHimalayanTOK_Nepali_64K(folding_rules=FOLDING_RULES)

    # Load the frozen Devanagari vocab. load_from_pairs re-tags markers (▁ DEV,
    # ▂ LAT) and appends ▂ if the old vocab predates it — so the Latin pass has a
    # base marker to merge from even though the loaded vocab never saw ▂.
    n0 = tok.load_vocab_tsv(IN_VOCAB)
    print(f"loaded frozen vocab: {n0} tokens", flush=True)
    if not tok.vocab_contains("\u2582"):
        print("  FATAL: ▂ (U+2582) not present after load — stale .so? "
              "run maturin develop --release", file=sys.stderr)
        sys.exit(1)

    # Baseline English fertility BEFORE the retrain, for comparison.
    probe = "the study of mathematics in Nepal"
    before = tok.encode(probe)
    print(f"  before: {len(before)} tok | {_show(tok, before)}\n", flush=True)

    print(f"latin retrain on {ENGLISH_PATH}", flush=True)
    print(f"  +{LAT_BUDGET} LAT tokens  (frozen DEV base = {n0})", flush=True)
    print(f"  min_word_freq={MIN_WORD_FREQ}\n", flush=True)

    t0 = time.perf_counter()
    final = tok.train_latin_from_file(
        ENGLISH_PATH,
        LAT_BUDGET,
        MIN_WORD_FREQ,
        PROGRESS_LINES,
        PROGRESS_MERGES,
    )
    wall = time.perf_counter() - t0
    print(f"\n=== latin retrain complete ===", flush=True)
    print(f"vocab: {n0} -> {final}  (+{final - n0} LAT)  in {wall/60:.1f} min",
          flush=True)

    # Save DEV + new LAT.
    m = tok.vocab_size()
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        for i in range(m):
            s = tok.get_token_surface(i)
            s = s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
            f.write(f"{i}\t{s}\n")
    print(f"vocab saved: {OUT_VOCAB}", flush=True)

    # After: same probe, and confirm Devanagari + abbreviation are unchanged.
    after = tok.encode(probe)
    print("\n=== smoke ===", flush=True)
    print(f"  english after : {len(after)} tok | {_show(tok, after)}")
    for s in ("नेपालको इतिहास पुरानो छ",
              "सन् 2020 मा गा.वि.स.को निर्णय",
              "काठमाडौं Nepal मा Battalion छ"):
        ids = tok.encode(s)
        ok = tok.decode(ids) == tok.normalize(s)
        print(f"  {s}")
        print(f"    {len(ids)} tok | roundtrip={'OK' if ok else 'FAIL'} | {_show(tok, ids)}")


if __name__ == "__main__":
    main()