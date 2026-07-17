#!/usr/bin/env python3
"""
Train the bilingual (Nepali + English) NepBPE tokenizer.

Pipeline:
  1. Build the word-frequency dictionary once, streaming the merged corpus.
  2. Phase 3 (constrained)  : Devanagari + punctuation -> dev_budget slots.
  3. Phase 4 (unconstrained): Latin + digits only      -> lat_budget slots.
  4. Save the vocab to TSV.

Prereq: rebuild the extension after any Rust change:
    maturin develop --release

Run:
    python train_bilingual.py
"""

import time
import sys
from tiny_llm_scratch_with_tokenizer import PyNepBPETokenizer

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_PATH = "dataset_merged/whole_train_corpus.txt"   

# Budget split is an explicit CHOICE. On a Nepali-dominant corpus, total token
# count is overwhelmingly Devanagari (in the last held-out set: ~758k DEV words
# vs ~12k Latin), so the DEV slice wins the joint-fertility trade. Moving slots
# to LAT helps ~12k words a lot but hurts ~758k words a little — net worse for
# total tokens. Keep DEV high unless you specifically need standalone English
# fluency for a downstream task, then raise LAT and re-measure.
DEV_BUDGET = 44_000
LAT_BUDGET =  4_000

THETA           = 100     # V_ambiguous frequency gate (Phase 3 only)

MIN_WORD_FREQ   = 2
PROGRESS_LINES  = 500_000
PROGRESS_MERGES = 1_000
OUT_VOCAB       = "nepbpe_vocab_bilingual_new.tsv"

# FROZEN AT TRAINING TIME. Whatever you train with, you must load with — the
# folding policy is baked into the vocabulary and cannot be changed after.
FOLDING_RULES = [
    ("सङ्ग", "संग"),   # explicit nasal conjunct -> anusvara
    ("सँग", "संग"),    # chandrabindu            -> anusvara
]

# Base vocabulary. Latin a-z A-Z 0-9 are seeded automatically inside
# initialize_vocab() as LAT-scripted tokens — do NOT add them here. The space
# marker (▁) and the General-Punctuation set (– — … curly quotes) are ALSO seeded
# automatically on the Rust side, so you don't need to list those either.
DEVANAGARI  = [chr(c) for c in range(0x0900, 0x0980)]          # U+0900..U+097F

# Only ASCII punctuation + the two Devanagari dandas here. Extended Unicode
# punctuation (en-dash, em-dash, ellipsis, curly quotes) is owned by the Rust
# EXTENDED_PUNCT set — keep it there so the two lists can't drift.
PUNCTUATION = list(".,!?;:()[]{}\"'`-/\\@#%&*+=<>|~") + ["।", "॥"]


SEED_MORPHEMES = [
    # Original prompts
    "गा.वि.स.", "वि.सं.", "ज.ब.रा.", "बि.स.", "इ.सं.", "ई.सं.",
    "प्र.", "डा.", "श्री", "रु.",

    # Administration & government
    "न.पा.", "उ.मा.वि.", "मा.वि.", "प्रा.वि.", "नि.मा.वि.",
    "जि.वि.स.", "गा.पा.", "प्र.जि.अ.", "जि.प्र.का.", "जि.प्र.शा.",
    "स.प्र.", "ने.प्र.", "ने.से.", "स.प्र.नि.", "व.प्र.अ.", "प्र.अ.",
    "ना.प्र.", "ह.प्र.", "जि.अ.", "उ.अ.", "स.अ.", "पुन.अ.", "मु.स.",
    "स.स.", "उ.स.", "स.अ.", "ना.सु.", "ख.सु.", "सु.", "का.मु.",
    "अ.प्र.",

    # Banks (Nepali abbreviations)
    "ने.रा.बैं.", "रा.ब.बैं.", "ने.बैं.लि.", "कृ.वि.बैं.",
    "ना.बि.बैं.", "ए.रे.बैं.", "हि.बि.बैं.", "सि.बि.बैं.",
    "प्र.ब.बैं.", "म.बि.बैं.", "ग्लो.बैं.", "ल.बि.बैं.",
    "ने.इ.बि.बैं.", "एन.आइ.सि.", "सि.बैं.", "कु.बि.बैं.",
    "सा.बि.बैं.", "ने.वि.प्रा.",

    # Media & telecom
    "ने.दू.सं.", "ने.टे.", "ने.ते.नि.", "ने.रे.", "ने.टि.भी.",

    # Political parties
    "ने.का.", "ने.क.पा.", "ने.क.पा.ए.मा.ले.", "ने.क.पा.मा.के.",
    "ने.क.पा.ए.स.", "रा.प्र.पा.", "रा.ज.पा.", "ज.म.पा.", "सं.पा.",

    # Titles and designations
    "प्रा.", "प्रा.लि.", "प.लि.", "लि.", "प्र.म.", "उ.प्र.म.",
    "स.प्र.म.नि.", "ने.प्र.म.नि.", "प्र.से.", "र.से.", "उ.र.से.",
    "स.से.", "म.न.पा.", "उ.म.न.पा.", "जि.स.", "न.स.", "गा.स.",
    "व.स.", "नि.प्रा.", "शा.अ.",

    # Education / degrees
    "एस.एल.सी.", "एस.इ.इ.", "पि.एच.डी.", "बि.ए.", "बि.कम.",
    "एम.ए.", "एम.कम.", "एम.बि.ए.", "बि.बि.ए.", "आइ.ए.",
    "आइ.कम.", "सि.ए.", "बि.एस्सी.", "एम.एस्सी.", "एम.डी.",
    "एम.बि.बि.एस.", "बि.डि.एस.",

    # Miscellaneous official / written
    "इ.का.", "म.स.", "ख.", "अ.स.", "का.खा.", "टि.का.", "द्र.",
    "उ.", "प्र.स.", "स.चि.", "सहा.स.", "प्र.आ.", "व.प्र.आ.",
    "स.प्र.अ.", "जि.पं.", "ने.म.सं.", "रा.स्वा.", "अ.ना.",
    "प्र.ले.", "म.ले.", "सि.डि.ओ.",

    # International organisations
    "ए.डि.बि.", "एन.जि.ओ.", "आइ.एन.जि.ओ.", "यु.एन.",
    "डब्लु.एच.ओ.", "यु.एन.डि.पि.",
]

# Multi-digit seeds so common years/numbers don't fragment. (Single digits 0-9
# already exist as base tokens.) These are pure-LAT surfaces with no period, so
# they go through the normal Latin run — NOT the atomic path — and do not affect
# the abbreviation pre-scan window.
#
# Trade-off worth knowing: every seed is a base token that counts against the
# total vocab budget, so ~200 year strings spend ~200 merge slots. With ASCII
# digits now routed through the Latin pass (fix #3), the LAT pass will learn the
# frequent multi-digit tokens on its own. If you want those slots back for
# Devanagari, trim _YEARS to the decades that actually appear in your corpus.
_YEARS = [str(y) for y in range(1900, 2100)]      # 1900-2099
_COMMON_NUMBERS = [
    "10", "20", "30", "40", "50", "60", "70", "80", "90",
    "100", "200", "500", "1000", "2000",
]
DIGIT_SEEDS = _YEARS + _COMMON_NUMBERS
SEED_MORPHEMES.extend(DIGIT_SEEDS)

V_STRICT    = []   # unconditionally frozen (terminal — never left-extend)
V_AMBIGUOUS = []   # frequency-gated by THETA


def _show_pieces(tok, ids):
    """Render token surfaces, showing the space marker ▁ as a visible dot."""
    out = []
    for i in ids:
        s = tok.get_token_surface(i)
        out.append("·" if s == "\u2581" else s)
    return " ".join(out)


def main() -> None:
    tok = PyNepBPETokenizer(folding_rules=FOLDING_RULES)

    base = tok.initialize_vocab(
        DEVANAGARI,
        SEED_MORPHEMES,
        PUNCTUATION,
        V_STRICT,
        V_AMBIGUOUS,
    )
    print(f"base vocab: {base} tokens "
          f"(128 Devanagari + {len(SEED_MORPHEMES)} seeds + "
          f"{len(PUNCTUATION)} punct + 62 Latin/digit + 256 bytes + ▁ + "
          f"extended-punct + ZWNJ)", flush=True)

    # Sanity: Latin base tokens must exist or Phase 4 merges nothing.
    for probe in ("a", "Z", "7"):
        if not tok.vocab_contains(probe):
            print(f"  FATAL: Latin base token '{probe}' missing. Phase 4 will "
                  f"produce ZERO merges. Is the rebuilt .so loaded?",
                  file=sys.stderr)
            sys.exit(1)
    # Sanity: the space marker must exist (fix #1) or spaces byte-fall-back to 3
    # tokens each and fertility explodes.
    if not tok.vocab_contains("\u2581"):
        print("  FATAL: space marker ▁ (U+2581) missing from vocab. The rebuilt "
              ".so is stale — run `maturin develop --release`.", file=sys.stderr)
        sys.exit(1)
    # Sanity: at least one abbreviation seed present (atomic pre-scan target).
    if not tok.vocab_contains("गा.वि.स."):
        print("  WARN: abbreviation seed 'गा.वि.स.' missing — atomic pre-scan "
              "has nothing to fire on.", file=sys.stderr)
    print("  base alphabet + ▁ marker + abbreviation seeds present\n", flush=True)

    print(f"training on {DATA_PATH}", flush=True)
    print(f"  budget: {DEV_BUDGET} DEV + {LAT_BUDGET} LAT = "
          f"{DEV_BUDGET + LAT_BUDGET} total", flush=True)
    print(f"  theta={THETA}  min_word_freq={MIN_WORD_FREQ}\n", flush=True)

    t0 = time.perf_counter()
    final = tok.train_bilingual_from_file(
        DATA_PATH,
        DEV_BUDGET,
        LAT_BUDGET,
        THETA,
        MIN_WORD_FREQ,
        PROGRESS_LINES,
        PROGRESS_MERGES,
    )
    wall = time.perf_counter() - t0

    print(f"\n=== training complete ===", flush=True)
    print(f"final vocab : {final}", flush=True)
    print(f"wall clock  : {wall:.1f}s  ({wall/60:.1f} min)", flush=True)

    # Save. Escape tab/newline/backslash so the TSV stays parseable; the Rust
    # loader (load_vocab_tsv) reverses exactly this escaping.
    n = tok.vocab_size()
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        for i in range(n):
            s = tok.get_token_surface(i)
            s = s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
            f.write(f"{i}\t{s}\n")
    print(f"vocab saved : {OUT_VOCAB}", flush=True)

    # Smoke test: Nepali, English, mixed, digits, and — critically — a MID-
    # SENTENCE abbreviation, which is the case the atomic pre-scan exists for.
    print("\n=== smoke test ===", flush=True)
    samples = [
        "नेपालको इतिहास धेरै पुरानो छ",
        "the study of mathematics in Nepal",
        "काठमाडौं Nepal मा UNIFIL छ",
        "सन् 2020 मा गा.वि.स.को निर्णय",
    ]
    for s in samples:
        ids = tok.encode(s)
        ok = tok.decode(ids) == tok.normalize(s)
        words = max(1, len(tok.normalize(s).split()))
        print(f"  {s}")
        print(f"    {len(ids)} tok ({len(ids)/words:.2f}/word) | "
              f"roundtrip={'OK' if ok else 'FAIL'}")
        print(f"    {_show_pieces(tok, ids)}")

    # Explicit assertion that the abbreviation fires as ONE token mid-sentence.
    # (Run-splitting + marker interaction is the fragile part; verify it landed.)
    print("\n=== abbreviation pre-scan check ===", flush=True)
    for phrase in ("गा.वि.स.", "सन् 2020 मा गा.वि.स.को निर्णय"):
        ids = tok.encode(phrase)
        surfaces = [tok.get_token_surface(i) for i in ids]
        fired = "गा.वि.स." in surfaces
        print(f"  {phrase!r}")
        print(f"    {len(ids)} tok | abbreviation fired={'YES' if fired else 'NO'}")
        print(f"    {_show_pieces(tok, ids)}")


if __name__ == "__main__":
    main()