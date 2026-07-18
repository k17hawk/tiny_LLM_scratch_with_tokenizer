"""
Turn an existing trained vocab.tsv into a proper HF tokenizer folder, then
prove from_pretrained() can load it with NO vocab_file argument.

Usage:
    python setup_and_test_pretrained.py \
        /home/lang-chain/Documents/tiny_LLM_scratch_with_tokenizer/vocab_nepbpe/vocab_v4.tsv \
        ./my-nepali-tokenizer
"""

import sys

from HimalTokWrapper import HimalayanTokenizer

SOURCE_VOCAB = sys.argv[1] if len(sys.argv) > 1 else \
    "vocab_nepbpe/vocab_v4.tsv"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "./my-nepali-tokenizer"


def main():
    # STEP 1 — load the trained vocab manually, ONE TIME, by path.
    # This is the only place you ever need to pass vocab_file explicitly.
    print(f"[step 1] loading {SOURCE_VOCAB} ...")
    tokenizer = HimalayanTokenizer(vocab_file=SOURCE_VOCAB)
    print(f"[step 1] vocab_size = {tokenizer.vocab_size}")

    # STEP 2 — save_pretrained writes the whole folder HF expects:
    #   OUT_DIR/vocab.tsv                 (via save_vocabulary, your override)
    #   OUT_DIR/tokenizer_config.json     (class name + init kwargs, auto)
    #   OUT_DIR/special_tokens_map.json   (special token strings, auto)
    print(f"[step 2] writing tokenizer folder to {OUT_DIR} ...")
    tokenizer.save_pretrained(OUT_DIR)

    import os
    print(f"[step 2] folder contents: {os.listdir(OUT_DIR)}")

    # STEP 3 — the actual test: load again with NO vocab_file argument.
    # from_pretrained reads vocab_files_names, finds "vocab.tsv" in OUT_DIR,
    # and passes it into __init__ for you.
    print(f"\n[step 3] reloading via from_pretrained('{OUT_DIR}') — no vocab_file passed")
    reloaded = HimalayanTokenizer.from_pretrained(OUT_DIR)
    print(f"[step 3] vocab_size = {reloaded.vocab_size}")

    # Sanity: both tokenizers must agree.
    text = "नेपाल एक सुन्दर देश हो"
    ids_original = tokenizer.encode(text)
    ids_reloaded = reloaded.encode(text)
    print(f"\n[check] original ids: {ids_original}")
    print(f"[check] reloaded ids: {ids_reloaded}")
    assert ids_original == ids_reloaded, "Mismatch after from_pretrained reload!"
    print("[check] PASSED — automatic vocab loading works")

    # OPTIONAL — register so generic AutoTokenizer.from_pretrained also works.
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.register("himalayan", HimalayanTokenizer)
        auto_loaded = AutoTokenizer.from_pretrained(OUT_DIR)
        print(f"\n[optional] AutoTokenizer.from_pretrained also works: "
              f"vocab_size={auto_loaded.vocab_size}")
    except Exception as e:
        print(f"\n[optional] AutoTokenizer path skipped/failed: {e}")


if __name__ == "__main__":
    main()


