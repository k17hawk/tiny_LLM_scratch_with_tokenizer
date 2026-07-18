"""
Smoke test for HimalayanTokenizer — no full training run required.

Builds a minimal vocab directly via initialize_vocab (base aksharas + a couple
of seed morphemes + special tokens), saves it to a TSV, then loads it through
the HF wrapper to confirm the whole load -> encode -> decode path works.

Run from the same environment where `maturin develop --release` succeeded:
    python test_himalayan_tokenizer.py
"""

import os
import tempfile

from HimalayanTOK_Nepali_64K import PyHimalayanTOK_Nepali_64K
from HimalTokWrapper import HimalayanTokenizer


def build_minimal_vocab(vocab_path: str):
    tok = PyHimalayanTOK_Nepali_64K()

    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    # A handful of real Devanagari aksharas so DEV-script text has something
    # to greedy-match against. Extend this with your real akshara list later.
    aksharas = ["न", "मा", "या", "को", "ने", "पा", "ल", "स", "त", "र", "ग", "ी"]

    tok.initialize_vocab(
        aksharas=aksharas,
        seed_morphemes=special_tokens,
        punctuation=[".", ",", "?", "!"],
        v_strict=[],
        v_ambiguous=[],
    )

    print(f"[build] vocab size after initialize_vocab: {tok.vocab_size()}")
    tok.save_vocab_tsv(vocab_path)
    print(f"[build] wrote {vocab_path}")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.tsv")

        # 1. Build + save a minimal vocab (stand-in for a real training run).
        build_minimal_vocab(vocab_path)

        # 2. Load it through the HF wrapper.
        tokenizer = HimalayanTokenizer(vocab_file=vocab_path)
        print(f"[load] vocab_size property: {tokenizer.vocab_size}")
        print(f"[load] cls_token_id: {tokenizer.cls_token_id}")
        print(f"[load] sep_token_id: {tokenizer.sep_token_id}")
        print(f"[load] pad_token_id: {tokenizer.pad_token_id}")
        print(f"[load] unk_token_id: {tokenizer.unk_token_id}")
        print(f"[load] mask_token_id: {tokenizer.mask_token_id}")

        # 3. Encode / decode roundtrip.
        text = "नमाया को"
        ids = tokenizer.encode(text, add_special_tokens=True)
        print(f"[encode] '{text}' -> {ids}")

        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"[encode] tokens: {tokens}")

        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"[decode] -> '{decoded}'")

        # 4. Batch call with padding, the thing most HF training loops need.
        batch = tokenizer(
            ["नमाया", "को स"],
            padding=True,
            truncation=True,
            return_tensors=None,
        )
        print(f"[batch] input_ids: {batch['input_ids']}")
        print(f"[batch] attention_mask: {batch['attention_mask']}")

        # 5. save_pretrained / from_pretrained round trip.
        save_dir = os.path.join(tmpdir, "saved_tokenizer")
        tokenizer.save_pretrained(save_dir)
        reloaded = HimalayanTokenizer.from_pretrained(save_dir)
        print(f"[reload] vocab_size after from_pretrained: {reloaded.vocab_size}")
        assert reloaded.encode(text) == ids, "Reloaded tokenizer produced different ids!"

        print("\nAll checks passed.")


if __name__ == "__main__":
    main()