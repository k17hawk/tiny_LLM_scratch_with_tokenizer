"""
Validate a loaded HimalayanTokenizer against real text: fertility (tokens per
word), UNK rate, and decode(encode(x)) == normalize(x) roundtrip fidelity.

Usage:
    python validate_tokenizer.py ./my-nepali-tokenizer path/to/sample.txt
    python validate_tokenizer.py ./my-nepali-tokenizer   # uses inline samples
"""

import sys

from HimalTokWrapper import HimalayanTokenizer

TOKENIZER_DIR = sys.argv[1] if len(sys.argv) > 1 else "./my-nepali-tokenizer"
SAMPLE_FILE = sys.argv[2] if len(sys.argv) > 2 else None

INLINE_SAMPLES = [
    "नेपालको संविधान २०७२ मा जारी भएको थियो",
    "यो एउटा राम्रो दिन हो",
    "काठमाडौं नेपालको राजधानी हो",
    "म भोलि स्कूल जान्छु",
    "This is some English text mixed in",
    "मिश्रित Nepali र English वाक्य",
]


def load_samples():
    if SAMPLE_FILE:
        with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return INLINE_SAMPLES


def main():
    tok = HimalayanTokenizer.from_pretrained(TOKENIZER_DIR)
    unk_id = tok.unk_token_id

    samples = load_samples()
    print(f"Loaded {len(samples)} sample lines\n")

    total_words = 0
    total_tokens = 0
    total_unks = 0
    roundtrip_fails = 0

    for line in samples:
        ids = tok.encode(line, add_special_tokens=False)
        n_words = len(line.split())
        n_tokens = len(ids)
        n_unks = sum(1 for i in ids if i == unk_id)

        decoded = tok.rust_tokenizer.decode(ids)
        normalized = tok.rust_tokenizer.normalize(line)
        roundtrip_ok = decoded.strip() == normalized.strip()

        total_words += n_words
        total_tokens += n_tokens
        total_unks += n_unks
        if not roundtrip_ok:
            roundtrip_fails += 1

        fertility = n_tokens / max(n_words, 1)
        print(f"[{fertility:4.2f} tok/word | {n_unks} unk | "
              f"roundtrip={'OK' if roundtrip_ok else 'FAIL'}] {line}")
        if not roundtrip_ok:
            print(f"    expected: {normalized!r}")
            print(f"    got:      {decoded!r}")

    print("\n--- Summary ---")
    print(f"Total words:  {total_words}")
    print(f"Total tokens: {total_tokens}")
    print(f"Fertility:    {total_tokens / max(total_words, 1):.3f} tokens/word")
    print(f"UNK rate:     {total_unks}/{total_tokens} "
          f"({100 * total_unks / max(total_tokens, 1):.2f}%)")
    print(f"Roundtrip failures: {roundtrip_fails}/{len(samples)}")


if __name__ == "__main__":
    main()