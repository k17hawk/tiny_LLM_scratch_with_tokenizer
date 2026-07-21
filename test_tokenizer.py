import sys
import statistics
import time

from HimalayanTOK_Nepali_64K import PyHimalayanTOK_Nepali_64K
VOCAB_TSV = "vocab_nepbpe/nepbpe_vocab_bilingual_v9.tsv"

# MUST be identical to what you trained with (train.py). I4f these differ,
# normalization drifts and surface lookups miss.
FOLDING_RULES = [
    ("सङ्ग", "संग"),
    ("सँग", "संग"),
]

# 'Ġ' (U+0120) is the byte-alphabet surface for space (0x20). Without
# Ġ-prefixing, each inter-word space is its own token.
SPACE_PIECE = "\u0120"


def show_piece(p: str) -> str:
    """Render a piece for display: space as ·, ZWNJ as <ZWNJ>."""
    if p == SPACE_PIECE:
        return "·"
    if p == "\u200c":
        return "<ZWNJ>"
    return p


SAMPLES = [
    "नेपालको राजधानी काठमाडौं हो।",
    "नेपाल दक्षिण एसियामा अवस्थित एक सुन्दर देश हो।",
    "सगरमाथा संसारको सबैभन्दा अग्लो हिमाल हो।",
    "आज मौसम निकै राम्रो छ।",
    "विद्यालयका विद्यार्थीहरू पुस्तकालयमा अध्ययन गरिरहेका छन्।",
    "नेपाल सरकारले नयाँ शिक्षा नीति लागू गर्यो।",
    "काठमाडौं महानगरपालिकाले सडक मर्मत सुरु गरेको छ।",
    "नेपाल राष्ट्र बैंकले नयाँ मौद्रिक नीति सार्वजनिक गर्यो।",
    "मलाई नेपाली भाषा धेरै मन पर्छ।",
    "हामी सबैले वातावरण संरक्षण गर्नुपर्छ।",
    "आज Docker install गरेँ।",
    "PyTorch को नयाँ version निकै राम्रो छ।",
    "TensorFlow भन्दा PyTorch प्रयोग गर्न सजिलो लाग्छ।",
    "GPU मा CUDA correctly install भएको छैन।",
    "आज meeting cancel भयो।",
    "Fine-tuning गर्दा memory धेरै चाहिन्छ।",
    "Model training अहिले चलिरहेको छ।",
    "Prompt engineering सिक्दैछु।",
    "LangChain प्रयोग गरेर RAG system बनाएको छु।",
    "Inference speed निकै राम्रो छ।",
    "The capital of Nepal is Kathmandu.",
    "Artificial Intelligence is transforming modern industries.",
    "Large Language Models require high-quality tokenizers.",
    "Python is one of the most popular programming languages.",
    "Machine learning models improve with better datasets.",
    "Docker containers simplify deployment.",
    "The experiment achieved state-of-the-art performance.",
    "The GPU utilization reached ninety-eight percent.",
    "Natural language processing is an exciting field.",
    "Open-source software accelerates innovation.",
    "आज ChatGPT ले मेरो tokenizer evaluate गर्यो।",
    "Model deployment सफल भयो तर GPU memory कम भयो।",
    "LangGraph प्रयोग गरेर workflow तयार गरियो।",
    "आज AI conference मा धेरै researchers आएका थिए।",
    "PySpark बाट data preprocessing गरियो।",
    "Ubuntu मा NVIDIA driver update गरेँ।",
    "Vector database मा embeddings store गरियो।",
    "Transformer architecture अहिले धेरै लोकप्रिय छ।",
    "Machine learning engineer ले model deploy गर्यो।",
    "आज meeting पछि code review गरियो।",
    "विद्यालयमा",
    "विद्यालयको",
    "विद्यालयदेखि",
    "विद्यालयसम्म",
    "विद्यालयबाट",
    "विद्यालयहरू",
    "विद्यालयहरूको",
    "विद्यालयहरूमध्ये",
    "घरमा",
    "घरबाट",
    "घरसम्म",
    "मानिसहरूले",
    "नेपालीहरूको",
    "कम्प्युटरहरू",
    "२०८३ साल",
    "१२३४५६७८९०",
    "Rs. ५०००",
    "$1250.75",
    "3.14159265",
    "५०%",
    "१०० किलोमिटर",
    "12,345.67",
    "https://openai.com",
    "https://huggingface.co",
    "https://github.com",
    "abc@gmail.com",
    "research@university.edu",
    "www.example.com",
    "😊",
    "😂",
    "🙏",
    "🚀",
    "🔥",
    "❤️",
    "Model.load_state_dict()",
    "torch.cuda.is_available()",
    "pip install transformers",
    "docker run -it ubuntu",
    "git clone https://github.com/example/repo.git",
    "k xa?",
    "mero ghar Kathmandu ho.",
    "tapai lai kasto cha?",
    "hajur sanchai hunuhuncha?",
    "ma aaja office gaye.",
    "yo model ramro cha.",
    "वि द्यालय",
    "नेपाल स र कार",
    "मोडेल train भइरहेको छ ।",
    "GPU   memory   full   भयो।"
]


def main(test_file=None) -> None:
    tok = PyHimalayanTOK_Nepali_64K(folding_rules=FOLDING_RULES)
    n = tok.load_vocab_tsv(VOCAB_TSV)
    print(f"loaded {n} tokens from {VOCAB_TSV}\n")

    raw_rates, content_rates = [], []
    tok_total = word_total = space_total = fails = 0

    print("=== sample tokenization ===")
    for idx, s in enumerate(SAMPLES, 1):
        print(f"  [{idx}/{len(SAMPLES)}] processing...", end="", flush=True)

        ids = tok.encode(s)
        pieces = [tok.get_token_surface(i) for i in ids]
        norm = tok.normalize(s)
        words = max(1, len(norm.split()))
        spaces = sum(1 for p in pieces if p == SPACE_PIECE)
        content = len(ids) - spaces
        ok = tok.decode(ids) == norm

        raw_rates.append(len(ids) / words)
        content_rates.append(content / words)
        tok_total += len(ids)
        word_total += words
        space_total += spaces
        if not ok:
            fails += 1

        shown = " ".join(show_piece(p) for p in pieces)
        print(f"\r  {s}")
        print(
            f"    {len(ids)} tok = {content} content + {spaces} space | "
            f"{len(ids)/words:.2f}/word ({content/words:.2f} ex-space) | "
            f"roundtrip={'OK' if ok else 'FAIL'}"
        )
        print(f"    {shown}")
        if not ok:
            print(f"    DECODED : {tok.decode(ids)!r}")
            print(f"    EXPECTED: {norm!r}")

    print("\n=== sample summary ===")
    print(
        f"  tokens/word   : mean={statistics.mean(raw_rates):.2f}  "
        f"median={statistics.median(raw_rates):.2f}"
    )
    print(
        f"  ex-space/word : mean={statistics.mean(content_rates):.2f}  "
        f"median={statistics.median(content_rates):.2f}   <- real subword fertility"
    )
    print(
        f"  micro/word    : {tok_total/max(1,word_total):.2f}  "
        f"(space tokens = {space_total}/{tok_total} = "
        f"{100*space_total/max(1,tok_total):.0f}%)"
    )
    print(f"  roundtrip     : {len(SAMPLES)-fails}/{len(SAMPLES)} OK")

    # Optional: fertility over a held-out file (fast, uses the Rust encode path).
    if test_file:
        print(f"\n=== fertility over {test_file} ===")
        space_id = tok.vocab_get_id(SPACE_PIECE)  # int (or None), computed once
        tt = ww = ss = lines = 0
        t0 = time.perf_counter()
        try:
            with open(test_file, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    if line_num % 1000 == 0:
                        elapsed = time.perf_counter() - t0
                        print(
                            f"  ... line {line_num}: {tt} tokens in {elapsed:.1f}s "
                            f"({tt/max(1,elapsed):.0f} tok/s)",
                            flush=True,
                        )

                    w = len(tok.normalize(line).split())
                    if w == 0:
                        continue

                    ids = tok.encode(line)
                    sp = ids.count(space_id) if space_id is not None else 0
                    tt += len(ids)
                    ss += sp
                    ww += w
                    lines += 1

                    if lines >= 20000:
                        print(f"  Reached {lines} lines limit", flush=True)
                        break

        except FileNotFoundError:
            print(f"  Error: File '{test_file}' not found. Skipping fertility analysis.")
            return
        except KeyboardInterrupt:
            print(f"\n  Interrupted after {lines} lines", flush=True)
            return

        dt = time.perf_counter() - t0
        print(
            f"  lines={lines} | tokens={tt} | tokens/word={tt/max(1,ww):.3f} | "
            f"ex-space/word={(tt-ss)/max(1,ww):.3f} | space-frac={ss/max(1,tt):.3f} | "
            f"{dt:.1f}s ({tt/max(1,dt):.0f} tok/s)"
        )


if __name__ == "__main__":
    # Handle both command-line and Jupyter environments.
    try:
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--f="):
            main(sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)