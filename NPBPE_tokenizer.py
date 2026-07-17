import heapq
import re
import unicodedata
from array import array
from collections import Counter, defaultdict
from itertools import pairwise
from typing import Dict, List, Optional, Set, Tuple

# ── constants (unchanged) ────────────────────────────────────────────────────
SCRIPT_RANK = {"DEV": 2, "PUN": 1}
THETA = 100
LATIN_MERGE_BUDGET = 2000

FOLD_TABLE = {
    "\u0938\u0902\u0917": "\u0938\u0901\u0917",
    "\u0938\u0919\u094D\u0917": "\u0938\u0901\u0917",
}

ZWNJ = "\u200C"
DEV_LO, DEV_HI = 0x0900, 0x097F
VIRAMA = 0x094D
VIRAMA_CH = chr(VIRAMA)
INDEP_VOWEL = set(range(0x0904, 0x0915)) | set(range(0x0960, 0x0964))
COMBINING = (set(range(0x093E, 0x094D))
             | {0x0900, 0x0901, 0x0902, 0x0903, 0x093A, 0x093B, 0x093C})
DEV_PUNCT = {"\u0964", "\u0965"}


def is_dev(cp: int) -> bool:
    return DEV_LO <= cp <= DEV_HI


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for variant, canon in FOLD_TABLE.items():
        text = text.replace(variant, canon)
    return text.replace("\u200D", "")


def akshara_split(text: str) -> List[str]:
    out: List[str] = []
    buf = ""
    for ch in text:
        cp = ord(ch)
        if ch == ZWNJ or not is_dev(cp) or cp in INDEP_VOWEL:
            if buf:
                out.append(buf); buf = ""
            out.append(ch)
        elif cp in COMBINING:
            buf += ch; out.append(buf); buf = ""
        elif cp == VIRAMA:
            buf += ch
        else:
            if buf and not buf.endswith(VIRAMA_CH):
                out.append(buf); buf = ""
            buf += ch
    if buf:
        out.append(buf)
    return out


def script_of(token: str, vocab_info: Dict[str, Set[str]]) -> str:
    if token in vocab_info["zwnj"]:
        return "FMT"
    if token in vocab_info["byte_fallback"]:
        return "MAL"
    if token in vocab_info["punctuation"] or token in DEV_PUNCT:
        return "PUN"
    if any(is_dev(ord(c)) for c in token):
        return "DEV"
    return "LAT"


# ── ParadigmStore (unchanged) ────────────────────────────────────────────────
class ParadigmStore:
    def __init__(self) -> None:
        self.roots: Set[str] = set()
        self.state_to_roots: Dict[str, Set[str]] = defaultdict(set)
        self.rules: Dict[Tuple[str, str], Set[str]] = {}

    def add_root(self, root: str, transitions: Dict[str, Set[str]]) -> None:
        self.roots.add(root)
        transitions.setdefault(root, set())
        for state, nxt in transitions.items():
            self.state_to_roots[state].add(root)
            self.rules[(root, state)] = set(nxt)

    def allowed_next(self, root: str, state: str, candidate: str) -> bool:
        return candidate in self.rules.get((root, state), ())

    def roots_of(self, token: str) -> Set[str]:
        return set(self.state_to_roots.get(token, ()))


def build_sample_store() -> ParadigmStore:
    store = ParadigmStore()
    store.add_root("जा", {"जा": {"नु", "यो", "ने", "दा"},
                          "जानु": {"हो", "छ"},
                          "जायो": {"।"}})
    store.add_root("ग", {"ग": {"र्नु", "रे", "र्यो"},
                         "गर्नु": {"होस्", "पर्छ"}})
    return store


# ── tokenizer ────────────────────────────────────────────────────────────────
class NepBPETokenizer:
    def __init__(self, paradigm_store=None, frozen_strict=None,
                 frozen_ambiguous=None, punctuation_set=None):
        self.paradigm = paradigm_store or ParadigmStore()
        self.frozen_strict = frozen_strict or set()
        self.frozen_ambiguous = frozen_ambiguous or set()
        self.v_seed = self.frozen_strict | self.frozen_ambiguous
        self.punctuation_set = punctuation_set or set()

        self.id2token: List[str] = []
        self.token2id: Dict[str, int] = {}
        self.merge_rules: List[Tuple[str, str, int]] = []
        self.vocab_info = {
            "zwnj": {ZWNJ},
            "byte_fallback": {f"<0x{b:02X}>" for b in range(256)},
            "punctuation": self.punctuation_set,
        }
        self._root_ids = {r: i for i, r in enumerate(sorted(self.paradigm.roots))}

        for tok in (*self.vocab_info["byte_fallback"], ZWNJ,
                    *self.v_seed, *self.punctuation_set):
            self._add_token(tok)

    # -- vocab (unchanged) --
    def _add_token(self, token: str) -> int:
        if token not in self.token2id:
            self.token2id[token] = len(self.id2token)
            self.id2token.append(token)
        return self.token2id[token]

    # -- segmentation (unchanged) --
    def _initial_tokenize(self, text: str) -> List[str]:
        seeds = sorted(self.v_seed, key=len, reverse=True)
        out: List[str] = []
        buf = ""
        i = 0
        while i < len(text):
            hit = next((s for s in seeds if text.startswith(s, i)), None)
            if hit:
                if buf:
                    out.extend(akshara_split(buf)); buf = ""
                out.append(hit); i += len(hit)
            else:
                buf += text[i]; i += 1
        if buf:
            out.extend(akshara_split(buf))
        return out

    # -- constraints (unchanged) --
    def _root_set(self, token: str) -> Set[str]:
        return set() if token in self.v_seed else self.paradigm.roots_of(token)

    def _morph(self, a: str, b: str) -> bool:
        roots = self._root_set(a)
        return not roots or any(self.paradigm.allowed_next(r, a, b) for r in roots)

    def _gate(self, a: str, b: str, freq_ab: int) -> bool:
        sa, sb = script_of(a, self.vocab_info), script_of(b, self.vocab_info)
        if sa in {"MAL", "FMT"} or sb in {"MAL", "FMT"}:
            return False
        if not (sa == sb and sa in {"DEV", "PUN"}):
            return False
        if b in self.v_seed or a in self.frozen_strict:
            return False
        if a in self.frozen_ambiguous and freq_ab < THETA:
            return False
        return True

    def _legal(self, a: str, b: str, freq_ab: int) -> bool:
        return self._gate(a, b, freq_ab) and self._morph(a, b)

    def _key(self, a: str, b: str, freq: int, constrained: bool) -> Tuple:
        if not constrained:
            return (-freq,)
        return (-SCRIPT_RANK.get(script_of(a, self.vocab_info), 0), -freq)

    # -- shared merge machinery (unchanged) --
    @staticmethod
    def _merge_seq(tokens: List[str], a: str, b: str, ab: str) -> List[str]:
        out, i, n = [], 0, len(tokens)
        while i < n:
            if i + 1 < n and tokens[i] == a and tokens[i + 1] == b:
                out.append(ab); i += 2
            else:
                out.append(tokens[i]); i += 1
        return out

    @staticmethod
    def _contains_pair(toks: List[str], a: str, b: str) -> bool:
        return any(toks[i] == a and toks[i + 1] == b for i in range(len(toks) - 1))

    # ══════════════════════════════════════════════════════════════════════════
    # FIXED _bpe — compact where (array) + periodic heap rebuild
    # ══════════════════════════════════════════════════════════════════════════
    def _bpe(self, items, budget: int, constrained: bool):
        words = [[list(toks), w] for toks, w in items if toks]
        del items                                             # free immediately

        pair_freq: Dict[Tuple[str, str], int] = defaultdict(int)
        where: Dict[Tuple[str, str], array] = {}             # ★ array('I') not Set[int]

        for idx, (toks, w) in enumerate(words):
            for p in pairwise(toks):
                pair_freq[p] += w
                if p not in where:
                    where[p] = array('I')
                where[p].append(idx)

        def is_legal(a: str, b: str) -> bool:
            return (not constrained) or self._legal(a, b, pair_freq.get((a, b), 0))

        def build_heap() -> List[Tuple]:
            """Fresh heap from current pair_freq — O(|A|) but bounds memory."""
            h: List[Tuple] = []
            for (a, b), f in pair_freq.items():
                if f > 0 and is_legal(a, b):
                    heapq.heappush(h, (self._key(a, b, f, constrained), a, b, f))
            return h

        heap = build_heap()
        REBUILD_EVERY = 500                                   # ★ prevents unbounded heap growth
        made = 0

        while made < budget and heap:
            # ── periodic full rebuild keeps heap size ≡ |legal pairs| ──
            if made > 0 and made % REBUILD_EVERY == 0:
                heap = build_heap()

            _, a, b, snap = heapq.heappop(heap)
            if pair_freq.get((a, b), 0) != snap or not is_legal(a, b):
                continue

            ab = a + b
            self.merge_rules.append((a, b, self._add_token(ab)))

            indices = where.pop((a, b), None)
            if indices is not None:
                seen: Set[int] = set()                       # dedup (array has no set semantics)
                dirty: Set[Tuple[str, str]] = set()
                for idx in indices:
                    if idx in seen:
                        continue
                    seen.add(idx)
                    toks_w = words[idx]
                    toks, w = toks_w[0], toks_w[1]
                    if not self._contains_pair(toks, a, b):  # stale index
                        continue
                    for p in pairwise(toks):                 # withdraw old pairs
                        pair_freq[p] -= w
                        dirty.add(p)
                    merged = self._merge_seq(toks, a, b, ab)
                    toks_w[0] = merged
                    for p in pairwise(merged):               # register new pairs
                        pair_freq[p] += w
                        if p not in where:
                            where[p] = array('I')
                        where[p].append(idx)
                        dirty.add(p)

                for p in dirty:
                    if pair_freq.get(p, 0) <= 0:
                        pair_freq.pop(p, None)
                        where.pop(p, None)
                    else:
                        f = pair_freq[p]
                        if is_legal(*p):
                            heapq.heappush(
                                heap, (self._key(*p, f, constrained), *p, f))

            made += 1

        return [(toks, w) for toks, w in words]

    # ══════════════════════════════════════════════════════════════════════════
    # FIXED train — re.finditer instead of .split(), early del, optional norm
    # ══════════════════════════════════════════════════════════════════════════
    def train(self, corpus: str, vocab_size: int,
              latin_budget: int = LATIN_MERGE_BUDGET,
              normalize_input: bool = True):
        """Train BPE on a corpus string.

        Parameters
        ----------
        corpus : str
            Raw text.  If *normalize_input* is False you MUST pass
            already-normalized text (``normalize(corpus)``).
        normalize_input : bool
            True  → ``train`` calls ``normalize()`` internally (default, backward-compatible).
            False → skips the extra copy; use when you pre-normalised outside.
        """
        if normalize_input:
            corpus = normalize(corpus)

        # ★ FIX: re.finditer streams matches — never materialises a full word list
        freqs: Counter = Counter()
        for m in re.finditer(r'\S+', corpus):
            freqs[m.group()] += 1
        del corpus                                        # ★ free the (huge) string immediately

        items = [(self._initial_tokenize(w), c) for w, c in freqs.items()]
        del freqs                                         # ★ free counter
        for toks, _ in items:
            for t in toks:
                self._add_token(t)

        budget = max(0, vocab_size - len(self.id2token) - latin_budget)
        items = self._bpe(items, budget, constrained=True)

        if latin_budget > 0:
            lat = [([t for t in toks if script_of(t, self.vocab_info) == "LAT"], w)
                   for toks, w in items]
            del items                                     # ★ free before latin pass
            self._bpe([(s, w) for s, w in lat if len(s) > 1],
                      latin_budget, constrained=False)

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: train_from_file — never loads the full corpus into RAM
    # ══════════════════════════════════════════════════════════════════════════
    def train_from_file(self, filepath: str, vocab_size: int,
                        latin_budget: int = LATIN_MERGE_BUDGET,
                        encoding: str = "utf-8"):
        """Memory-safe training: reads the file line-by-line.

        Peak RAM ≈ size of the *unique-word* Counter + BPE working set,
        independent of total corpus size.  Use this for corpora > 1 GB.
        """
        freqs: Counter = Counter()
        with open(filepath, 'r', encoding=encoding) as fh:
            for line in fh:
                normed = normalize(line)
                for m in re.finditer(r'\S+', normed):
                    freqs[m.group()] += 1

        items = [(self._initial_tokenize(w), c) for w, c in freqs.items()]
        del freqs
        for toks, _ in items:
            for t in toks:
                self._add_token(t)

        budget = max(0, vocab_size - len(self.id2token) - latin_budget)
        items = self._bpe(items, budget, constrained=True)

        if latin_budget > 0:
            lat = [([t for t in toks if script_of(t, self.vocab_info) == "LAT"], w)
                   for toks, w in items]
            del items
            self._bpe([(s, w) for s, w in lat if len(s) > 1],
                      latin_budget, constrained=False)

    # -- encode / decode (unchanged) --
    def encode(self, text: str) -> List[int]:
        tokens: List[str] = []
        for part in re.split(r"(\s+)", normalize(text)):
            if not part:
                continue
            if part.isspace():
                for ch in part:
                    self._add_token(ch); tokens.append(ch)
            else:
                tokens.extend(self._initial_tokenize(part))
        for a, b, nid in self.merge_rules:
            tokens = self._merge_seq(tokens, a, b, self.id2token[nid])
        for t in tokens:
            self._add_token(t)
        return [self.token2id[t] for t in tokens]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id2token[i] for i in ids)

    def encode_bytes(self, data: bytes) -> List[int]:
        ids, run = [], []
        flush = lambda: (ids.extend(self.encode("".join(run))), run.clear())
        for ch in data.decode("utf-8", "surrogateescape"):
            o = ord(ch)
            if 0xDC80 <= o <= 0xDCFF:
                if run:
                    flush()
                ids.append(self.token2id[f"<0x{o - 0xDC00:02X}>"])
            else:
                run.append(ch)
        if run:
            flush()
        return ids

    def decode_bytes(self, ids: List[int]) -> bytes:
        out = bytearray()
        for tid in ids:
            tok = self.id2token[tid]
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                out.append(int(tok[3:5], 16))
            else:
                out.extend(tok.encode("utf-8"))
        return bytes(out)

    def vocab_size(self) -> int:
        return len(self.id2token)

    def script_embedding_id(self, token: str) -> int:
        return {"DEV": 0, "LAT": 1, "PUN": 2, "FMT": 3, "MAL": 4}.get(
            script_of(token, self.vocab_info), 5)

    def paradigm_id(self, token: str) -> int:
        roots = self._root_set(token)
        return self._root_ids.get(min(roots), -1) if roots else -1