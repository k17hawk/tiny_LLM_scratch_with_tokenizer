import heapq
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

SCRIPT_RANK = {
    "DEV": 2,
    "PUN": 1,
    "LAT": 0,
}

SCRIPT_FAMILIES = {"DEV", "PUN", "LAT", "FMT", "MAL"}

THETA = 100

LATIN_MERGE_BUDGET = 2000

FOLD_TABLE = {
    "\u0938\u0901\u0917": "\u0938\u0901\u0917",
    "\u0938\u0902\u0917": "\u0938\u0901\u0917",
    "\u0938\u0919\u094D\u0917": "\u0938\u0901\u0917",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for variant, canon in FOLD_TABLE.items():
        text = text.replace(variant, canon)
    text = text.replace("\u200D", "")
    return text


DEVANAGARI_RANGE = (0x0900, 0x097F)
DEV_VOWEL_SIGNS = set(range(0x093E, 0x094D)) | {0x0902, 0x0901, 0x0903, 0x093A, 0x093B}
DEV_VIRAMA = 0x094D
ZWNJ = "\u200C"
ZWJ = "\u200D"


def is_devanagari(cp: int) -> bool:
    return DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]


class AksharaTokenizer:
    @staticmethod
    def tokenize_devanagari(text: str) -> List[str]:
        tokens = []
        buffer = []
        i = 0
        while i < len(text):
            ch = text[i]
            cp = ord(ch)

            if ch == ZWNJ:
                if buffer:
                    tokens.append("".join(buffer))
                    buffer = []
                tokens.append(ZWNJ)
                i += 1
                continue

            if not is_devanagari(cp):
                if buffer:
                    tokens.append("".join(buffer))
                    buffer = []
                tokens.append(ch)
                i += 1
                continue

            # Independent vowel – always a standalone akshara
            if (0x0904 <= cp <= 0x0914) or (0x0960 <= cp <= 0x0963):
                if buffer:
                    tokens.append("".join(buffer))
                    buffer = []
                tokens.append(ch)
                i += 1
                continue

            # Vowel signs, anusvara, visarga, candrabindu – attach to current consonant(s)
            if cp in DEV_VOWEL_SIGNS:
                buffer.append(ch)
                tokens.append("".join(buffer))
                buffer = []
                i += 1
                continue

            # Virama – does not end an akshara; continue in buffer
            if cp == DEV_VIRAMA:
                buffer.append(ch)
                i += 1
                # If virama is at end of text, close buffer (halanta form)
                if i >= len(text):
                    tokens.append("".join(buffer))
                    buffer = []
                continue

            # Consonant
            buffer.append(ch)
            i += 1

            if i >= len(text):
                tokens.append("".join(buffer))
                buffer = []
                continue

            next_cp = ord(text[i])
            if next_cp == DEV_VIRAMA or next_cp in DEV_VOWEL_SIGNS:
                continue       # stay in buffer
            # Next char starts a new syllable
            tokens.append("".join(buffer))
            buffer = []

        if buffer:
            tokens.append("".join(buffer))
        return tokens


def script_of(token: str, vocab_info: Dict[str, Set[str]]) -> str:
    if token in vocab_info["zwnj"]:
        return "FMT"
    if token in vocab_info["byte_fallback"]:
        return "MAL"
    if token in vocab_info["punctuation"]:
        return "PUN"
    if any(is_devanagari(ord(c)) for c in token):
        return "DEV"
    return "LAT"


class ParadigmStore:
    def __init__(self):
        self.transitions = {}

    def allowed_next(self, root: str, state: str, nxt: str) -> bool:
        if root not in self.transitions:
            return False
        st = self.transitions[root]
        if state not in st:
            return False
        return nxt in st[state]

    def prefixes_of(self, root: str) -> Set[str]:
        if root not in self.transitions:
            return set()
        return set()


class NepBPETokenizer:
    def __init__(
        self,
        paradigm_store: Optional[ParadigmStore] = None,
        frozen_strict: Optional[Set[str]] = None,
        frozen_ambiguous: Optional[Set[str]] = None,
        punctuation_set: Optional[Set[str]] = None,
    ):
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
            "byte_fallback": set(),
            "punctuation": self.punctuation_set,
        }

        for b in range(256):
            token = f"<0x{b:02X}>"
            self.vocab_info["byte_fallback"].add(token)
            self._add_token(token)

        self._add_token(ZWNJ)

        for tok in self.v_seed:
            self._add_token(tok)
        for tok in self.punctuation_set:
            self._add_token(tok)

    def _add_token(self, token: str) -> int:
        if token not in self.token2id:
            self.token2id[token] = len(self.id2token)
            self.id2token.append(token)
        return self.token2id[token]

    def _initial_tokenize(self, text: str) -> List[str]:
        protected = []
        i = 0
        sorted_seeds = sorted(self.v_seed, key=len, reverse=True)
        while i < len(text):
            matched = False
            for seed in sorted_seeds:
                if text.startswith(seed, i):
                    protected.append(seed)
                    i += len(seed)
                    matched = True
                    break
            if not matched:
                protected.append(text[i])
                i += 1

        final_tokens = []
        for seg in protected:
            if seg in self.v_seed:
                final_tokens.append(seg)
            elif seg in self.punctuation_set:
                final_tokens.append(seg)
            elif seg == ZWNJ:
                final_tokens.append(ZWNJ)
            else:
                sub_tokens = AksharaTokenizer.tokenize_devanagari(seg)
                final_tokens.extend(sub_tokens)
        return final_tokens

    def _initial_tokenize_words(self, corpus: str) -> List[List[str]]:
        words = corpus.split()
        return [self._initial_tokenize(word) for word in words]

    def _root_set(self, token: str) -> Set[str]:
        if token in self.v_seed:
            return set()
        return set()

    def _morph(self, a: str, b: str) -> bool:
        roots = self._root_set(a)
        if not roots:
            return True
        return any(self.paradigm.allowed_next(root, a, b) for root in roots)

    def _script_compat(self, a: str, b: str) -> bool:
        sa = script_of(a, self.vocab_info)
        sb = script_of(b, self.vocab_info)
        return sa == sb and sa in {"DEV", "PUN"}

    def _gate(self, a: str, b: str, freq_ab: int) -> bool:
        sa = script_of(a, self.vocab_info)
        sb = script_of(b, self.vocab_info)

        if sa in {"MAL", "FMT"} or sb in {"MAL", "FMT"}:
            return False
        if not self._script_compat(a, b):
            return False
        if b in self.v_seed:
            return False
        if a in self.frozen_strict:
            return False
        if a in self.frozen_ambiguous and freq_ab < THETA:
            return False
        return True

    def _legal(self, a: str, b: str, freq_ab: int) -> bool:
        return self._gate(a, b, freq_ab) and self._morph(a, b)

    def _priority_key(self, a: str, b: str, freq: int) -> Tuple:
        scr = script_of(a, self.vocab_info)
        rank = SCRIPT_RANK.get(scr, 0)
        weighted_freq = freq
        return (-rank, -weighted_freq)

    def train(self, corpus: str, vocab_size: int, latin_budget: int = LATIN_MERGE_BUDGET):
        norm_corpus = normalize(corpus)

        word_seqs = self._initial_tokenize_words(norm_corpus)

        for word in word_seqs:
            for tok in word:
                self._add_token(tok)

        pair_counts = defaultdict(int)
        for word in word_seqs:
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i+1])] += 1

        heap = []
        for (a, b), freq in pair_counts.items():
            if self._legal(a, b, freq):
                prio = self._priority_key(a, b, freq)
                heapq.heappush(heap, (prio, a, b, freq))

        merges_done = 0
        total_merges_budget = vocab_size - len(self.id2token) - latin_budget

        while merges_done < total_merges_budget and heap:
            prio, a, b, snap_freq = heapq.heappop(heap)
            current_freq = pair_counts.get((a, b), 0)
            if current_freq != snap_freq:
                continue
            if not self._legal(a, b, current_freq):
                continue

            new_token = a + b
            new_id = self._add_token(new_token)
            self.merge_rules.append((a, b, new_id))

            new_word_seqs = []
            for word in word_seqs:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and word[i] == a and word[i+1] == b:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_seqs.append(new_word)
            word_seqs = new_word_seqs

            pair_counts.clear()
            for word in word_seqs:
                for i in range(len(word)-1):
                    pair_counts[(word[i], word[i+1])] += 1

            for (a2, b2), f in pair_counts.items():
                if self._legal(a2, b2, f):
                    heapq.heappush(heap, (self._priority_key(a2, b2, f), a2, b2, f))

            merges_done += 1

        if latin_budget > 0:
            lat_sequences = []
            for word in word_seqs:
                lat_seq = [tok for tok in word if script_of(tok, self.vocab_info) == "LAT"]
                if lat_seq:
                    lat_sequences.append(lat_seq)

            lat_pair_counts = defaultdict(int)
            for seq in lat_sequences:
                for i in range(len(seq)-1):
                    lat_pair_counts[(seq[i], seq[i+1])] += 1

            lat_heap = []
            for (a, b), f in lat_pair_counts.items():
                heapq.heappush(lat_heap, (-f, a, b, f))

            latin_merges = 0
            while latin_merges < latin_budget and lat_heap:
                negf, a, b, snap = heapq.heappop(lat_heap)
                if lat_pair_counts.get((a, b), 0) != snap:
                    continue
                new_token = a + b
                new_id = self._add_token(new_token)
                self.merge_rules.append((a, b, new_id))

                new_lat_seqs = []
                for seq in lat_sequences:
                    new_seq = []
                    i = 0
                    while i < len(seq):
                        if i < len(seq)-1 and seq[i] == a and seq[i+1] == b:
                            new_seq.append(new_token)
                            i += 2
                        else:
                            new_seq.append(seq[i])
                            i += 1
                    new_lat_seqs.append(new_seq)
                lat_sequences = new_lat_seqs

                lat_pair_counts.clear()
                for seq in lat_sequences:
                    for i in range(len(seq)-1):
                        lat_pair_counts[(seq[i], seq[i+1])] += 1
                for (a2, b2), f in lat_pair_counts.items():
                    heapq.heappush(lat_heap, (-f, a2, b2, f))
                latin_merges += 1

    def encode(self, text: str) -> List[int]:
        text = normalize(text)
        # Split on whitespace but preserve it as tokens
        import re
        parts = re.split(r'(\s+)', text)
        tokens = []
        for part in parts:
            if part.isspace():
                # Handle each whitespace character individually
                for ch in part:
                    if ch in self.token2id:
                        tokens.append(ch)
                    else:
                        # Add space to vocab on the fly if needed
                        self._add_token(ch)
                        tokens.append(ch)
            else:
                tokens.extend(self._initial_tokenize(part))

        for a, b, new_id in self.merge_rules:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(self.id2token[new_id])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Ensure all tokens have IDs
        result = []
        for tok in tokens:
            if tok not in self.token2id:
                self._add_token(tok)
            result.append(self.token2id[tok])
        return result

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id2token[i] for i in ids]
        return "".join(tokens)

    def encode_bytes(self, data: bytes) -> List[int]:
        ids = []
        i = 0
        while i < len(data):
            # Try to decode as UTF-8 starting at position i
            decoded = False
            for length in range(1, min(5, len(data) - i + 1)):
                chunk = data[i:i+length]
                try:
                    text = chunk.decode("utf-8")
                    # If we get here, we have valid UTF-8
                    # Encode the decoded text
                    ids.extend(self.encode(text))
                    i += length
                    decoded = True
                    break
                except UnicodeDecodeError:
                    continue
            
            if not decoded:
                # Single byte fallback
                byte_val = data[i]
                token = f"<0x{byte_val:02X}>"
                ids.append(self.token2id[token])
                i += 1

        return ids

    def decode_bytes(self, ids: List[int]) -> bytes:
        out = bytearray()
        for tid in ids:
            token = self.id2token[tid]
            if token.startswith("<0x") and token.endswith(">"):
                byte_val = int(token[3:5], 16)
                out.append(byte_val)
            else:
                out.extend(token.encode("utf-8"))
        return bytes(out)

    def vocab_size(self) -> int:
        return len(self.id2token)

    def get_script_embedding_id(self, token: str) -> int:
        scr = script_of(token, self.vocab_info)
        mapping = {"DEV": 0, "LAT": 1, "PUN": 2, "FMT": 3, "MAL": 4}
        return mapping.get(scr, 5)

    def get_paradigm_id(self, token: str) -> int:
        roots = self._root_set(token)
        if not roots:
            return -1
        root = min(roots)
        return hash(root) % (2**31)