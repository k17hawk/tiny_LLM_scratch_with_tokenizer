import os
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from HimalayanTOK_Nepali_64K import PyHimalayanTOK_Nepali_64K


class HimalayanTokenizer(PreTrainedTokenizer):
    """
    Hugging Face tokenizer wrapper for the Rust-based HimalayanTOK_Nepali_64K.

    IMPORTANT: special tokens ([CLS], [SEP], [PAD], [UNK], [MASK]) must already
    exist in the Rust vocab (seeded via initialize_vocab's `punctuation` /
    `seed_morphemes` args, or present in the TSV loaded by load_vocab_tsv).
    This wrapper cannot inject new tokens into the Rust vocab at runtime.
    """

    vocab_files_names = {"vocab_file": "vocab.tsv"}

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        self.rust_tokenizer = PyHimalayanTOK_Nepali_64K()

        if vocab_file is not None:
            if not os.path.isfile(vocab_file):
                raise ValueError(f"vocab_file not found: {vocab_file}")
            self.rust_tokenizer.load_vocab_tsv(vocab_file)

        super().__init__(
            unk_token=unk_token,
            cls_token=cls_token,
            sep_token=sep_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        self._verify_special_tokens_present()

    def _verify_special_tokens_present(self):
        """Fail loudly at load time rather than silently at first encode()."""
        missing = [
            tok
            for tok in (
                self.unk_token,
                self.cls_token,
                self.sep_token,
                self.pad_token,
                self.mask_token,
            )
            if tok is not None and self.rust_tokenizer.vocab_get_id(tok) is None
        ]
        if missing:
            raise ValueError(
                f"Special tokens missing from the Rust vocab: {missing}. "
                "Seed them via initialize_vocab(...) or the loaded TSV before "
                "constructing HimalayanTokenizer."
            )

    # ---- Required overrides ----

    @property
    def vocab_size(self) -> int:
        return self.rust_tokenizer.vocab_size()

    def _tokenize(self, text: str) -> List[str]:
        return self.rust_tokenizer.tokenize_to_strings(text)

    def _convert_token_to_id(self, token: str) -> int:
        token_id = self.rust_tokenizer.vocab_get_id(token)
        if token_id is None:
            return self.rust_tokenizer.vocab_get_id(self.unk_token)
        return token_id

    def _convert_id_to_token(self, index: int) -> str:
        try:
            return self.rust_tokenizer.get_token_surface(index)
        except ValueError:
            return self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        vocab = self.rust_tokenizer.get_vocab_dict()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        vocab_file = os.path.join(save_directory, f"{prefix}vocab.tsv")
        self.rust_tokenizer.save_vocab_tsv(vocab_file)
        return (vocab_file,)

    # ---- Optional but recommended for full compatibility ----

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0, token_ids_1, already_has_special_tokens=True
            )
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]