"""
Phase 1 integration test – ParadigmStore and Morph constraint.
Verifies that the Morph filter correctly blocks or allows merges
based on a hand‑crafted paradigm store.
"""
import pytest
from NPBPE_tokenizer import NepBPETokenizer, ParadigmStore, normalize, build_sample_store


def test_paradigm_valid_merge_happens():
    """
    Train on a corpus where a valid continuation exists.
    The tokenizer must merge the prefix and its allowed suffix.
    """
    store = build_sample_store()
    tokenizer = NepBPETokenizer(
        paradigm_store=store,
        punctuation_set={"।", " "},
        frozen_strict=set(),
        frozen_ambiguous=set(),
    )
    corpus = "जानु"          # valid merge जा + नु → जानु
    # vocab_size must be large enough to allow merges (base tokens ~260)
    tokenizer.train(corpus, vocab_size=300, latin_budget=0)

    assert "जानु" in tokenizer.token2id, "Expected merged token 'जानु' not found"


def test_paradigm_invalid_merge_blocked():
    """
    Train on a corpus with an invalid continuation.
    The tokenizer must NOT merge the prefix with a token not allowed.
    """
    store = build_sample_store()
    tokenizer = NepBPETokenizer(
        paradigm_store=store,
        punctuation_set={"।", " "},
        frozen_strict=set(),
        frozen_ambiguous=set(),
    )
    corpus = "जामा"          # मा is not allowed after जा; pair (जा, मा) exists inside word
    tokenizer.train(corpus, vocab_size=300, latin_budget=0)

    assert "जामा" not in tokenizer.token2id, "Invalid merge 'जामा' should not have happened"


def test_root_set_non_empty_for_prefix():
    """
    _root_set should return the correct root(s) for a token.
    """
    store = build_sample_store()
    tokenizer = NepBPETokenizer(paradigm_store=store, punctuation_set={"।", " "})
    roots = tokenizer._root_set("जा")
    assert roots == {"जा"}, f"Expected root 'जा' but got {roots}"

    roots = tokenizer._root_set("जानु")
    assert roots == {"जा"}, f"Expected root 'जा' for 'जानु' but got {roots}"

    roots = tokenizer._root_set("random")
    assert roots == set(), "Token 'random' should have empty RootSet"


def test_morph_function():
    """
    Directly test the Morph(a,b) logic.
    """
    store = build_sample_store()
    tokenizer = NepBPETokenizer(paradigm_store=store, punctuation_set={"।", " "})

    # Valid transition
    assert tokenizer._morph("जा", "नु") == True, "जा + नु should be allowed"
    # Invalid transition
    assert tokenizer._morph("जा", "मा") == False, "जा + मा should NOT be allowed"
    # Non-paradigm token (empty RootSet) should always return True
    assert tokenizer._morph("random", "x") == True, "Non-paradigm should be unrestricted"


def test_paradigm_store_direct():
    """
    Unit tests for the ParadigmStore itself.
    """
    store = build_sample_store()

    # allowed_next
    assert store.allowed_next("जा", "जा", "नु") == True
    assert store.allowed_next("जा", "जा", "मा") == False
    # state "जानु" belongs to root "जा", but its allowed next are {"हो", "छ"}
    assert store.allowed_next("जा", "जानु", "नु") == False
    # root not existing
    assert store.allowed_next("xyz", "xyz", "a") == False

    # prefixes_of
    prefixes = store.prefixes_of("जा")
    assert "जा" in prefixes
    assert "जानु" in prefixes
    assert "जायो" in prefixes