use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use unicode_normalization::UnicodeNormalization;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::Bound;

// ============================================================================
// Type aliases and basic types
// ============================================================================

type TokenId = usize;
type RootId = usize;
type ByteVal = u8;
type Frequency = u64;

// ============================================================================
// Space marking + extended punctuation (FIX #1 and #2)
// ----------------------------------------------------------------------------
// Word-start markers (DEV_MARKER ▁, LAT_MARKER ▂) are SentencePiece-style. Every
// ASCII space is turned into the marker matching the FOLLOWING character's script
// (plus one dummy prefix at the start of each encoded unit) so that word-start
// context is a real, in-script character that BPE can merge into ▁माया / ▂the,
// instead of a standalone Ġ byte token
// that could never merge (Ġ is a ByteFallback → MAL → Gate rejects it). decode()
// reverses this: every ▁ becomes a space and the single dummy-prefix space is
// dropped.
//
// TWO space markers, SentencePiece-style, chosen by the SCRIPT OF THE FOLLOWING
// character so each folds into its own script's run:
//   DEV_MARKER (▁, U+2581) precedes Devanagari/other -> classified DEV.
//   LAT_MARKER (▂, U+2582) precedes ASCII letters/digits -> classified LAT.
// This is what lets ▂the fold into one Latin token exactly as ▁माया folds for
// Devanagari, removing the +1 standalone-marker penalty English used to pay.
// decode() maps BOTH back to a space, so nothing downstream needs to know which
// marker was used. Devanagari word-starts still get ▁, so a frozen Devanagari
// vocab keeps encoding identically — this change is backward-compatible with an
// already-trained DEV vocabulary and only affects Latin/digit word-starts.
const DEV_MARKER: char = '\u{2581}';
const LAT_MARKER: char = '\u{2582}';

#[inline]
fn is_marker(ch: char) -> bool {
    ch == DEV_MARKER || ch == LAT_MARKER
}

/// Marker to place before a boundary, based on the character that FOLLOWS it.
/// ASCII alphanumeric (Latin letters + ASCII digits) -> LAT marker so it joins
/// the Latin run; everything else (Devanagari, punctuation, end-of-string) ->
/// DEV marker. Must be identical in training and encoding so surfaces agree.
#[inline]
fn marker_for(next: Option<char>) -> char {
    match next {
        Some(c) if c.is_ascii_alphanumeric() => LAT_MARKER,
        _ => DEV_MARKER,
    }
}

// General-Punctuation characters that otherwise fall to MAL and cost 3 byte
// tokens each (en-dash, curly quotes, ellipsis…). classify_char routes these to
// PUN and initialize() seeds them, so each resolves to a single token. Kept to
// the U+2010.. block only: none of these overlap the GPT-2 byte alphabet (which
// tops out well below U+0180), so there is no aliasing to reason about. « » are
// deliberately excluded because 0xAB/0xBB DO alias into the byte alphabet.
const EXTENDED_PUNCT: &[char] = &[
    '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2015}', // ‐‑‒–—―
    '\u{2018}', '\u{2019}', '\u{201A}', '\u{201B}',                         // ‘’‚‛
    '\u{201C}', '\u{201D}', '\u{201E}', '\u{201F}',                         // “”„‟
    '\u{2026}',                                                             // …
];

#[inline]
fn is_unicode_punct(ch: char) -> bool {
    EXTENDED_PUNCT.contains(&ch)
}

// ============================================================================
// GPT-2 byte<->unicode alphabet (Blocker 5)
// ----------------------------------------------------------------------------
// Every raw byte gets a *printable, single-char* surface. Printable ASCII and
// Latin-1 map to themselves; everything else (control bytes, 0x20 space, 0x7F,
// 0x80..0xA0, 0xAD) maps to an obscure char at 256+n. Space (0x20) is NOT
// special-cased -> it becomes U+0120 'Ġ'. With FIX #1 in place spaces are turned
// into ▁ before tokenization, so Ġ is no longer emitted for spaces in practice;
// the byte token still exists and still decodes to a space if an external id
// stream contains it. Decode never needs the inverse map because byte tokens are
// recovered by *type* (Token::ByteFallback), not by surface.
// ============================================================================

/// Reverse the training driver's TSV escaping (\\ -> \, \t -> tab, \n -> nl).
/// Single left-to-right pass so a real backslash isn't mis-paired with a
/// following t/n.
fn unescape_tsv(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\u{0}'; 256];
    let mut n: u32 = 0;
    for b in 0u8..=255u8 {
        let code: u32 = if b.is_ascii_graphic()               // 0x21..=0x7E
            || (0xA1u8..=0xAC).contains(&b)                    // ¡..¬
            || (0xAEu8..=0xFF).contains(&b)                    // ®..ÿ
        {
            b as u32
        } else {
            let c = 256 + n;
            n += 1;
            c
        };
        table[b as usize] = char::from_u32(code).expect("valid scalar");
    }
    table
}

// ============================================================================
// Script types (total assignment per §2.1)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    DEV,
    LAT,
    PUN,
    FMT,
    MAL,
}

impl Script {
    fn rank(&self) -> u8 {
        match self {
            Script::DEV => 2,
            Script::PUN => 1,
            _ => 0,
        }
    }
}

// ============================================================================
// Phase 1: Normalization N  (Blocker 4 — sequence rewriter)
// ----------------------------------------------------------------------------
// N(s) = StripZWJ ∘ Fold_O ∘ NFC(s).
// Fold_O is now a set of string->string rules applied leftmost-longest, so it
// can collapse multi-codepoint sequences (e.g. सङ्ग -> संग) that a char->char
// table cannot. ZWJ (U+200D) is stripped; ZWNJ (U+200C) is preserved.
//
// Idempotence caveat: N(N(s)) = N(s) holds only if every rule's REPLACEMENT is
// already NFC-stable. संग is NFC-stable, so the shipped rules are fine; if you
// add a rule whose output is not NFC-stable, either fix the rule or re-run
// `.nfc()` on the final result.
// ============================================================================

pub struct Normalizer {
    fold_rules: Vec<(Vec<char>, String)>, // (pattern chars, replacement), longest-first
}

impl Normalizer {
    pub fn new(fold_rules: Vec<(String, String)>) -> Self {
        let mut rules: Vec<(Vec<char>, String)> = fold_rules
            .into_iter()
            .map(|(p, r)| (p.chars().collect::<Vec<char>>(), r))
            .collect();
        // Longer patterns first so ङ्ग wins over ङ at the same position.
        rules.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Self { fold_rules: rules }
    }

    pub fn normalize(&self, s: &str) -> String {
        let nfc: String = s.nfc().collect();
        let chars: Vec<char> = nfc.chars().collect();
        let mut result = String::with_capacity(nfc.len());
        let mut i = 0;

        while i < chars.len() {
            let mut matched = false;
            for (pat, rep) in &self.fold_rules {
                let plen = pat.len();
                if plen > 0 && i + plen <= chars.len() && chars[i..i + plen] == pat[..] {
                    result.push_str(rep);
                    i += plen;
                    matched = true;
                    break;
                }
            }
            if !matched {
                let ch = chars[i];
                if ch != '\u{200D}' {
                    // StripZWJ; ZWNJ (U+200C) is deliberately preserved.
                    result.push(ch);
                }
                i += 1;
            }
        }
        result
    }
}

// ============================================================================
// Phase 2: Akshara DFA
// ----------------------------------------------------------------------------
// (Unchanged. If you later swap this for UAX#29 grapheme segmentation via the
// unicode-segmentation crate, verify the resolved version implements GB9c/InCB
// so conjuncts like क्ष are not split, and test ZWNJ-embedded aksharas.)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Akshara {
    pub surface: String,
    pub root_set: Vec<RootId>,
}

#[derive(Debug, Clone)]
pub struct AksharaDFA {
    pub transitions: HashMap<(usize, char), usize>,
    pub accepting_states: HashSet<usize>,
    initial_state: usize,
}

impl AksharaDFA {
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            accepting_states: HashSet::new(),
            initial_state: 0,
        }
    }

    /// Debug: check if a transition exists
    pub fn has_transition(&self, state: usize, ch: char) -> Option<usize> {
        self.transitions.get(&(state, ch)).copied()
    }

    /// Debug: get all transitions from a state
    pub fn get_transitions_from(&self, state: usize) -> Vec<(char, usize, bool)> {
        self.transitions
            .iter()
            .filter(|((s, _), _)| *s == state)
            .map(|((_, ch), &next)| (*ch, next, self.accepting_states.contains(&next)))
            .collect()
    }

    /// Tokenize using maximal munch: find the LONGEST valid akshara at each position
    pub fn tokenize(&self, dev_text: &str) -> Vec<Akshara> {
        let mut aksharas = Vec::new();
        let chars: Vec<char> = dev_text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut state = self.initial_state;
            let mut last_accepting_pos: Option<usize> = None;
            let mut j = i;

            while j < chars.len() {
                let ch = chars[j];
                if let Some(&next_state) = self.transitions.get(&(state, ch)) {
                    state = next_state;
                    if self.accepting_states.contains(&state) {
                        last_accepting_pos = Some(j + 1);
                    }
                    j += 1;
                } else {
                    break;
                }
            }

            if let Some(end) = last_accepting_pos {
                let akshara_str: String = chars[i..end].iter().collect();
                aksharas.push(Akshara {
                    surface: akshara_str,
                    root_set: Vec::new(), // populated later via ParadigmRegistry (§2 tagging)
                });
                i = end;
            } else {
                let ch = chars[i];
                aksharas.push(Akshara {
                    surface: ch.to_string(),
                    root_set: Vec::new(),
                });
                i += 1;
            }
        }
        aksharas
    }
}

// ============================================================================
// Token types and vocabulary
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    Akshara(Akshara),
    Punctuation(String),
    ZWNJ,
    ByteFallback(ByteVal),
    SeededMorpheme(String),
    MergedToken(Vec<TokenId>),
    /// Latin/ASCII alphanumeric base unit (Phase 4). These MUST exist as their
    /// own script class: if 'h' were only a ByteFallback token its script would
    /// be MAL, and the Gate's atomic-class clause would forbid every Latin
    /// merge. Seeded before the byte alphabet so it wins the surface key.
    Latin(String),
    /// Reconstructed from a saved vocab (surface only, no merge/child history).
    /// Enough for encode + decode; NOT enough to correctly resume training.
    Loaded(String),
}

impl Token {
    /// Default script by variant. NOTE: for MergedToken this is only a
    /// placeholder — Vocabulary::create_merged overrides id_to_script with the
    /// left child's actual script, and Vocabulary::get_script is authoritative.
    /// For Loaded tokens the script is set directly during load, not from here.
    pub fn script(&self) -> Script {
        match self {
            Token::Akshara(_) => Script::DEV,
            Token::Punctuation(_) => Script::PUN,
            Token::ZWNJ => Script::FMT,
            Token::ByteFallback(_) => Script::MAL,
            Token::SeededMorpheme(_) => Script::DEV,
            Token::MergedToken(_) => Script::DEV,
            Token::Latin(_) => Script::LAT,
            Token::Loaded(_) => Script::DEV,
        }
    }
}

// ============================================================================
// Vocabulary management (surface string as primary key)
// ============================================================================

#[derive(Default)]
pub struct Vocabulary {
    tokens: Vec<Arc<Token>>,
    surface_to_id: HashMap<String, TokenId>,
    id_to_script: HashMap<TokenId, Script>,
    v_strict: HashSet<TokenId>,
    v_ambiguous: HashSet<TokenId>,
    token_to_root_set: HashMap<TokenId, Vec<RootId>>,
    surfaces: HashMap<TokenId, String>,
    /// Longest surface in CHARS — the cap for greedy longest-match encoding.
    max_surface_len: usize,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn initialize(
        &mut self,
        base_aksharas: Vec<Akshara>,
        seed_morphemes: Vec<String>,
        punctuation: Vec<String>,
        v_strict: HashSet<String>,
        v_ambiguous: HashSet<String>,
        byte_encoder: &[char; 256],
    ) {
        for akshara in base_aksharas {
            let surface = akshara.surface.clone();
            let token = Arc::new(Token::Akshara(akshara));
            self.add_token(token, surface, false, false);
        }
        for morph in seed_morphemes {
            let surface = morph.clone();
            let is_strict = v_strict.contains(&morph);
            let is_ambiguous = v_ambiguous.contains(&morph);
            let token = Arc::new(Token::SeededMorpheme(morph));
            self.add_token(token, surface, is_strict, is_ambiguous);
        }

        // Seed BOTH space markers as freely-mergeable base tokens.
        //   ▁ DEV-scripted: greedy fallback for a run-initial ▁, and BPE learns
        //     ▁माया in the Devanagari pass.
        //   ▂ LAT-scripted: same role for the Latin pass, learning ▂the.
        // Neither is strict/ambiguous, so the Gate never blocks them as a left
        // element.
        self.add_token(
            Arc::new(Token::SeededMorpheme(DEV_MARKER.to_string())),
            DEV_MARKER.to_string(),
            false,
            false,
        );
        self.add_token(
            Arc::new(Token::Latin(LAT_MARKER.to_string())),
            LAT_MARKER.to_string(),
            false,
            false,
        );

        for punct in punctuation {
            let surface = punct.clone();
            let token = Arc::new(Token::Punctuation(punct));
            self.add_token(token, surface, false, false);
        }

        // FIX #2: seed extended (non-ASCII) punctuation as PUN tokens so each
        // resolves to a single token instead of 3 byte-fallback tokens. Seeded
        // BEFORE the byte alphabet, though none of these overlap it anyway.
        for &ch in EXTENDED_PUNCT {
            let surface = ch.to_string();
            let token = Arc::new(Token::Punctuation(surface.clone()));
            self.add_token(token, surface, false, false);
        }

        // Phase 4 base alphabet: Latin letters + ASCII digits as LAT-scripted
        // tokens. Seeded BEFORE the byte alphabet so these surfaces resolve to
        // LAT (mergeable) rather than MAL byte tokens (never mergeable).
        for ch in ('a'..='z').chain('A'..='Z').chain('0'..='9') {
            let surface = ch.to_string();
            let token = Arc::new(Token::Latin(surface.clone()));
            self.add_token(token, surface, false, false);
        }

        // Byte-fallback alphabet, GPT-2 style. Punctuation and Latin are added
        // *before* this, so an ASCII-graphic byte that collides with an existing
        // surface aliases to that token; harmless — the surface still decodes
        // correctly, and single-byte ASCII never appears as a UTF-8 continuation
        // byte, so the multibyte flush is unaffected.
        for byte_val in 0u8..=255 {
            let surface = byte_encoder[byte_val as usize].to_string();
            let token = Arc::new(Token::ByteFallback(byte_val));
            self.add_token(token, surface, false, false);
        }
        let zwnj = Arc::new(Token::ZWNJ);
        self.add_token(zwnj, "\u{200C}".to_string(), false, false);
    }

    fn add_token(
        &mut self,
        token: Arc<Token>,
        surface: String,
        is_strict: bool,
        is_ambiguous: bool,
    ) -> TokenId {
        if let Some(&existing_id) = self.surface_to_id.get(&surface) {
            return existing_id;
        }

        let id = self.tokens.len();
        let script = token.script();
        let clen = surface.chars().count();
        if clen > self.max_surface_len {
            self.max_surface_len = clen;
        }
        self.surface_to_id.insert(surface.clone(), id);
        self.id_to_script.insert(id, script);
        self.tokens.push(token);
        self.surfaces.insert(id, surface);

        if is_strict {
            self.v_strict.insert(id);
        }
        if is_ambiguous {
            self.v_ambiguous.insert(id);
        }
        if let Token::Akshara(a) = &*self.tokens[id] {
            self.token_to_root_set.insert(id, a.root_set.clone());
        }
        id
    }

    /// §2 tagging: RootSet(α) = { root : α ∈ prefixes(P(root)) }.
    /// Call AFTER paradigms are loaded and BEFORE training. Costs nothing
    /// behaviorally if no paradigms are registered (all root sets stay empty,
    /// Morph stays in its RootSet=∅ ⇒ 1 branch). This is the wire that makes the
    /// paradigm machinery non-inert once a real FST populates the registry.
    pub fn assign_roots_from_registry(&mut self, registry: &ParadigmRegistry) {
        for id in 0..self.tokens.len() {
            let eligible = matches!(
                &*self.tokens[id],
                Token::Akshara(_) | Token::SeededMorpheme(_) | Token::MergedToken(_)
            );
            if !eligible {
                continue;
            }
            if let Some(surface) = self.surfaces.get(&id).cloned() {
                let roots = registry.get_root_set(&surface);
                if !roots.is_empty() {
                    self.token_to_root_set.insert(id, roots);
                }
            }
        }
    }

    pub fn get_id_by_surface(&self, surface: &str) -> Option<TokenId> {
        self.surface_to_id.get(surface).copied()
    }

    pub fn max_surface_len(&self) -> usize {
        self.max_surface_len
    }

    /// Rebuild the vocabulary from saved (id, surface) pairs — enough for
    /// encode + decode. Byte tokens are recovered as ByteFallback via the byte
    /// alphabet; everything else becomes a Loaded surface token. v_strict /
    /// v_ambiguous / root sets are NOT restored, so a loaded vocab can tokenize
    /// but should not be used to resume training.
    ///
    /// NOTE (FIX #1): a vocab trained with the space marker will contain ▁ and
    /// ▁-prefixed surfaces; those load fine as Loaded (DEV) tokens and encode /
    /// decode correctly. A vocab trained BEFORE this change has no ▁ token, so
    /// loading it here makes every space byte-fall-back to 3 tokens. Retrain and
    /// regenerate the TSV after adopting this file.
    ///
    /// `pairs` must have contiguous ids 0..N; they are sorted defensively.
    pub fn load_from_pairs(&mut self, mut pairs: Vec<(TokenId, String)>, byte_decoder: &HashMap<char, u8>) {
        self.tokens.clear();
        self.surface_to_id.clear();
        self.id_to_script.clear();
        self.v_strict.clear();
        self.v_ambiguous.clear();
        self.token_to_root_set.clear();
        self.surfaces.clear();
        self.max_surface_len = 0;

        pairs.sort_by_key(|(id, _)| *id);

        for (expected_id, surface) in pairs {
            let id = self.tokens.len();
            debug_assert_eq!(id, expected_id, "vocab ids must be contiguous from 0");

            // Classify the surface. Order matters: Latin/digit chars are checked
            // BEFORE the byte alphabet, because 'a' is both a valid Latin base
            // token and byte 0x61 — it must come back as LAT (mergeable), not
            // MAL. Merged surfaces are always >= 2 chars, so the single-char
            // checks never misfire on a real merge.
            let mut token = Arc::new(Token::Loaded(surface.clone()));
            let first = surface.chars().next();
            let mut chs = surface.chars();
            if let (Some(ch), None) = (chs.next(), chs.next()) {
                // Single char.
                if ch == DEV_MARKER {
                    token = Arc::new(Token::SeededMorpheme(surface.clone())); // DEV
                } else if ch == LAT_MARKER {
                    token = Arc::new(Token::Latin(surface.clone())); // LAT
                } else if ch.is_ascii_alphanumeric() {
                    token = Arc::new(Token::Latin(surface.clone()));
                } else if let Some(&b) = byte_decoder.get(&ch) {
                    token = Arc::new(Token::ByteFallback(b));
                }
            } else if first == Some(LAT_MARKER)
                && surface.chars().skip(1).all(|c| c.is_ascii_alphanumeric())
            {
                // ▂-prefixed Latin merge (e.g. ▂the) -> LAT. The ▂ is non-ASCII,
                // so the plain all-ASCII check below would miss it and wrongly
                // tag this DEV; handle it explicitly.
                token = Arc::new(Token::Latin(surface.clone()));
            } else if surface.chars().all(|c| c.is_ascii_alphanumeric()) {
                // Multi-char pure-ASCII surface = a bare Latin merge.
                token = Arc::new(Token::Latin(surface.clone()));
            }
            // Everything else (▁-prefixed Devanagari merges, plain Devanagari
            // merges, abbreviations) stays Loaded -> DEV, which is correct for
            // encode/decode.

            let clen = surface.chars().count();
            if clen > self.max_surface_len {
                self.max_surface_len = clen;
            }
            let script = token.script();
            self.surface_to_id.insert(surface.clone(), id);
            self.id_to_script.insert(id, script);
            self.surfaces.insert(id, surface);
            self.tokens.push(token);
        }

        // A vocab trained before the two-marker change won't contain ▂ (and an
        // ancient one might lack ▁). Append any missing marker as its base token
        // so encode works and a Latin retrain has the ▂ base to merge from. Ids
        // stay contiguous because we append at the current length.
        for (m, is_lat) in [(DEV_MARKER, false), (LAT_MARKER, true)] {
            let s = m.to_string();
            if !self.surface_to_id.contains_key(&s) {
                let id = self.tokens.len();
                let tok: Arc<Token> = if is_lat {
                    Arc::new(Token::Latin(s.clone()))
                } else {
                    Arc::new(Token::SeededMorpheme(s.clone()))
                };
                let script = tok.script();
                self.surface_to_id.insert(s.clone(), id);
                self.id_to_script.insert(id, script);
                self.surfaces.insert(id, s);
                self.tokens.push(tok);
            }
        }
    }

    pub fn get_token(&self, id: TokenId) -> Option<&Arc<Token>> {
        self.tokens.get(id)
    }

    pub fn get_script(&self, id: TokenId) -> Script {
        *self.id_to_script.get(&id).unwrap_or(&Script::MAL)
    }

    pub fn is_strict(&self, id: TokenId) -> bool {
        self.v_strict.contains(&id)
    }

    pub fn is_ambiguous(&self, id: TokenId) -> bool {
        self.v_ambiguous.contains(&id)
    }

    pub fn get_root_set(&self, id: TokenId) -> &[RootId] {
        self.token_to_root_set
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Blocker 3: a merged token inherits the LEFT child's script (ScriptCompat
    /// guarantees both children share it). Guarded so that if `merged_surface`
    /// already exists, we return the existing id WITHOUT clobbering its script
    /// or root set.
    pub fn create_merged(&mut self, a: TokenId, b: TokenId, merged_root_set: Vec<RootId>) -> TokenId {
        let surface_a = self.get_surface(a).unwrap_or_default();
        let surface_b = self.get_surface(b).unwrap_or_default();
        let merged_surface = format!("{}{}", surface_a, surface_b);

        let existed = self.surface_to_id.contains_key(&merged_surface);
        let merged = Arc::new(Token::MergedToken(vec![a, b]));
        let id = self.add_token(merged, merged_surface, false, false);

        if !existed {
            if let Some(&script) = self.id_to_script.get(&a) {
                self.id_to_script.insert(id, script);
            }
            self.token_to_root_set.insert(id, merged_root_set);
        }
        id
    }

    pub fn get_surface(&self, id: TokenId) -> Option<String> {
        self.surfaces.get(&id).cloned()
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Debug: get all surfaces
    pub fn get_all_surfaces(&self) -> Vec<(TokenId, String)> {
        self.surfaces.iter().map(|(&id, s)| (id, s.clone())).collect()
    }

    /// Debug: check if surface exists
    pub fn contains_surface(&self, surface: &str) -> bool {
        self.surface_to_id.contains_key(surface)
    }
}

// ============================================================================
// Paradigm morphology (FST-backed)
// ============================================================================

pub struct Paradigm {
    pub root_id: RootId,
    pub allowed_transitions: HashMap<String, HashSet<String>>,
}

impl Paradigm {
    pub fn new(root_id: RootId) -> Self {
        Self {
            root_id,
            allowed_transitions: HashMap::new(),
        }
    }

    pub fn allows(&self, prefix: &str, continuation: &str) -> bool {
        self.allowed_transitions
            .get(prefix)
            .map(|allowed| allowed.contains(continuation))
            .unwrap_or(false)
    }

    pub fn has_prefix(&self, prefix: &str) -> bool {
        self.allowed_transitions.contains_key(prefix)
    }
}

#[derive(Default)]
pub struct ParadigmRegistry {
    paradigms: Vec<Paradigm>,
    root_to_paradigm: HashMap<RootId, usize>,
}

impl ParadigmRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_paradigm(&mut self, paradigm: Paradigm) {
        let idx = self.paradigms.len();
        self.root_to_paradigm.insert(paradigm.root_id, idx);
        self.paradigms.push(paradigm);
    }

    pub fn get_root_set(&self, prefix: &str) -> Vec<RootId> {
        self.paradigms
            .iter()
            .filter(|p| p.has_prefix(prefix))
            .map(|p| p.root_id)
            .collect()
    }

    pub fn check_allowed(&self, root_id: RootId, prefix: &str, continuation: &str) -> bool {
        if let Some(&idx) = self.root_to_paradigm.get(&root_id) {
            self.paradigms[idx].allows(prefix, continuation)
        } else {
            false
        }
    }
}

// ============================================================================
// Phase 3: Constrained BPE
// ============================================================================

/// Decrement a pair's count by `by`; prune the entry entirely when it reaches 0
/// so that pair_freqs.keys() never iterates dead entries.
fn dec_pair(freqs: &mut HashMap<(TokenId, TokenId), Frequency>, key: (TokenId, TokenId), by: Frequency) {
    if let Some(f) = freqs.get_mut(&key) {
        *f = f.saturating_sub(by);
        if *f == 0 {
            freqs.remove(&key);
        }
    }
}

fn inc_pair(freqs: &mut HashMap<(TokenId, TokenId), Frequency>, key: (TokenId, TokenId), by: Frequency) {
    *freqs.entry(key).or_insert(0) += by;
}

/// One deduplicated word TYPE: its initial token sequence and how many times it
/// occurred in the corpus. This is the RAM + speed fix — the trainer iterates a
/// few million unique word types, not billions of raw token positions, and pair
/// frequencies are weighted by `count`.
struct Word {
    tokens: Vec<TokenId>,
    count: Frequency,
}

pub struct Corpus {
    words: Vec<Word>,
    pair_freqs: HashMap<(TokenId, TokenId), Frequency>,
    vocab_budget: usize,
}

impl Corpus {
    /// Build from a word-frequency dictionary (streaming path). Consumes `counts`.
    pub fn from_word_counts(counts: HashMap<Vec<TokenId>, Frequency>, vocab_budget: usize) -> Self {
        let words: Vec<Word> = counts
            .into_iter()
            .map(|(tokens, count)| Word { tokens, count })
            .collect();
        let mut corpus = Self {
            words,
            pair_freqs: HashMap::new(),
            vocab_budget,
        };
        corpus.recompute_all_frequencies();
        corpus
    }

    /// Build from raw sequences with no dedup (each sequence has count 1).
    /// Kept for the in-memory `train_bpe` / `train_from_text` path; do NOT use
    /// this for very large corpora — use `from_word_counts` via `train_from_file`.
    pub fn from_sequences(sequences: Vec<Vec<TokenId>>, vocab_budget: usize) -> Self {
        let words: Vec<Word> = sequences
            .into_iter()
            .map(|tokens| Word { tokens, count: 1 })
            .collect();
        let mut corpus = Self {
            words,
            pair_freqs: HashMap::new(),
            vocab_budget,
        };
        corpus.recompute_all_frequencies();
        corpus
    }

    /// Full scan — used ONCE at construction only. Weighted by word count.
    fn recompute_all_frequencies(&mut self) {
        self.pair_freqs.clear();
        for w in &self.words {
            for window in w.tokens.windows(2) {
                *self.pair_freqs
                    .entry((window[0], window[1]))
                    .or_insert(0) += w.count;
            }
        }
    }

    pub fn get_freq(&self, a: TokenId, b: TokenId) -> Frequency {
        self.pair_freqs.get(&(a, b)).copied().unwrap_or(0)
    }

    /// Blocker 6: incremental merge. Applies (a,b)->new_id across the corpus and
    /// updates pair frequencies with local deltas only (no full recompute).
    /// Returns the deduplicated set of AFFECTED pairs — both the newly formed
    /// pairs AND the decremented neighbours — so the caller re-checks Legal and
    /// pushes a fresh heap entry for each. Re-pushing the *decremented*
    /// neighbours is the fix for the "legal pair silently vanishes" bug: their
    /// old snapshots are now stale and would be discarded on pop with nothing to
    /// replace them.
    ///
    /// Borrow note: `self.words.iter_mut()` and `&mut self.pair_freqs` are
    /// disjoint fields, so the split borrow compiles. Keep the freq mutation as
    /// free functions (dec_pair/inc_pair) — routing it through a `&mut self`
    /// method would re-borrow all of `self` and fail to compile.
    pub fn apply_merge(&mut self, a: TokenId, b: TokenId, new_id: TokenId) -> Vec<(TokenId, TokenId)> {
        let mut touched: HashSet<(TokenId, TokenId)> = HashSet::new();

        for w in self.words.iter_mut() {
            let cnt = w.count;
            let mut i = 0;
            while i + 1 < w.tokens.len() {
                if w.tokens[i] == a && w.tokens[i + 1] == b {
                    // Destroy old neighbour pairs (weighted by this word's count).
                    if i > 0 {
                        let l = w.tokens[i - 1];
                        dec_pair(&mut self.pair_freqs, (l, a), cnt);
                        touched.insert((l, a));
                    }
                    if i + 2 < w.tokens.len() {
                        let r = w.tokens[i + 2];
                        dec_pair(&mut self.pair_freqs, (b, r), cnt);
                        touched.insert((b, r));
                    }

                    // Consume (a, b) -> new_id.
                    w.tokens[i] = new_id;
                    w.tokens.remove(i + 1);

                    // Form new neighbour pairs.
                    if i > 0 {
                        let l = w.tokens[i - 1];
                        inc_pair(&mut self.pair_freqs, (l, new_id), cnt);
                        touched.insert((l, new_id));
                    }
                    if i + 1 < w.tokens.len() {
                        let r = w.tokens[i + 1];
                        inc_pair(&mut self.pair_freqs, (new_id, r), cnt);
                        touched.insert((new_id, r));
                    }
                    // Do NOT advance i: staying lets an overlapping (a,b) newly
                    // beginning at i be caught.
                } else {
                    i += 1;
                }
            }
        }

        // (a, b) is fully consumed corpus-wide.
        self.pair_freqs.remove(&(a, b));
        touched.into_iter().collect()
    }
}

#[derive(Debug, Clone)]
struct MergeCandidate {
    a: TokenId,
    b: TokenId,
    priority_key: (u8, u64),
    freq_snapshot: Frequency,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.priority_key == other.priority_key && self.a == other.a && self.b == other.b
    }
}
impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Blocker 1: NO .reverse(). BinaryHeap is a max-heap, so this pops the LARGEST
// (ScriptRank, freq) first = §3.3 argmax. Ties are broken deterministically by
// (a, b) so trained vocabularies are reproducible, and Ord is consistent with
// the PartialEq above.
impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_key
            .cmp(&other.priority_key)
            .then_with(|| self.a.cmp(&other.a))
            .then_with(|| self.b.cmp(&other.b))
    }
}

pub struct ConstrainedBPETrainer {
    vocab: Vocabulary,
    paradigm_registry: ParadigmRegistry,
    pub theta: Frequency,
    /// PHASE 4. When true this is the Latin secondary pass: unconstrained BPE
    /// over LAT runs only. No V_seed/V_strict/V_ambiguous (those are Devanagari
    /// morphemes) and no Morph (no paradigms for English) — pure frequency.
    /// When false it is the Phase-3 constrained Devanagari pass, unchanged.
    pub latin_pass: bool,
}

impl ConstrainedBPETrainer {
    pub fn new(vocab: Vocabulary, paradigm_registry: ParadigmRegistry) -> Self {
        Self {
            vocab,
            paradigm_registry,
            theta: 100,
            latin_pass: false,
        }
    }

    fn script_compat(&self, a: TokenId, b: TokenId) -> bool {
        let sa = self.vocab.get_script(a);
        let sb = self.vocab.get_script(b);
        if self.latin_pass {
            // Phase 4: Latin only. Fully separate from the DEV vocabulary, so no
            // DEV/LAT token can ever form (the v2 "no token spans two scripts"
            // guarantee still holds). The ▁ marker is DEV, so it never enters a
            // LAT merge — Latin word-starts keep a standalone ▁.
            return sa == sb && sa == Script::LAT;
        }
        sa == sb && (sa == Script::DEV || sa == Script::PUN)
    }

    fn gate(&self, a: TokenId, b: TokenId, freq: Frequency) -> bool {
        let sa = self.vocab.get_script(a);
        let sb = self.vocab.get_script(b);

        if sa == Script::MAL || sa == Script::FMT || sb == Script::MAL || sb == Script::FMT {
            return false;
        }
        if !self.script_compat(a, b) {
            return false;
        }
        if self.latin_pass {
            return true; // unconstrained: frequency alone decides
        }
        if self.vocab.is_strict(b) || self.vocab.is_ambiguous(b) {
            return false;
        }
        if self.vocab.is_strict(a) {
            return false;
        }
        if self.vocab.is_ambiguous(a) && freq < self.theta {
            return false;
        }
        true
    }

    fn morph(&self, a: TokenId, b: TokenId) -> bool {
        if self.latin_pass {
            return true; // no paradigms for Latin
        }
        let root_set = self.vocab.get_root_set(a);
        if root_set.is_empty() {
            return true; // non-paradigm token: defer to Gate + Freq (§3.2 fix #2)
        }
        let surface_a = self.vocab.get_surface(a).unwrap_or_default();
        let surface_b = self.vocab.get_surface(b).unwrap_or_default();

        root_set
            .iter()
            .any(|&root_id| self.paradigm_registry.check_allowed(root_id, &surface_a, &surface_b))
    }

    fn legal(&self, a: TokenId, b: TokenId, freq: Frequency) -> bool {
        self.gate(a, b, freq) && self.morph(a, b)
    }

    fn narrow_root_set(&self, a: TokenId, b: TokenId) -> Vec<RootId> {
        if self.latin_pass {
            return Vec::new();
        }
        let root_set_a = self.vocab.get_root_set(a);
        if root_set_a.is_empty() {
            return Vec::new();
        }
        let surface_a = self.vocab.get_surface(a).unwrap_or_default();
        let surface_b = self.vocab.get_surface(b).unwrap_or_default();

        root_set_a
            .iter()
            .filter(|&&root_id| {
                self.paradigm_registry
                    .check_allowed(root_id, &surface_a, &surface_b)
            })
            .copied()
            .collect()
    }

    fn priority_key(&self, a: TokenId, _b: TokenId, freq: Frequency) -> (u8, u64) {
        (self.vocab.get_script(a).rank(), freq)
    }

    /// Train until `budget` (total vocab size) is reached or no admissible merge
    /// remains. `progress_every` merges, prints a timing line to stderr (0=off).
    /// The budget is a TOTAL vocab-size target, not a merge count — so for the
    /// Phase-4 pass you pass (dev_budget + lat_budget).
    pub fn train(&mut self, corpus: &mut Corpus, budget: usize, progress_every: u64) {
        let t0 = std::time::Instant::now();
        let start_vocab = self.vocab.len();
        let tag = if self.latin_pass { "LAT" } else { "DEV" };

        let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();
        self.initialize_heap(corpus, &mut heap);
        if progress_every > 0 {
            eprintln!(
                "[train:{}] heap seeded with {} admissible pairs in {:.1}s",
                tag,
                heap.len(),
                t0.elapsed().as_secs_f64()
            );
        }

        let mut merges: u64 = 0;

        while self.vocab.len() < budget {
            let mut applied = false;

            while let Some(candidate) = heap.pop() {
                let current_freq = corpus.get_freq(candidate.a, candidate.b);

                // §3.4 lazy invalidation.
                if current_freq != candidate.freq_snapshot {
                    continue;
                }
                if current_freq == 0 {
                    continue;
                }
                if !self.legal(candidate.a, candidate.b, current_freq) {
                    continue;
                }

                // Mint the merged token (mutable vocab borrow ends here).
                let narrowed = self.narrow_root_set(candidate.a, candidate.b);
                let new_id = self.vocab.create_merged(candidate.a, candidate.b, narrowed);

                // Incremental corpus update -> only the affected pairs.
                let touched = corpus.apply_merge(candidate.a, candidate.b, new_id);

                // Re-check Legal and push a fresh entry for every affected pair.
                for (x, y) in touched {
                    let f = corpus.get_freq(x, y);
                    if f > 0 && self.legal(x, y, f) {
                        heap.push(MergeCandidate {
                            a: x,
                            b: y,
                            priority_key: self.priority_key(x, y, f),
                            freq_snapshot: f,
                        });
                    }
                }

                merges += 1;
                applied = true;

                if progress_every > 0 && merges % progress_every == 0 {
                    let secs = t0.elapsed().as_secs_f64();
                    eprintln!(
                        "[train:{}] {} merges | vocab {} | heap {} | {:.1}s | {:.0} merges/s",
                        tag,
                        merges,
                        self.vocab.len(),
                        heap.len(),
                        secs,
                        merges as f64 / secs.max(1e-9)
                    );
                }
                break;
            }

            if !applied {
                break; // heap exhausted of admissible pairs
            }
        }

        if progress_every > 0 {
            eprintln!(
                "[train:{}] done: {} merges ({} -> {} vocab) in {:.1}s",
                tag,
                merges,
                start_vocab,
                self.vocab.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    fn initialize_heap(&self, corpus: &Corpus, heap: &mut BinaryHeap<MergeCandidate>) {
        for &(a, b) in corpus.pair_freqs.keys() {
            let freq = corpus.get_freq(a, b);
            if self.legal(a, b, freq) {
                heap.push(MergeCandidate {
                    a,
                    b,
                    priority_key: self.priority_key(a, b, freq),
                    freq_snapshot: freq,
                });
            }
        }
    }
}

// ============================================================================
// Main tokenizer
// ============================================================================

#[allow(dead_code)]
pub struct HimalayanTOK_Nepali_64K {
    pub normalizer: Normalizer,
    pub akshara_dfa: AksharaDFA,
    pub vocab: Vocabulary,
    pub paradigm_registry: ParadigmRegistry,
    byte_encoder: [char; 256],
    trainer: Option<ConstrainedBPETrainer>,
    paradigm_embedding: HashMap<TokenId, Option<RootId>>,
    /// Longest punctuation-containing seed surface, in CHARS. Caps the atomic
    /// abbreviation pre-scan in encode_normalized. 0 = no such seeds, so the
    /// pre-scan is skipped entirely. Set by PyHimalayanTOK_Nepali_64K::initialize_vocab.
    seed_max_len: usize,
}

impl HimalayanTOK_Nepali_64K {
    pub fn new(fold_rules: Vec<(String, String)>, paradigm_registry: ParadigmRegistry) -> Self {
        Self {
            normalizer: Normalizer::new(fold_rules),
            akshara_dfa: AksharaDFA::new(),
            vocab: Vocabulary::new(),
            paradigm_registry,
            byte_encoder: bytes_to_unicode(),
            trainer: None,
            paradigm_embedding: HashMap::new(),
            seed_max_len: 0,
        }
    }

    pub fn encode(&self, s: &str) -> Vec<TokenId> {
        let normalized = self.normalizer.normalize(s);
        self.encode_normalized(&normalized)
    }

    /// Rebuild the vocab from saved (id, surface) pairs. Encode/decode-ready;
    /// not training-ready (strict/ambiguous/roots are not restored).
    ///
    /// CRITICAL: also recompute `seed_max_len` here. It is normally set in
    /// PyHimalayanTOK_Nepali_64K::initialize_vocab, but a load-only workflow (encode /
    /// fertility scripts that call load_vocab_tsv and never initialize_vocab)
    /// would otherwise leave it at 0 — which makes try_emit_atomic_seed early-
    /// return on every call, so seeded abbreviations like गा.वि.स. silently
    /// fragment even though they ARE in the loaded vocab. Scan the loaded
    /// surfaces for the same punctuation-bearing, multi-char criterion used at
    /// seed time.
    pub fn load_vocab(&mut self, pairs: Vec<(TokenId, String)>) {
        // Invert the byte alphabet: surface char -> byte value.
        let mut byte_decoder: HashMap<char, u8> = HashMap::with_capacity(256);
        for (b, &ch) in self.byte_encoder.iter().enumerate() {
            byte_decoder.insert(ch, b as u8);
        }

        // Recompute the atomic pre-scan cap from the surfaces about to load.
        let mut seed_max_len = 0usize;
        for (_, surface) in &pairs {
            if surface.contains('.') || surface.contains('\u{0964}') {
                let clen = surface.chars().count();
                if clen >= 2 && clen > seed_max_len {
                    seed_max_len = clen;
                }
            }
        }
        self.seed_max_len = seed_max_len;

        self.vocab.load_from_pairs(pairs, &byte_decoder);
    }

    /// FIX (atomic seeds): emit a single pre-seeded, punctuation-containing
    /// surface (e.g. "गा.वि.स.") from position `start` in the already-built
    /// marked char slice. Cross-script abbreviations can't be reached by the
    /// per-script run splitter — the run boundary at each '.' fragments them —
    /// so this pre-scan is the only way a seeded abbreviation ever fires.
    ///
    /// Three properties that keep this cheap and correct:
    ///  1. Takes `&[char]` (the slice encode_normalized already built) — no
    ///     per-call re-collection, so no O(n^2) on the hot path.
    ///  2. Up-front guard: if no '.' or '।' occurs within seed_max_len chars of
    ///     `start`, returns immediately with zero string allocations. Plain words
    ///     (नेपालको, hello) never pay more than this O(seed_max_len) window scan.
    ///  3. Leading-marker skip: word-initial abbreviations sit behind a ▁ (the
    ///     space marker). We match the seed at start+1 and, on success, emit the
    ///     standalone ▁ token first so decode still reproduces the space. Without
    ///     this the scan would only fire at absolute position 0 and miss every
    ///     mid-sentence abbreviation.
    ///
    /// The inner match no longer re-checks contains('.') — the up-front guard
    /// already proved a period is in range, and get_id_by_surface is self-
    /// guarding: a window that isn't a seed surface simply isn't in the vocab.
    ///
    /// Note: this fires only when `start` lands on the abbreviation's first char
    /// (or the ▁ before it), i.e. at run starts. An abbreviation glued directly
    /// after Devanagari letters with no space is not caught — acceptable, since
    /// real abbreviations are word-initial.
    fn try_emit_atomic_seed(
        &self,
        chars: &[char],
        start: usize,
        tokens: &mut Vec<TokenId>,
    ) -> Option<usize> {
        let n = chars.len();
        if self.seed_max_len == 0 || start >= n {
            return None;
        }

        // Cheap window guard (no allocations).
        let window = self.seed_max_len.min(n - start);
        let has_punct = chars[start..start + window]
            .iter()
            .any(|&c| c == '.' || c == '\u{0964}'); // '.' or '।'
        if !has_punct {
            return None;
        }

        // Skip a leading space marker so word-initial abbreviations match the
        // bare seed surface; the SAME marker is re-emitted below on success so
        // decode still reproduces the space. Abbreviations are Devanagari (▁),
        // but accept either marker defensively.
        let leading_marker = if is_marker(chars[start]) { Some(chars[start]) } else { None };
        let offset = if leading_marker.is_some() { 1 } else { 0 };
        if start + offset >= n {
            return None;
        }
        let avail = n - (start + offset);
        let max_match = self.seed_max_len.min(avail);

        // Greedy longest-first, same principle as tokenize_greedy.
        for len in (2..=max_match).rev() {
            let candidate: String = chars[start + offset..start + offset + len].iter().collect();
            if let Some(seed_id) = self.vocab.get_id_by_surface(&candidate) {
                if let Some(m) = leading_marker {
                    if let Some(marker_id) = self.vocab.get_id_by_surface(&m.to_string()) {
                        tokens.push(marker_id);
                    }
                }
                tokens.push(seed_id);
                return Some(start + offset + len);
            }
        }
        None
    }

    /// Tokenize text that is ALREADY normalized (N applied). The streaming
    /// trainer normalizes a whole line once, then calls this per whitespace word
    /// — avoiding one NFC/fold pass per word (billions of them at multi-GB).
    ///
    /// Two transforms, in order:
    ///   Space marking: a dummy word-start marker is prepended and every ASCII
    ///     space is replaced by a marker. The marker is chosen by LOOKAHEAD at
    ///     the following character — ▂ before ASCII letters/digits (folds into
    ///     the Latin run), ▁ otherwise (folds into the Devanagari run). Because
    ///     the trainer calls this per whitespace word, each training word becomes
    ///     ▁word or ▂word — exactly what encode() produces for that word when a
    ///     space precedes it. That alignment lets ▁माया and ▂the merges both form
    ///     in training and apply at encode. decode() maps both markers back to a
    ///     space. IMPORTANT: Devanagari word-starts always resolve to ▁, so a
    ///     frozen Devanagari vocabulary keeps encoding identically.
    ///   Atomic seeds: at each position, try_emit_atomic_seed runs BEFORE the
    ///     per-script run splitter, so cross-script abbreviation seeds fire as
    ///     one token instead of fragmenting at every '.'.
    pub fn encode_normalized(&self, normalized: &str) -> Vec<TokenId> {
        // Build the marked working string with per-boundary lookahead markers.
        let src: Vec<char> = normalized.chars().collect();
        let mut marked = String::with_capacity(normalized.len() + 4);
        marked.push(marker_for(src.first().copied())); // dummy prefix by 1st char
        for (idx, &ch) in src.iter().enumerate() {
            if ch == ' ' {
                marked.push(marker_for(src.get(idx + 1).copied()));
            } else {
                marked.push(ch);
            }
        }

        let chars: Vec<char> = marked.chars().collect();
        let n = chars.len();
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < n {
            // STEP 1 — atomic abbreviation seed (reuses the char slice).
            if let Some(end) = self.try_emit_atomic_seed(&chars, i, &mut tokens) {
                i = end;
                continue;
            }

            // STEP 2 — per-script run: extend while the script matches, then
            // hand the whole run to tokenize_run (greedy match / byte fallback).
            //
            // CRITICAL: a run must not cross a following marker (▁ OR ▂). A
            // marker shares its script with the words it precedes (▁ is DEV like
            // Devanagari, ▂ is LAT like Latin), so without this stop two marked
            // same-script words (▁w₁▁w₂ or ▂w₁▂w₂) fuse into one run — the atomic
            // pre-scan never gets a clean boundary and abbreviations fragment.
            // The run still BEGINS with its own leading marker (i started on it),
            // so ▁word / ▂word within-word merges are unaffected; only cross-word
            // fusing is prevented.
            let start = i;
            let first_script = self.classify_char(chars[i]);
            i += 1;
            while i < n
                && self.classify_char(chars[i]) == first_script
                && !is_marker(chars[i])
            {
                i += 1;
            }
            let run: String = chars[start..i].iter().collect();
            self.tokenize_run(&run, first_script, &mut tokens);
        }

        tokens
    }

    fn classify_char(&self, ch: char) -> Script {
        if ch == DEV_MARKER {
            // Word-start marker before Devanagari/other: DEV, so it folds into
            // Devanagari runs and BPE learns ▁माया.
            Script::DEV
        } else if ch == LAT_MARKER {
            // Word-start marker before Latin/digits: LAT, so it folds into Latin
            // runs and BPE learns ▂the — removing the +1 standalone-marker cost.
            Script::LAT
        } else if ch == '\u{200C}' {
            Script::FMT
        } else if ch.is_ascii_punctuation()
            || ch == '\u{0964}'
            || ch == '\u{0965}'
            || is_unicode_punct(ch)
        {
            // FIX #2: route extended Unicode punctuation to PUN (seeded in
            // initialize) so it resolves to one token instead of 3 byte tokens.
            Script::PUN
        } else if ch.is_ascii_alphabetic() || ch.is_ascii_digit() {
            // FIX #3: ASCII digits are seeded as LAT and their merges are
            // learnable in the Latin pass, but without this they never reached
            // the greedy matcher (is_ascii_alphabetic() is false for digits).
            Script::LAT
        } else if ('\u{0900}'..='\u{097F}').contains(&ch) {
            Script::DEV
        } else {
            Script::MAL
        }
    }

    fn tokenize_run(&self, run: &str, script: Script, tokens: &mut Vec<TokenId>) {
        match script {
            // DEV/LAT use greedy longest-match over the vocab, so the learned
            // merges are actually applied at encode time. When the DFA is
            // populated you may instead want to segment DEV to aksharas first and
            // greedy-match at akshara granularity to keep the hard conjunct
            // guarantee; char-level greedy is what matches the trained vocab.
            Script::DEV | Script::LAT => self.tokenize_greedy(run, tokens),
            Script::PUN => {
                for ch in run.chars() {
                    if let Some(id) = self.vocab.get_id_by_surface(&ch.to_string()) {
                        tokens.push(id);
                    } else {
                        self.emit_byte_fallback(ch.to_string().as_bytes(), tokens);
                    }
                }
            }
            Script::FMT => {
                if let Some(id) = self.vocab.get_id_by_surface("\u{200C}") {
                    tokens.push(id);
                }
            }
            Script::MAL => {
                self.emit_byte_fallback(run.as_bytes(), tokens);
            }
        }
    }

    /// Greedy longest-match over vocab surfaces, capped at the longest known
    /// surface length; misses fall through to per-byte fallback. Used for both
    /// DEV and LAT runs.
    fn tokenize_greedy(&self, run: &str, tokens: &mut Vec<TokenId>) {
        let chars: Vec<char> = run.chars().collect();
        let n = chars.len();
        let cap = self.vocab.max_surface_len().max(1);
        let mut i = 0;
        while i < n {
            let hi = n.min(i + cap);
            let mut matched = false;
            for j in (i + 1..=hi).rev() {
                let candidate: String = chars[i..j].iter().collect();
                if let Some(id) = self.vocab.get_id_by_surface(&candidate) {
                    tokens.push(id);
                    i = j;
                    matched = true;
                    break;
                }
            }
            if !matched {
                // Single char not in vocab -> spell it out in byte fallback.
                let ch_str = chars[i].to_string();
                self.emit_byte_fallback(ch_str.as_bytes(), tokens);
                i += 1;
            }
        }
    }

    /// Emit one byte-fallback token per raw byte, using the GPT-2 alphabet.
    fn emit_byte_fallback(&self, bytes: &[u8], tokens: &mut Vec<TokenId>) {
        for &byte in bytes {
            let surface = self.byte_encoder[byte as usize].to_string();
            if let Some(id) = self.vocab.get_id_by_surface(&surface) {
                tokens.push(id);
            }
        }
    }

    /// Blocker 2 + decode-by-type: reconstruct bytes from ByteFallback tokens
    /// (detected by TYPE, not surface, so it is independent of the byte
    /// alphabet), and preserve ZWNJ verbatim (NO surface skip). Flushes buffered
    /// bytes with from_utf8_lossy so a model-generated arbitrary byte stream
    /// degrades gracefully instead of dropping a whole run.
    ///
    /// FIX #1: after reconstruction, every ▁ marker becomes a space and the
    /// single dummy-prefix space is dropped. This is the inverse of the marking
    /// done in encode_normalized and is what makes decode(encode(s)) ==
    /// normalize(s) hold, including for runs of multiple spaces.
    pub fn decode(&self, token_ids: &[TokenId]) -> String {
        let mut result = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();

        for &id in token_ids {
            match self.vocab.get_token(id).map(|arc| arc.as_ref()) {
                Some(Token::ByteFallback(byte)) => {
                    byte_buf.push(*byte);
                }
                Some(_) => {
                    if !byte_buf.is_empty() {
                        result.push_str(&String::from_utf8_lossy(&std::mem::take(&mut byte_buf)));
                    }
                    if let Some(surface) = self.vocab.get_surface(id) {
                        result.push_str(&surface); // ZWNJ included, like any other surface
                    }
                }
                None => {
                    if !byte_buf.is_empty() {
                        result.push_str(&String::from_utf8_lossy(&std::mem::take(&mut byte_buf)));
                    }
                }
            }
        }

        if !byte_buf.is_empty() {
            result.push_str(&String::from_utf8_lossy(&byte_buf));
        }

        // Reverse the space marking: BOTH markers become a space, then drop the
        // single dummy-prefix space that encode_normalized adds.
        let spaced = result.replace(DEV_MARKER, " ").replace(LAT_MARKER, " ");
        match spaced.strip_prefix(' ') {
            Some(rest) => rest.to_string(),
            None => spaced,
        }
    }

    pub fn verify_roundtrip(&self, s: &str) -> bool {
        let encoded = self.encode(s);
        let decoded = self.decode(&encoded);
        decoded == self.normalizer.normalize(s)
    }
}

// ============================================================================
// Python bindings
// ============================================================================

#[pyclass]
pub struct PyHimalayanTOK_Nepali_64K {
    inner: HimalayanTOK_Nepali_64K,
}

#[pymethods]
impl PyHimalayanTOK_Nepali_64K {
    /// Blocker 4: pass folding rules as a list of (pattern, replacement) string
    /// pairs, e.g. [("सङ्ग", "संग"), ("सँग", "संग")], instead of a char->char dict.
    #[new]
    #[pyo3(signature = (folding_rules=None))]
    fn new(folding_rules: Option<Vec<(String, String)>>) -> PyResult<Self> {
        let rules = folding_rules.unwrap_or_default();
        let registry = ParadigmRegistry::new();
        let inner = HimalayanTOK_Nepali_64K::new(rules, registry);
        Ok(Self { inner })
    }

    fn normalize(&self, s: &str) -> String {
        self.inner.normalizer.normalize(s)
    }

    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        Ok(self.inner.encode(text))
    }

    fn decode(&self, ids: Vec<usize>) -> String {
        self.inner.decode(&ids)
    }

    fn initialize_vocab(
        &mut self,
        aksharas: Vec<String>,
        seed_morphemes: Vec<String>,
        punctuation: Vec<String>,
        v_strict: Vec<String>,
        v_ambiguous: Vec<String>,
    ) -> PyResult<usize> {
        let base_aksharas: Vec<Akshara> = aksharas
            .into_iter()
            .map(|s| Akshara {
                surface: s,
                root_set: Vec::new(),
            })
            .collect();

        // Cap for the atomic abbreviation pre-scan: the longest seed surface
        // that contains punctuation ('.' or '।'). Computed BEFORE seed_morphemes
        // is moved into vocab.initialize. Only punctuation-bearing seeds use the
        // atomic path; pure-Devanagari or pure-digit seeds go through normal runs
        // and must NOT widen this window.
        let mut seed_max_len = 0usize;
        for surf in &seed_morphemes {
            if surf.contains('.') || surf.contains('\u{0964}') {
                seed_max_len = seed_max_len.max(surf.chars().count());
            }
        }
        self.inner.seed_max_len = seed_max_len;

        self.inner.vocab.initialize(
            base_aksharas,
            seed_morphemes,
            punctuation,
            v_strict.into_iter().collect(),
            v_ambiguous.into_iter().collect(),
            &self.inner.byte_encoder,
        );

        Ok(self.inner.vocab.len())
    }

    /// Explicitly tag base/seed tokens with paradigm roots. Also called
    /// automatically at the start of training; safe to call more than once.
    fn assign_initial_roots(&mut self) {
        self.inner
            .vocab
            .assign_roots_from_registry(&self.inner.paradigm_registry);
    }

    fn add_dfa_transition(&mut self, state: usize, ch: char, next_state: usize, accepting: bool) {
        self.inner
            .akshara_dfa
            .transitions
            .insert((state, ch), next_state);
        if accepting {
            self.inner.akshara_dfa.accepting_states.insert(next_state);
        }
    }

    /// Debug: Check if a DFA transition exists
    fn dfa_has_transition(&self, state: usize, ch: char) -> bool {
        self.inner.akshara_dfa.has_transition(state, ch).is_some()
    }

    /// Debug: Get transitions from a state
    fn dfa_get_transitions(&self, state: usize) -> Vec<(String, usize, bool)> {
        self.inner
            .akshara_dfa
            .get_transitions_from(state)
            .into_iter()
            .map(|(ch, next, acc)| (ch.to_string(), next, acc))
            .collect()
    }

    /// Debug: Test DFA tokenization
    fn dfa_tokenize_debug(&self, text: &str) -> Vec<String> {
        self.inner
            .akshara_dfa
            .tokenize(text)
            .into_iter()
            .map(|a| a.surface)
            .collect()
    }

    /// Debug: Check if surface exists in vocab
    fn vocab_contains(&self, surface: &str) -> bool {
        self.inner.vocab.contains_surface(surface)
    }

    /// Debug: Get token ID for surface
    fn vocab_get_id(&self, surface: &str) -> Option<usize> {
        self.inner.vocab.get_id_by_surface(surface)
    }

    /// Debug: Get surface for token ID
    fn vocab_get_surface(&self, id: usize) -> PyResult<String> {
        self.inner
            .vocab
            .get_surface(id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid token ID"))
    }

    fn add_paradigm(&mut self, root_id: usize, transitions: Bound<'_, PyDict>) -> PyResult<()> {
        let mut paradigm = Paradigm::new(root_id);
        for (prefix, continuations) in transitions.iter() {
            let prefix_str: String = prefix.extract()?;
            let cont_list: Vec<String> = continuations.extract()?;
            paradigm
                .allowed_transitions
                .insert(prefix_str, cont_list.into_iter().collect());
        }
        self.inner.paradigm_registry.add_paradigm(paradigm);
        Ok(())
    }

    fn train_bpe(
        &mut self,
        sequences: Vec<Vec<usize>>,
        vocab_budget: usize,
        theta: u64,
    ) -> PyResult<usize> {
        // Wire paradigm roots into the vocab before training. No-op behaviorally
        // if no paradigms have been registered.
        self.inner
            .vocab
            .assign_roots_from_registry(&self.inner.paradigm_registry);

        let mut corpus = Corpus::from_sequences(sequences, vocab_budget);
        let mut trainer = ConstrainedBPETrainer::new(
            std::mem::take(&mut self.inner.vocab),
            std::mem::take(&mut self.inner.paradigm_registry),
        );
        trainer.theta = theta;
        trainer.train(&mut corpus, vocab_budget, 0);

        self.inner.vocab = trainer.vocab;
        self.inner.paradigm_registry = trainer.paradigm_registry;

        Ok(self.inner.vocab.len())
    }

    /// Streaming, word-deduplicated training over a text file — the path to use
    /// for large corpora (multi-GB). Reads the file line by line, so peak RAM is
    /// the word-frequency dictionary, NOT the tokenized corpus.
    ///
    /// - `min_word_freq`: drop word types occurring fewer than this many times
    ///   (cuts the huge hapax tail of morphologically rich Nepali; 1 = keep all).
    /// - `progress_lines`: print a build-progress line every N input lines (0=off).
    /// - `progress_merges`: print a train-progress line every N merges (0=off).
    ///
    /// Timing for the build phase and the train phase is printed to stderr. The
    /// return value is the final vocab size.
    #[pyo3(signature = (path, vocab_budget, theta, min_word_freq=1, progress_lines=500000, progress_merges=1000))]
    fn train_from_file(
        &mut self,
        py: Python<'_>,
        path: String,
        vocab_budget: usize,
        theta: u64,
        min_word_freq: u64,
        progress_lines: u64,
        progress_merges: u64,
    ) -> PyResult<usize> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        use std::time::Instant;

        let file = File::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
        })?;

        // Release the GIL for the whole heavy phase; we only touch pure-Rust state.
        let (counts, lines_done, word_occ, build_secs) = py.allow_threads(|| {
            let t0 = Instant::now();
            let reader = BufReader::with_capacity(1 << 20, file);
            let mut counts: HashMap<Vec<TokenId>, Frequency> = HashMap::new();
            let mut lines_done: u64 = 0;
            let mut word_occ: u64 = 0;

            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue, // skip unreadable line rather than abort a long job
                };
                let norm = self.inner.normalizer.normalize(&line);
                for word in norm.split_whitespace() {
                    let toks = self.inner.encode_normalized(word);
                    if !toks.is_empty() {
                        *counts.entry(toks).or_insert(0) += 1;
                        word_occ += 1;
                    }
                }
                lines_done += 1;
                if progress_lines > 0 && lines_done % progress_lines == 0 {
                    eprintln!(
                        "[build] {} lines | {} word-occ | {} unique | {:.1}s",
                        lines_done,
                        word_occ,
                        counts.len(),
                        t0.elapsed().as_secs_f64()
                    );
                }
            }

            if min_word_freq > 1 {
                counts.retain(|_, &mut c| c >= min_word_freq);
            }
            (counts, lines_done, word_occ, t0.elapsed().as_secs_f64())
        });

        eprintln!(
            "[build] done: {} lines, {} word-occ, {} unique types kept (min_freq={}) in {:.1}s",
            lines_done,
            word_occ,
            counts.len(),
            min_word_freq,
            build_secs
        );

        // Tag roots (no-op unless paradigms loaded), then train (GIL released).
        self.inner
            .vocab
            .assign_roots_from_registry(&self.inner.paradigm_registry);

        let final_vocab = py.allow_threads(|| {
            let mut corpus = Corpus::from_word_counts(counts, vocab_budget);
            let mut trainer = ConstrainedBPETrainer::new(
                std::mem::take(&mut self.inner.vocab),
                std::mem::take(&mut self.inner.paradigm_registry),
            );
            trainer.theta = theta;
            trainer.train(&mut corpus, vocab_budget, progress_merges);

            self.inner.vocab = trainer.vocab;
            self.inner.paradigm_registry = trainer.paradigm_registry;
            self.inner.vocab.len()
        });

        Ok(final_vocab)
    }

    fn train_from_text(
        &mut self,
        texts: Vec<String>,
        vocab_budget: usize,
        theta: u64,
    ) -> PyResult<usize> {
        let sequences: Vec<Vec<usize>> = texts
            .iter()
            .map(|t| self.inner.encode(t))
            .filter(|seq| !seq.is_empty())
            .collect();

        if sequences.is_empty() {
            return Ok(self.inner.vocab.len());
        }

        self.train_bpe(sequences, vocab_budget, theta)
    }

    /// PHASE 4 — bilingual training. Builds the word-frequency dictionary ONCE
    /// from a mixed Nepali+English file, then runs two passes over it:
    ///
    ///   1. Phase 3 (constrained): Devanagari + punctuation, up to `dev_budget`.
    ///   2. Phase 4 (unconstrained): Latin/digits only, up to
    ///      `dev_budget + lat_budget` total.
    ///
    /// The budget split is an explicit CHOICE, not a default. Nepali is the
    /// priority language and needs the larger slice; English reaches acceptable
    /// fertility with far fewer slots (its high-frequency subword core is small).
    /// A reasonable start is dev=40000, lat=8000.
    ///
    /// Script separation is preserved: no DEV/LAT token can ever form, because
    /// each pass's ScriptCompat admits only its own script. The ▁ marker is DEV,
    /// so ▁-prefixed merges are learned only in pass 1 (Devanagari word-starts).
    #[pyo3(signature = (path, dev_budget, lat_budget, theta, min_word_freq=1, progress_lines=500000, progress_merges=1000))]
    fn train_bilingual_from_file(
        &mut self,
        py: Python<'_>,
        path: String,
        dev_budget: usize,
        lat_budget: usize,
        theta: u64,
        min_word_freq: u64,
        progress_lines: u64,
        progress_merges: u64,
    ) -> PyResult<usize> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        use std::time::Instant;

        let file = File::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
        })?;

        let (counts, lines_done, word_occ, build_secs) = py.allow_threads(|| {
            let t0 = Instant::now();
            let reader = BufReader::with_capacity(1 << 20, file);
            let mut counts: HashMap<Vec<TokenId>, Frequency> = HashMap::new();
            let mut lines_done: u64 = 0;
            let mut word_occ: u64 = 0;

            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let norm = self.inner.normalizer.normalize(&line);
                for word in norm.split_whitespace() {
                    let toks = self.inner.encode_normalized(word);
                    if !toks.is_empty() {
                        *counts.entry(toks).or_insert(0) += 1;
                        word_occ += 1;
                    }
                }
                lines_done += 1;
                if progress_lines > 0 && lines_done % progress_lines == 0 {
                    eprintln!(
                        "[build] {} lines | {} word-occ | {} unique | {:.1}s",
                        lines_done,
                        word_occ,
                        counts.len(),
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
            if min_word_freq > 1 {
                counts.retain(|_, &mut c| c >= min_word_freq);
            }
            (counts, lines_done, word_occ, t0.elapsed().as_secs_f64())
        });

        eprintln!(
            "[build] done: {} lines, {} word-occ, {} unique types kept (min_freq={}) in {:.1}s",
            lines_done, word_occ, counts.len(), min_word_freq, build_secs
        );

        self.inner
            .vocab
            .assign_roots_from_registry(&self.inner.paradigm_registry);

        let total_budget = dev_budget + lat_budget;

        let final_vocab = py.allow_threads(|| {
            let mut corpus = Corpus::from_word_counts(counts, total_budget);
            let mut trainer = ConstrainedBPETrainer::new(
                std::mem::take(&mut self.inner.vocab),
                std::mem::take(&mut self.inner.paradigm_registry),
            );
            trainer.theta = theta;

            // Pass 1 — Phase 3, constrained, Devanagari + punctuation.
            trainer.latin_pass = false;
            trainer.train(&mut corpus, dev_budget, progress_merges);
            let after_dev = trainer.vocab.len();

            // Pass 2 — Phase 4, unconstrained, Latin only. The heap is rebuilt
            // from scratch inside train(), and initialize_heap now admits LAT
            // pairs because script_compat flipped.
            trainer.latin_pass = true;
            trainer.train(&mut corpus, total_budget, progress_merges);
            let after_lat = trainer.vocab.len();

            eprintln!(
                "[train] budget split: DEV {} (target {}) | LAT +{} (target +{})",
                after_dev,
                dev_budget,
                after_lat - after_dev,
                lat_budget
            );

            self.inner.vocab = trainer.vocab;
            self.inner.paradigm_registry = trainer.paradigm_registry;
            self.inner.vocab.len()
        });

        Ok(final_vocab)
    }

    /// LATIN-ONLY retrain onto an ALREADY-LOADED vocab (call load_vocab_tsv
    /// first). Runs a single unconstrained Latin pass over the corpus, adding up
    /// to `lat_budget` new LAT tokens on top of the current vocab size. The
    /// Devanagari vocabulary is untouched: script_compat admits only LAT–LAT in
    /// this pass, so no DEV token can enter a merge and no DEV token is dropped.
    /// This is how you improve English WITHOUT retraining Devanagari.
    ///
    /// Memory: the word-count dictionary is filtered to words containing ASCII
    /// alphanumerics, so pure-Devanagari words never enter it. The dictionary is
    /// therefore English/digit-sized (small) and will not reproduce the
    /// Devanagari hapax memory blow-up — even though the whole corpus is streamed.
    ///
    /// Streaming cost is still the full file: if the corpus is blended, consider
    /// extracting English lines to their own file once and pointing this at that.
    #[pyo3(signature = (path, lat_budget, min_word_freq=1, progress_lines=500000, progress_merges=1000))]
    fn train_latin_from_file(
        &mut self,
        py: Python<'_>,
        path: String,
        lat_budget: usize,
        min_word_freq: u64,
        progress_lines: u64,
        progress_merges: u64,
    ) -> PyResult<usize> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        use std::time::Instant;

        if self.inner.vocab.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "train_latin_from_file: vocab is empty — call load_vocab_tsv first",
            ));
        }

        let file = File::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
        })?;

        let (counts, lines_done, word_occ, build_secs) = py.allow_threads(|| {
            let t0 = Instant::now();
            let reader = BufReader::with_capacity(1 << 20, file);
            let mut counts: HashMap<Vec<TokenId>, Frequency> = HashMap::new();
            let mut lines_done: u64 = 0;
            let mut word_occ: u64 = 0;

            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let norm = self.inner.normalizer.normalize(&line);
                for word in norm.split_whitespace() {
                    // Latin filter: skip words with no ASCII alphanumeric. This
                    // drops all pure-Devanagari words BEFORE encoding, keeping the
                    // dictionary English/digit-sized (the memory fix).
                    if !word.chars().any(|c| c.is_ascii_alphanumeric()) {
                        continue;
                    }
                    let toks = self.inner.encode_normalized(word);
                    if !toks.is_empty() {
                        *counts.entry(toks).or_insert(0) += 1;
                        word_occ += 1;
                    }
                }
                lines_done += 1;
                if progress_lines > 0 && lines_done % progress_lines == 0 {
                    eprintln!(
                        "[build:LAT] {} lines | {} latin-word-occ | {} unique | {:.1}s",
                        lines_done, word_occ, counts.len(), t0.elapsed().as_secs_f64()
                    );
                }
            }
            if min_word_freq > 1 {
                counts.retain(|_, &mut c| c >= min_word_freq);
            }
            (counts, lines_done, word_occ, t0.elapsed().as_secs_f64())
        });

        eprintln!(
            "[build:LAT] done: {} lines, {} latin-word-occ, {} unique types kept \
             (min_freq={}) in {:.1}s",
            lines_done, word_occ, counts.len(), min_word_freq, build_secs
        );

        let target = self.inner.vocab.len() + lat_budget;

        let final_vocab = py.allow_threads(|| {
            let mut corpus = Corpus::from_word_counts(counts, target);
            let mut trainer = ConstrainedBPETrainer::new(
                std::mem::take(&mut self.inner.vocab),
                std::mem::take(&mut self.inner.paradigm_registry),
            );
            trainer.theta = 100; // unused in latin pass, set for completeness
            trainer.latin_pass = true; // LAT–LAT only; DEV vocab frozen
            trainer.train(&mut corpus, target, progress_merges);

            self.inner.vocab = trainer.vocab;
            self.inner.paradigm_registry = trainer.paradigm_registry;
            self.inner.vocab.len()
        });

        Ok(final_vocab)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab.len()
    }

    /// Rebuild the vocab from (id, surface) pairs — encode/decode-ready, not
    /// training-ready. Ids must be contiguous 0..N.
    fn load_vocab(&mut self, pairs: Vec<(usize, String)>) -> PyResult<usize> {
        self.inner.load_vocab(pairs);
        Ok(self.inner.vocab.len())
    }

    

    /// Load the vocab from a TSV written by the training driver ("id\tsurface"),
    /// reversing the tab/newline/backslash escaping. Encode/decode-ready.
    fn load_vocab_tsv(&mut self, path: String) -> PyResult<usize> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
        })?;
        let reader = BufReader::new(file);

        let mut pairs: Vec<(usize, String)> = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("read: {}", e))
            })?;
            if line.is_empty() {
                continue;
            }
            let mut it = line.splitn(2, '\t');
            let id_str = match it.next() {
                Some(s) => s,
                None => continue,
            };
            let surf_raw = it.next().unwrap_or("");
            let id: usize = match id_str.parse() {
                Ok(v) => v,
                Err(_) => continue, // skip a malformed line rather than abort
            };
            let surface = unescape_tsv(surf_raw);
            pairs.push((id, surface));
        }

        self.inner.load_vocab(pairs);
        Ok(self.inner.vocab.len())
    }

    fn get_token_surface(&self, id: usize) -> PyResult<String> {
        self.inner
            .vocab
            .get_surface(id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid token ID"))
    }
}

#[pymodule(name = "HimalayanTOK_Nepali_64K")]
fn himalayan_tok(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHimalayanTOK_Nepali_64K>()?;
    Ok(())
}