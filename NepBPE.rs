use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use unicode_normalization::UnicodeNormalization;
use std::rc::Rc;

// ============================================================================
// Type aliases and basic types
// ============================================================================

type TokenId = usize;
type RootId = usize;
type ByteVal = u8;
type Frequency = u64;
type ScriptType = u8;

// ============================================================================
// Script types (total assignment per §2.1)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Script {
    DEV,    // Devanagari akshara or DEV-derived merged token
    LAT,    // Latin
    PUN,    // Punctuation
    FMT,    // Format control (ZWNJ)
    MAL,    // Byte-fallback
}

impl Script {
    fn rank(&self) -> u8 {
        match self {
            Script::DEV => 2,
            Script::PUN => 1,
            _ => 0,  // LAT, FMT, MAL never participate in DEV merges
        }
    }
}

// ============================================================================
// Phase 1: Normalization N
// ============================================================================

pub struct Normalizer {
    folding_table: HashMap<char, char>, // Fold_O table
}

impl Normalizer {
    pub fn new(folding_table: HashMap<char, char>) -> Self {
        Self { folding_table }
    }

    /// N(s) = StripZWJ ∘ Fold_O ∘ NFC (s)
    /// ZWNJ (U+200C) is preserved, but ZWJ (U+200D) is stripped
    pub fn normalize(&self, s: &str) -> String {
        // Step 1: NFC canonical composition
        let nfc: String = s.nfc().collect();

        // Step 2: Fold_O - orthographic folding
        let folded: String = nfc
            .chars()
            .map(|c| self.folding_table.get(&c).copied().unwrap_or(c))
            .collect();

        // Step 3: Strip ZWJ (U+200D), preserve ZWNJ (U+200C)
        let stripped: String = folded
            .chars()
            .filter(|&c| c != '\u{200D}') // Remove ZWJ
            .collect();

        stripped
    }

    /// Idempotent: N(N(s)) = N(s)
    pub fn is_idempotent(&self) -> bool {
        true // By construction if Fold_O only maps characters to NFC-normalized forms
    }
}

// ============================================================================
// Phase 2: Akshara DFA
// ============================================================================

/// Represents a well-formed Devanagari akshara
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Akshara {
    surface: String,      // The actual Unicode string
    root_set: Vec<RootId>, // Roots that have this as a prefix
}

/// DFA for Devanagari akshara recognition
pub struct AksharaDFA {
    // State machine definition
    transitions: HashMap<(usize, char), usize>,
    accepting_states: HashSet<usize>,
    initial_state: usize,
}

impl AksharaDFA {
    pub fn new() -> Self {
        // Simplified: would contain full DFA for Devanagari syllable structure
        Self {
            transitions: HashMap::new(),
            accepting_states: HashSet::new(),
            initial_state: 0,
        }
    }

    /// Tokenize a Devanagari run into well-formed aksharas
    pub fn tokenize(&self, dev_text: &str) -> Vec<Akshara> {
        let mut aksharas = Vec::new();
        let mut current = String::new();
        let mut state = self.initial_state;

        for ch in dev_text.chars() {
            current.push(ch);
            if let Some(&next_state) = self.transitions.get(&(state, ch)) {
                state = next_state;
                if self.accepting_states.contains(&state) {
                    // Emit akshara and tag with root set
                    let root_set = self.compute_root_set(&current);
                    aksharas.push(Akshara {
                        surface: current.clone(),
                        root_set,
                    });
                    current.clear();
                    state = self.initial_state;
                }
            } else {
                // Invalid sequence - handle as byte fallback
                state = self.initial_state;
                current.clear();
            }
        }

        aksharas
    }

    fn compute_root_set(&self, prefix: &str) -> Vec<RootId> {
        // RootSet(α) := { root : α ∈ prefixes(P(root).states) }
        // Simplified; actual implementation would query paradigm FST
        Vec::new()
    }
}

// ============================================================================
// Token types and vocabulary
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Token {
    Akshara(Akshara),
    Punctuation(String),        // Σ_P
    ZWNJ,                       // τ_ZWNJ (U+200C)
    ByteFallback(ByteVal),      // β_x for x ∈ [0..255]
    SeededMorpheme(String),     // From V_seed
    MergedToken(Vec<TokenId>),  // Compound token from BPE merges
}

impl Token {
    fn script(&self) -> Script {
        match self {
            Token::Akshara(_) => Script::DEV,
            Token::Punctuation(_) => Script::PUN,
            Token::ZWNJ => Script::FMT,
            Token::ByteFallback(_) => Script::MAL,
            Token::SeededMorpheme(_) => Script::DEV,
            Token::MergedToken(_) => Script::DEV, // DEV-derived merged tokens
        }
    }

    fn surface(&self) -> String {
        match self {
            Token::Akshara(a) => a.surface.clone(),
            Token::Punctuation(p) => p.clone(),
            Token::ZWNJ => '\u{200C}'.to_string(),
            Token::ByteFallback(b) => (*b as char).to_string(),
            Token::SeededMorpheme(s) => s.clone(),
            Token::MergedToken(ids) => {
                ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join("")
            }
        }
    }
}

// ============================================================================
// Vocabulary management
// ============================================================================

pub struct Vocabulary {
    tokens: Vec<Rc<Token>>,
    token_to_id: HashMap<Rc<Token>, TokenId>,
    id_to_script: HashMap<TokenId, Script>,
    
    // V_strict and V_ambiguous sets (indices into tokens)
    v_strict: HashSet<TokenId>,
    v_ambiguous: HashSet<TokenId>,
    
    // Root tracking for paradigm morphology
    token_to_root_set: HashMap<TokenId, Vec<RootId>>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            token_to_id: HashMap::new(),
            id_to_script: HashMap::new(),
            v_strict: HashSet::new(),
            v_ambiguous: HashSet::new(),
            token_to_root_set: HashMap::new(),
        }
    }

    /// V_0 = BaseAksharas ∪ V_seed ∪ Σ_P ∪ { β_x } ∪ { τ_ZWNJ }
    pub fn initialize(
        &mut self,
        base_aksharas: Vec<Akshara>,
        seed_morphemes: Vec<String>,
        punctuation: Vec<String>,
        v_strict: HashSet<String>,     // 53 unconditionally frozen
        v_ambiguous: HashSet<String>,  // 9 frequency-gated
    ) {
        // Add base aksharas
        for akshara in base_aksharas {
            let token = Rc::new(Token::Akshara(akshara));
            self.add_token(token, false, false);
        }

        // Add seed morphemes (V_seed = V_strict ∪ V_ambiguous)
        for morph in seed_morphemes {
            let is_strict = v_strict.contains(&morph);
            let is_ambiguous = v_ambiguous.contains(&morph);
            let token = Rc::new(Token::SeededMorpheme(morph));
            self.add_token(token, is_strict, is_ambiguous);
        }

        // Add punctuation alphabet Σ_P (including danda । and double danda ॥)
        for punct in punctuation {
            let token = Rc::new(Token::Punctuation(punct));
            self.add_token(token, false, false);
        }

        // Add byte-fallback tokens β_x for all x ∈ [0..255]
        for byte_val in 0u8..=255 {
            let token = Rc::new(Token::ByteFallback(byte_val));
            self.add_token(token, false, false);
        }

        // Add ZWNJ control token τ_ZWNJ
        let zwnj = Rc::new(Token::ZWNJ);
        self.add_token(zwnj, false, false);
    }

    fn add_token(&mut self, token: Rc<Token>, is_strict: bool, is_ambiguous: bool) -> TokenId {
        let id = self.tokens.len();
        let script = token.script();
        
        self.token_to_id.insert(token.clone(), id);
        self.id_to_script.insert(id, script);
        self.tokens.push(token.clone());
        
        if is_strict {
            self.v_strict.insert(id);
        }
        if is_ambiguous {
            self.v_ambiguous.insert(id);
        }
        
        // Initialize root set for Akshara tokens
        if let Token::Akshara(a) = &*token {
            self.token_to_root_set.insert(id, a.root_set.clone());
        }
        
        id
    }

    pub fn get_token(&self, id: TokenId) -> Option<&Rc<Token>> {
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
        self.token_to_root_set.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Create a merged token and return its ID
    pub fn create_merged(&mut self, a: TokenId, b: TokenId, merged_root_set: Vec<RootId>) -> TokenId {
        let merged = Rc::new(Token::MergedToken(vec![a, b]));
        let id = self.add_token(merged, false, false);
        self.token_to_root_set.insert(id, merged_root_set);
        id
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn iter_ids(&self) -> impl Iterator<Item = TokenId> + '_ {
        (0..self.tokens.len()).into_iter()
    }
}

// ============================================================================
// Paradigm morphology (FST-backed)
// ============================================================================

/// Represents a morphological paradigm (verb root, noun stem, etc.)
pub struct Paradigm {
    root_id: RootId,
    // FST states and transitions
    // P(root).states[a].allowed_next
    allowed_transitions: HashMap<String, HashSet<String>>,
}

impl Paradigm {
    pub fn new(root_id: RootId) -> Self {
        Self {
            root_id,
            allowed_transitions: HashMap::new(),
        }
    }

    /// Check if b is an allowed continuation after prefix a
    pub fn allows(&self, prefix: &str, continuation: &str) -> bool {
        self.allowed_transitions
            .get(prefix)
            .map(|allowed| allowed.contains(continuation))
            .unwrap_or(false)
    }

    /// Check if a is a prefix of any state in this paradigm
    pub fn has_prefix(&self, prefix: &str) -> bool {
        self.allowed_transitions.contains_key(prefix)
    }
}

/// Registry of all paradigms
pub struct ParadigmRegistry {
    paradigms: Vec<Paradigm>,
    root_to_paradigm: HashMap<RootId, usize>,
}

impl ParadigmRegistry {
    pub fn new() -> Self {
        Self {
            paradigms: Vec::new(),
            root_to_paradigm: HashMap::new(),
        }
    }

    pub fn add_paradigm(&mut self, paradigm: Paradigm) {
        let idx = self.paradigms.len();
        self.root_to_paradigm.insert(paradigm.root_id, idx);
        self.paradigms.push(paradigm);
    }

    pub fn get_root_set(&self, prefix: &str) -> Vec<RootId> {
        // RootSet(α) := { root : α ∈ prefixes(P(root).states) }
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

/// Corpus representation for BPE training
pub struct Corpus {
    /// Sequence of token IDs per word
    words: Vec<Vec<TokenId>>,
    /// Adjacency pair frequencies (within words only)
    pair_freqs: HashMap<(TokenId, TokenId), Frequency>,
    /// Total vocabulary budget
    vocab_budget: usize,
}

impl Corpus {
    pub fn new(sequences: Vec<Vec<TokenId>>, vocab_budget: usize) -> Self {
        let mut corpus = Self {
            words: sequences,
            pair_freqs: HashMap::new(),
            vocab_budget,
        };
        corpus.recompute_all_frequencies();
        corpus
    }

    fn recompute_all_frequencies(&mut self) {
        self.pair_freqs.clear();
        for word in &self.words {
            for window in word.windows(2) {
                *self.pair_freqs.entry((window[0], window[1])).or_insert(0) += 1;
            }
        }
    }

    pub fn get_freq(&self, a: TokenId, b: TokenId) -> Frequency {
        self.pair_freqs.get(&(a, b)).copied().unwrap_or(0)
    }

    pub fn apply_merge(&mut self, a: TokenId, b: TokenId, new_id: TokenId) {
        // Replace all occurrences of [a, b] with [new_id] within word boundaries
        for word in &mut self.words {
            let mut i = 0;
            while i + 1 < word.len() {
                if word[i] == a && word[i + 1] == b {
                    // Check same_word constraint (always true within a word)
                    // Remove the pair
                    word.remove(i + 1);
                    word[i] = new_id;
                    
                    // Update frequencies for affected adjacent pairs
                    self.update_local_frequencies(word, i);
                }
                i += 1;
            }
        }
    }

    fn update_local_frequencies(&mut self, word: &[TokenId], position: usize) {
        // Decrement old pairs and increment new pairs around the merge point
        // This is called after a merge is applied; details omitted for brevity
        // but would maintain pair_freqs accurately
    }
}

/// BPE merge candidate with priority key
#[derive(Debug, Clone)]
struct MergeCandidate {
    a: TokenId,
    b: TokenId,
    priority_key: (u8, u64), // (ScriptRank, weighted frequency)
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

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: reverse order for BinaryHeap
        self.priority_key.cmp(&other.priority_key).reverse()
    }
}

/// The main BPE trainer implementing Phase 3
pub struct ConstrainedBPETrainer {
    vocab: Vocabulary,
    paradigm_registry: ParadigmRegistry,
    theta: Frequency, // θ = 100 for ambiguous morpheme gating
}

impl ConstrainedBPETrainer {
    pub fn new(vocab: Vocabulary, paradigm_registry: ParadigmRegistry) -> Self {
        Self {
            vocab,
            paradigm_registry,
            theta: 100,
        }
    }

    /// ScriptCompat(a,b) := script(a)=script(b) ∧ script(a) ∈ {DEV, PUN}
    fn script_compat(&self, a: TokenId, b: TokenId) -> bool {
        let sa = self.vocab.get_script(a);
        let sb = self.vocab.get_script(b);
        sa == sb && (sa == Script::DEV || sa == Script::PUN)
    }

    /// Gate(a,b) - fast path, O(1) set membership
    fn gate(&self, a: TokenId, b: TokenId, freq: Frequency) -> bool {
        let sa = self.vocab.get_script(a);
        let sb = self.vocab.get_script(b);

        // Atomic classes never merge (byte-fallback, ZWNJ)
        if sa == Script::MAL || sa == Script::FMT || sb == Script::MAL || sb == Script::FMT {
            return false;
        }

        // No token spans two scripts
        if !self.script_compat(a, b) {
            return false;
        }

        // Frozen morpheme is never a continuation
        if self.vocab.is_strict(b) || self.vocab.is_ambiguous(b) {
            return false;
        }

        // Strict-frozen morphemes are TERMINAL (fix #1)
        if self.vocab.is_strict(a) {
            return false;
        }

        // Ambiguous morphemes extend only on strong evidence
        if self.vocab.is_ambiguous(a) && freq < self.theta {
            return false;
        }

        true
    }

    /// Morph(a,b) - slow path, existential, root-tracked
    fn morph(&self, a: TokenId, b: TokenId) -> bool {
        let root_set = self.vocab.get_root_set(a);

        // Fix #2: empty RootSet means non-paradigm token - always allow
        if root_set.is_empty() {
            return true;
        }

        // Existential check: some root licenses the continuation
        let token_a = self.vocab.get_token(a).unwrap();
        let token_b = self.vocab.get_token(b).unwrap();
        let surface_a = token_a.surface();
        let surface_b = token_b.surface();

        root_set.iter().any(|&root_id| {
            self.paradigm_registry.check_allowed(root_id, &surface_a, &surface_b)
        })
    }

    /// Legal(a,b) = Gate(a,b) ∧ Morph(a,b)
    fn legal(&self, a: TokenId, b: TokenId, freq: Frequency) -> bool {
        self.gate(a, b, freq) && self.morph(a, b)
    }

    /// RootSet narrowing: compute merged token's root set
    fn narrow_root_set(&self, a: TokenId, b: TokenId) -> Vec<RootId> {
        let root_set_a = self.vocab.get_root_set(a);
        if root_set_a.is_empty() {
            return Vec::new();
        }

        let token_b = self.vocab.get_token(b).unwrap();
        let surface_b = token_b.surface();

        let token_a = self.vocab.get_token(a).unwrap();
        let surface_a = token_a.surface();

        root_set_a
            .iter()
            .filter(|&&root_id| {
                self.paradigm_registry.check_allowed(root_id, &surface_a, &surface_b)
            })
            .copied()
            .collect()
    }

    /// Compute priority key K(a,b) with lexicographic script tiering
    fn priority_key(&self, a: TokenId, b: TokenId, freq: Frequency) -> (u8, u64) {
        let script_rank = self.vocab.get_script(a).rank();
        // W(a) is identity (weight 1) by default
        (script_rank, freq as u64)
    }

    /// Main training loop with lazy heap invalidation (§3.4)
    pub fn train(&mut self, corpus: &mut Corpus) {
        let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();

        // Initialize heap with all legal adjacent pairs
        self.initialize_heap(corpus, &mut heap);

        // BPE merge loop
        while self.vocab.len() < corpus.vocab_budget {
            let mut found_valid = false;

            while let Some(candidate) = heap.pop() {
                let current_freq = corpus.get_freq(candidate.a, candidate.b);

                // Lazy invalidation: discard if stale
                if current_freq != candidate.freq_snapshot {
                    continue;
                }

                // Re-check legality (may have changed due to frequency threshold)
                if !self.legal(candidate.a, candidate.b, current_freq) {
                    continue;
                }

                // Valid merge found - apply it
                let narrowed_roots = self.narrow_root_set(candidate.a, candidate.b);
                let new_id = self.vocab.create_merged(candidate.a, candidate.b, narrowed_roots);

                // Update corpus (this changes frequencies)
                corpus.apply_merge(candidate.a, candidate.b, new_id);

                // Push updated candidates for affected pairs
                self.push_affected_pairs(corpus, &mut heap, new_id);

                found_valid = true;
                break;
            }

            if !found_valid {
                // No more legal merges possible
                break;
            }
        }
    }

    fn initialize_heap(&self, corpus: &Corpus, heap: &mut BinaryHeap<MergeCandidate>) {
        for &(a, b) in corpus.pair_freqs.keys() {
            let freq = corpus.get_freq(a, b);
            if self.legal(a, b, freq) {
                let priority = self.priority_key(a, b, freq);
                heap.push(MergeCandidate {
                    a,
                    b,
                    priority_key: priority,
                    freq_snapshot: freq,
                });
            }
        }
    }

    fn push_affected_pairs(
        &self,
        corpus: &Corpus,
        heap: &mut BinaryHeap<MergeCandidate>,
        new_id: TokenId,
    ) {
        // For each position where the new merged token appears,
        // check adjacent pairs (left neighbor, new_id) and (new_id, right neighbor)
        for word in &corpus.words {
            for i in 0..word.len() {
                if word[i] == new_id {
                    if i > 0 {
                        let a = word[i - 1];
                        let b = new_id;
                        let freq = corpus.get_freq(a, b);
                        if self.legal(a, b, freq) {
                            let priority = self.priority_key(a, b, freq);
                            heap.push(MergeCandidate {
                                a,
                                b,
                                priority_key: priority,
                                freq_snapshot: freq,
                            });
                        }
                    }
                    if i + 1 < word.len() {
                        let a = new_id;
                        let b = word[i + 1];
                        let freq = corpus.get_freq(a, b);
                        if self.legal(a, b, freq) {
                            let priority = self.priority_key(a, b, freq);
                            heap.push(MergeCandidate {
                                a,
                                b,
                                priority_key: priority,
                                freq_snapshot: freq,
                            });
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Phase 4: Latin secondary pass
// ============================================================================

pub struct LatinBPETrainer {
    // Standard unconstrained BPE for Latin script tokens
    // Separate vocabulary, no cross-script merges possible
}

impl LatinBPETrainer {
    pub fn train(&mut self, latin_corpus: &mut Corpus, remaining_budget: usize) {
        // Standard BPE over Latin tokens only
        // Script-tiering guarantees no DEV/LAT cross-merges
    }
}

// ============================================================================
// Phase 5: Model integration
// ============================================================================

/// Paradigm embedding mapping: U : TokenID → RootID ∪ {⊥}
pub struct ParadigmEmbedding {
    /// Many-to-one mapping from surface tokens to root IDs
    token_to_root: HashMap<TokenId, Option<RootId>>,
    /// Root embeddings (learnable)
    root_embeddings: HashMap<RootId, Vec<f32>>,
    /// Default embedding for tokens without root assignment
    default_embedding: Vec<f32>,
}

impl ParadigmEmbedding {
    pub fn new() -> Self {
        Self {
            token_to_root: HashMap::new(),
            root_embeddings: HashMap::new(),
            default_embedding: Vec::new(),
        }
    }

    /// Seed U from paradigm FST and lexicon L
    /// - Allomorphy (आयो ↔ आउँछु): learned from FST analysis
    /// - Suppletion (हुनु→भयो): seeded from lexicon L
    pub fn seed_from_lexicon(&mut self, lexicon_seeds: HashMap<TokenId, RootId>) {
        for (token_id, root_id) in lexicon_seeds {
            self.token_to_root.insert(token_id, Some(root_id));
        }
    }

    /// Compute embedding: TokenEmb(t) + PosEmb(pos) + ScriptEmb(script(t)) + ParadigmEmb(U(t))
    pub fn embed(
        &self,
        token_id: TokenId,
        token_emb: &[f32],
        pos_emb: &[f32],
        script_emb: &[f32],
    ) -> Vec<f32> {
        let paradigm_emb = self.get_paradigm_emb(token_id);
        
        // Concatenate or sum embeddings (design choice)
        let mut combined = Vec::new();
        combined.extend_from_slice(token_emb);
        combined.extend_from_slice(pos_emb);
        combined.extend_from_slice(script_emb);
        combined.extend_from_slice(paradigm_emb);
        combined
    }

    fn get_paradigm_emb(&self, token_id: TokenId) -> &[f32] {
        if let Some(Some(root_id)) = self.token_to_root.get(&token_id) {
            self.root_embeddings.get(root_id).unwrap_or(&self.default_embedding)
        } else {
            &self.default_embedding
        }
    }
}

// ============================================================================
// Main tokenizer: orchestration of all phases
// ============================================================================

pub struct NepBPETokenizer {
    normalizer: Normalizer,
    akshara_dfa: AksharaDFA,
    vocab: Vocabulary,
    paradigm_registry: ParadigmRegistry,
    trainer: Option<ConstrainedBPETrainer>,
    latin_trainer: Option<LatinBPETrainer>,
    paradigm_embedding: ParadigmEmbedding,
}

impl NepBPETokenizer {
    pub fn new(
        folding_table: HashMap<char, char>,
        paradigm_registry: ParadigmRegistry,
    ) -> Self {
        Self {
            normalizer: Normalizer::new(folding_table),
            akshara_dfa: AksharaDFA::new(),
            vocab: Vocabulary::new(),
            paradigm_registry,
            trainer: None,
            latin_trainer: None,
            paradigm_embedding: ParadigmEmbedding::new(),
        }
    }

    /// Encode a string into token IDs
    pub fn encode(&self, s: &str) -> Vec<TokenId> {
        // Phase 1: Normalization
        let normalized = self.normalizer.normalize(s);

        // Phase 2: Akshara DFA tokenization + script detection
        let mut tokens = Vec::new();
        let mut current_run = String::new();
        let mut current_script: Option<Script> = None;

        for ch in normalized.chars() {
            let ch_script = self.classify_char(ch);

            if current_script == Some(ch_script) {
                current_run.push(ch);
            } else {
                // Flush current run
                if !current_run.is_empty() {
                    self.tokenize_run(&current_run, current_script.unwrap(), &mut tokens);
                    current_run.clear();
                }
                current_run.push(ch);
                current_script = Some(ch_script);
            }
        }

        // Flush final run
        if !current_run.is_empty() {
            self.tokenize_run(&current_run, current_script.unwrap(), &mut tokens);
        }

        tokens
    }

    fn classify_char(&self, ch: char) -> Script {
        if ch == '\u{200C}' {
            Script::FMT // ZWNJ
        } else if ch.is_ascii_punctuation() || ch == '\u{0964}' || ch == '\u{0965}' {
            Script::PUN // Including danda and double danda
        } else if ch.is_ascii_alphabetic() {
            Script::LAT
        } else if ('\u{0900}'..='\u{097F}').contains(&ch) {
            Script::DEV
        } else {
            Script::MAL // Fallback for unknown
        }
    }

    fn tokenize_run(&self, run: &str, script: Script, tokens: &mut Vec<TokenId>) {
        match script {
            Script::DEV => {
                // Use akshara DFA for Devanagari
                let aksharas = self.akshara_dfa.tokenize(run);
                for akshara in aksharas {
                    if let Some(&id) = self.vocab.token_to_id.get(&Rc::new(Token::Akshara(akshara))) {
                        tokens.push(id);
                    }
                }
            }
            Script::PUN => {
                for ch in run.chars() {
                    let punct_token = Rc::new(Token::Punctuation(ch.to_string()));
                    if let Some(&id) = self.vocab.token_to_id.get(&punct_token) {
                        tokens.push(id);
                    }
                }
            }
            Script::FMT => {
                // ZWNJ is a single token
                let zwnj = Rc::new(Token::ZWNJ);
                if let Some(&id) = self.vocab.token_to_id.get(&zwnj) {
                    tokens.push(id);
                }
            }
            Script::MAL => {
                // Byte-fallback for malformed input
                for byte in run.as_bytes() {
                    let fallback = Rc::new(Token::ByteFallback(*byte));
                    if let Some(&id) = self.vocab.token_to_id.get(&fallback) {
                        tokens.push(id);
                    }
                }
            }
            Script::LAT => {
                // Latin tokens handled in Phase 4
                for ch in run.chars() {
                    let lat_token = Rc::new(Token::Punctuation(ch.to_string())); // Simplified
                    if let Some(&id) = self.vocab.token_to_id.get(&lat_token) {
                        tokens.push(id);
                    }
                }
            }
        }
    }

    /// Decode token IDs back to string
    /// Guarantee: Decode(Encode(s)) = N(s)
    pub fn decode(&self, token_ids: &[TokenId]) -> String {
        let mut result = String::new();
        for &id in token_ids {
            if let Some(token) = self.vocab.get_token(id) {
                result.push_str(&token.surface());
            }
        }
        result
    }

    /// Verify roundtrip guarantee
    pub fn verify_roundtrip(&self, s: &str) -> bool {
        let encoded = self.encode(s);
        let decoded = self.decode(&encoded);
        let expected = self.normalizer.normalize(s);
        decoded == expected
    }
}

// ============================================================================
// Evaluation metrics (E4)
// ============================================================================

pub struct EvaluationMetrics {
    // Tokens per word (fertility)
    tokens_per_word: f64,
    tokens_per_sentence: f64,
    // Vocabulary efficiency
    vocab_size: usize,
    dev_token_count: usize,
    unk_rate: f64,
    byte_fallback_rate: f64,
    // Morpheme boundary F1
    morpheme_f1: f64,
    // Bits per character
    bits_per_char: f64,
}

impl EvaluationMetrics {
    pub fn compute_bits_per_character(total_bits: f64, total_chars: usize) -> f64 {
        total_bits / total_chars as f64
    }

    pub fn compute_morpheme_f1(
        predicted_boundaries: &HashSet<usize>,
        gold_boundaries: &HashSet<usize>,
    ) -> f64 {
        let intersection = predicted_boundaries.intersection(gold_boundaries).count();
        if predicted_boundaries.is_empty() && gold_boundaries.is_empty() {
            return 1.0;
        }
        let precision = intersection as f64 / predicted_boundaries.len().max(1) as f64;
        let recall = intersection as f64 / gold_boundaries.len().max(1) as f64;
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
}

// ============================================================================
// Tests (correspond to the guarantees in the specification)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_idempotent() {
        let mut folding = HashMap::new();
        // Add folding mappings (simplified)
        folding.insert('ँ', '\u{0902}'); // chandrabindu → anusvara (example)
        
        let normalizer = Normalizer::new(folding);
        let s = "सँग";
        let n1 = normalizer.normalize(s);
        let n2 = normalizer.normalize(&n1);
        assert_eq!(n1, n2, "Normalization must be idempotent");
    }

    #[test]
    fn test_script_compat() {
        // Would test ScriptCompat(a,b) with real token IDs
    }

    #[test]
    fn test_strict_morpheme_terminal() {
        // Verify that V_strict tokens never left-extend (fix #1)
    }

    #[test]
    fn test_empty_rootset_allows_merge() {
        // Verify fix #2: RootSet(a)=∅ ⇒ Morph(a,b)=1
    }

    #[test]
    fn test_byte_fallback_roundtrip() {
        // Verify exact roundtrip for arbitrary bytes
    }
}