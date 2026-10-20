//! Rule-driven POS-tag pattern chunker (`PatternChunker`) and composition with
//! trained `ChunkerME` models (`PatternChunkComposer`, `ComposeMode`).

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, IxRange, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_layers::{PhraseChunks, PosTags, Tag, TagSet, Tokens};
use turbo_sequence::chunker::TokenTag;

use crate::chunker::{PhraseChunker, sentence_ranges};
use crate::pattern::{PatternError, TagPatternRule};

/// A matched phrase chunk produced by [`PatternChunker`] within a token sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatternMatch {
    /// Start token index (inclusive).
    pub start: usize,
    /// End token index (exclusive, `> start`).
    pub end: usize,
    /// Phrase chunk kind label (for example `"NP"`, `"VP"`, `"PP"`).
    pub kind: String,
    /// Index of the [`TagPatternRule`] that produced this match.
    pub rule_index: usize,
}

/// Zero-training rule-driven phrase chunker over [`Tokens`] and [`PosTags`].
///
/// Evaluates [`TagPatternRule`]s in rule priority order using longest-match
/// left-to-right non-overlapping resolution per rule and writes [`PhraseChunks`].
#[derive(Clone, Debug)]
pub struct PatternChunker {
    rules: Vec<TagPatternRule>,
    kinds: Arc<TagSet>,
    config_hash: u64,
}

impl PatternChunker {
    /// Constructs a [`PatternChunker`] from an ordered list of compiled [`TagPatternRule`]s.
    /// Earlier rules take priority over later rules when matches would overlap.
    pub fn new(rules: Vec<TagPatternRule>) -> Self {
        let mut kind_labels: Vec<&str> = Vec::new();
        for r in &rules {
            if !kind_labels.contains(&r.kind()) {
                kind_labels.push(r.kind());
            }
        }
        let kinds = TagSet::new("chunks", &kind_labels);
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        h = (h ^ (rules.len() as u64)).wrapping_mul(0x0000_0100_0000_01b3);
        for r in &rules {
            for b in r.fingerprint().to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Self { rules, kinds, config_hash: h }
    }

    /// Compiles a list of `(kind, pattern)` specifications in priority order.
    pub fn from_patterns<I, K, P>(specs: I) -> Result<Self, PatternError>
    where
        I: IntoIterator<Item = (K, P)>,
        K: Into<String>,
        P: Into<String>,
    {
        let mut rules = Vec::new();
        for (kind, pattern) in specs {
            rules.push(TagPatternRule::compile(kind, pattern)?);
        }
        Ok(Self::new(rules))
    }

    /// Standard shallow phrase-chunking rules (`NP`, `VP`, `PP`) for Penn Treebank POS tags.
    pub fn penn_default() -> Self {
        Self::from_patterns([
            ("NP", "<PDT>?<DT|PRP$|WDT|WP$>?<CD>*<RB.*|JJ.*>*<NN.*|PRP|WP>+"),
            ("VP", "<MD>*<RB.*>*<VB.*>+<RP>?"),
            ("PP", "<IN|TO>"),
        ])
        .expect("valid built-in Penn Treebank chunk patterns")
    }

    /// Standard shallow phrase-chunking rules (`NP`, `VP`, `PP`) for Universal Dependencies POS tags.
    pub fn ud_default() -> Self {
        Self::from_patterns([
            ("NP", "<DET>?<NUM>*<ADV|ADJ>*<NOUN|PROPN|PRON>+"),
            ("VP", "<AUX>*<ADV>*<VERB>+<PART>?"),
            ("PP", "<ADP>"),
        ])
        .expect("valid built-in Universal Dependencies chunk patterns")
    }

    /// Compiled rules in priority order.
    pub fn rules(&self) -> &[TagPatternRule] {
        &self.rules
    }

    /// Phrase kind [`TagSet`] emitted by this chunker.
    pub fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }

    /// Chunks a single sentence of tagged tokens using longest-match left-to-right
    /// non-overlapping resolution per rule in rule priority order.
    pub fn chunk_sentence(&self, sentence: &[TokenTag<'_>]) -> Vec<PatternMatch> {
        let mut occupied = vec![false; sentence.len()];
        self.chunk_sentence_with_occupied(sentence, &mut occupied)
    }

    /// Chunks unclaimed tokens in `sentence`, marking newly claimed tokens in `occupied`.
    pub fn chunk_sentence_with_occupied(&self, sentence: &[TokenTag<'_>], occupied: &mut [bool]) -> Vec<PatternMatch> {
        let n = sentence.len();
        if n == 0 || self.rules.is_empty() {
            return Vec::new();
        }
        let mut matches = Vec::new();
        for (rule_index, rule) in self.rules.iter().enumerate() {
            let mut pos = 0;
            while pos < n {
                if occupied[pos] {
                    pos += 1;
                    continue;
                }
                if let Some(end) = rule.longest_match_unclaimed(sentence, pos, occupied) {
                    debug_assert!(end > pos && end <= n);
                    for slot in &mut occupied[pos..end] {
                        *slot = true;
                    }
                    matches.push(PatternMatch { start: pos, end, kind: rule.kind().to_owned(), rule_index });
                    pos = end;
                } else {
                    pos += 1;
                }
            }
        }
        matches.sort_by_key(|m| (m.start, m.end));
        matches
    }
}

pub(crate) fn validate_tokens_and_tags(
    annotator_id: &'static str,
    view: &View,
    tokens: &Tokens,
    tags: &PosTags,
) -> Result<usize, Error> {
    if tokens.spans.view() != view.id() {
        return Err(Error::Annotator {
            annotator: annotator_id,
            detail: format!(
                "the tokens are in view {}, not in view {} that is chunked",
                tokens.spans.view().0,
                view.id().0
            ),
        });
    }
    let text = view.text();
    let (starts, ends) = tokens.spans.columns();
    if starts.len() != ends.len() {
        return Err(Error::Annotator {
            annotator: annotator_id,
            detail: format!("token span start count ({}) does not match end count ({})", starts.len(), ends.len()),
        });
    }
    let n = starts.len();
    if let Some(col) = &tokens.sentence
        && col.len() != n
    {
        return Err(Error::Annotator {
            annotator: annotator_id,
            detail: format!("{} sentence indices for {n} tokens", col.len()),
        });
    }
    if tags.tags.len() != n {
        return Err(Error::Annotator {
            annotator: annotator_id,
            detail: format!("{} tags for {n} tokens", tags.tags.len()),
        });
    }
    let tagset_len = tags.tagset.len();
    for (i, &tag) in tags.tags.iter().enumerate() {
        if usize::from(tag.0) >= tagset_len {
            return Err(Error::Annotator {
                annotator: annotator_id,
                detail: format!("tag index {} at token {i} is out of bounds for tagset of length {tagset_len}", tag.0),
            });
        }
    }
    for i in 0..n {
        let s = starts[i] as usize;
        let e = ends[i] as usize;
        if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
            return Err(Error::Annotator {
                annotator: annotator_id,
                detail: format!(
                    "token {i} span {s}..{e} is not a valid UTF-8 slice of view text of length {}",
                    text.len()
                ),
            });
        }
    }
    for (from, to) in sentence_ranges(tokens) {
        for i in (from + 1)..to {
            if starts[i] < starts[i - 1] || ends[i] < ends[i - 1] {
                return Err(Error::Annotator {
                    annotator: annotator_id,
                    detail: format!(
                        "token {i} span {}..{} precedes previous token in sentence {from}..{to}",
                        starts[i], ends[i]
                    ),
                });
            }
        }
    }
    Ok(n)
}

impl Annotator for PatternChunker {
    const ID: &'static str = "turbo-phrase-chunker:pattern";
    type Reads = (View, Tokens, PosTags);
    type Writes = (PhraseChunks,);

    fn annotate(&self, (view, tokens, tags): Refs<'_, Self>) -> Result<(PhraseChunks,), Error> {
        let _n = validate_tokens_and_tags(Self::ID, view, tokens, tags)?;
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();

        let mut spans = Spans::new(view.id());
        let mut ranges = Vec::new();
        let mut kind = Vec::new();
        let mut words: Vec<TokenTag<'_>> = Vec::new();

        for (from, to) in sentence_ranges(tokens) {
            words.clear();
            words.extend((from..to).map(|i| TokenTag {
                token: &text[starts[i] as usize..ends[i] as usize],
                tag: tags.tagset.label(tags.tags[i]),
            }));
            for m in self.chunk_sentence(&words) {
                let tok_start = from + m.start;
                let tok_end = from + m.end;
                spans.push(Span::new(starts[tok_start], ends[tok_end - 1]));
                ranges.push(IxRange::new(tok_start as u32, tok_end as u32));
                kind.push(self.kinds.tag(&m.kind).expect("rule kind registered in PatternChunker::kinds"));
            }
        }

        Ok((PhraseChunks { spans, tokens: ranges, kind, kinds: self.kinds.clone() },))
    }

    fn config_hash(&self) -> u64 {
        self.config_hash
    }
}

/// Strategy for combining [`PatternChunker`] rule matches with trained `ChunkerME` [`PhraseChunks`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComposeMode {
    /// Retains all trained `ChunkerME` chunks and adds non-overlapping [`PatternChunker`]
    /// matches over tokens not covered by any trained chunk.
    Fallback,
    /// Applies [`PatternChunker`] rules first and retains only trained `ChunkerME` chunks
    /// whose token ranges do not overlap any rule match.
    Override,
}

/// Composes [`PatternChunker`] rules with trained [`PhraseChunker`] (`ChunkerME`) output
/// in either [`ComposeMode::Fallback`] or [`ComposeMode::Override`] mode.
#[derive(Clone, Debug)]
pub struct PatternChunkComposer {
    pattern_chunker: PatternChunker,
    mode: ComposeMode,
    me_chunker: Option<Arc<PhraseChunker>>,
}

impl PatternChunkComposer {
    /// Creates a composer for `pattern_chunker` with the given [`ComposeMode`].
    pub fn new(pattern_chunker: PatternChunker, mode: ComposeMode) -> Self {
        Self { pattern_chunker, mode, me_chunker: None }
    }

    /// Attaches a trained [`PhraseChunker`] (`ChunkerME`) so [`Annotator::annotate`] runs
    /// `ChunkerME` and composes its output with the pattern rules in a single step.
    pub fn with_me(mut self, me_chunker: impl Into<Arc<PhraseChunker>>) -> Self {
        self.me_chunker = Some(me_chunker.into());
        self
    }

    pub fn mode(&self) -> ComposeMode {
        self.mode
    }

    pub fn pattern_chunker(&self) -> &PatternChunker {
        &self.pattern_chunker
    }

    /// Merges an existing `base` [`PhraseChunks`] layer (such as from `ChunkerME`) with
    /// [`PatternChunker`] matches over `(view, tokens, tags)` according to [`Self::mode`].
    pub fn compose(
        &self,
        view: &View,
        tokens: &Tokens,
        tags: &PosTags,
        base: &PhraseChunks,
    ) -> Result<PhraseChunks, Error> {
        let n = validate_tokens_and_tags(Self::ID, view, tokens, tags)?;
        if base.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("base PhraseChunks are in view {}, not in view {}", base.spans.view().0, view.id().0),
            });
        }
        let (b_starts, b_ends) = base.spans.columns();
        if b_starts.len() != b_ends.len() || base.tokens.len() != b_starts.len() || base.kind.len() != b_starts.len() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "base PhraseChunks column lengths do not match".into(),
            });
        }

        let (starts, ends) = tokens.spans.columns();
        let text = view.text();
        let sentences = sentence_ranges(tokens);

        // Validate base chunks and extract (tok_start, tok_end, kind_label)
        let mut base_items: Vec<(usize, usize, String)> = Vec::with_capacity(base.tokens.len());
        for i in 0..base.tokens.len() {
            let r = base.tokens[i];
            let ts = r.start.get();
            let te = r.end.get();
            if ts >= te || te > n {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("base chunk {i} token range {ts}..{te} is invalid for {n} tokens"),
                });
            }
            if b_starts[i] != starts[ts] || b_ends[i] != ends[te - 1] {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "base chunk {i} span {}..{} does not match token range {ts}..{te} ({}..{})",
                        b_starts[i],
                        b_ends[i],
                        starts[ts],
                        ends[te - 1]
                    ),
                });
            }
            if !sentences.iter().any(|&(from, to)| ts >= from && te <= to) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("base chunk {i} token range {ts}..{te} crosses a sentence boundary"),
                });
            }
            if usize::from(base.kind[i].0) >= base.kinds.len() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("base chunk {i} kind index {} out of bounds", base.kind[i].0),
                });
            }
            if i > 0 && ts < base.tokens[i - 1].end.get() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("base chunk {i} at {ts}..{te} overlaps previous chunk"),
                });
            }
            base_items.push((ts, te, base.kinds.label(base.kind[i]).to_owned()));
        }

        let mut merged: Vec<(usize, usize, String)> = Vec::new();
        let mut words: Vec<TokenTag<'_>> = Vec::new();

        for (from, to) in sentences {
            let sent_len = to - from;
            let mut occupied = vec![false; sent_len];
            words.clear();
            words.extend((from..to).map(|i| TokenTag {
                token: &text[starts[i] as usize..ends[i] as usize],
                tag: tags.tagset.label(tags.tags[i]),
            }));

            let sent_base: Vec<&(usize, usize, String)> =
                base_items.iter().filter(|&&(ts, te, _)| ts >= from && te <= to).collect();

            match self.mode {
                ComposeMode::Fallback => {
                    for &&(ts, te, ref k) in &sent_base {
                        for slot in &mut occupied[(ts - from)..(te - from)] {
                            *slot = true;
                        }
                        merged.push((ts, te, k.clone()));
                    }
                    for m in self.pattern_chunker.chunk_sentence_with_occupied(&words, &mut occupied) {
                        merged.push((from + m.start, from + m.end, m.kind));
                    }
                }
                ComposeMode::Override => {
                    for m in self.pattern_chunker.chunk_sentence_with_occupied(&words, &mut occupied) {
                        merged.push((from + m.start, from + m.end, m.kind));
                    }
                    for &&(ts, te, ref k) in &sent_base {
                        let rel_s = ts - from;
                        let rel_e = te - from;
                        if !occupied[rel_s..rel_e].iter().any(|&b| b) {
                            for slot in &mut occupied[rel_s..rel_e] {
                                *slot = true;
                            }
                            merged.push((ts, te, k.clone()));
                        }
                    }
                }
            }
        }

        merged.sort_by_key(|&(ts, te, _)| (ts, te));

        let mut kind_labels: Vec<&str> = Vec::new();
        for i in 0..base.kinds.len() {
            let l = base.kinds.label(Tag(i as u16));
            if !kind_labels.contains(&l) {
                kind_labels.push(l);
            }
        }
        for i in 0..self.pattern_chunker.kinds().len() {
            let l = self.pattern_chunker.kinds().label(Tag(i as u16));
            if !kind_labels.contains(&l) {
                kind_labels.push(l);
            }
        }
        let kinds = TagSet::new("chunks", &kind_labels);

        let mut spans = Spans::new(view.id());
        let mut ranges = Vec::with_capacity(merged.len());
        let mut kind_col = Vec::with_capacity(merged.len());
        for (ts, te, k_label) in merged {
            spans.push(Span::new(starts[ts], ends[te - 1]));
            ranges.push(IxRange::new(ts as u32, te as u32));
            kind_col.push(kinds.tag(&k_label).expect("merged kind registered in TagSet"));
        }

        Ok(PhraseChunks { spans, tokens: ranges, kind: kind_col, kinds })
    }
}

impl Annotator for PatternChunkComposer {
    const ID: &'static str = "turbo-phrase-chunker:pattern-composer";
    type Reads = (View, Tokens, PosTags);
    type Writes = (PhraseChunks,);

    fn annotate(&self, (view, tokens, tags): Refs<'_, Self>) -> Result<(PhraseChunks,), Error> {
        if let Some(me) = &self.me_chunker {
            let (_chunk_tags, base_chunks) = me.annotate((view, tokens, tags))?;
            let composed = self.compose(view, tokens, tags, &base_chunks)?;
            Ok((composed,))
        } else {
            self.pattern_chunker.annotate((view, tokens, tags))
        }
    }

    fn model(&self) -> Option<ModelRef> {
        self.me_chunker.as_ref().and_then(|me| me.model())
    }

    fn config_hash(&self) -> u64 {
        let mut h = self.pattern_chunker.config_hash();
        let mode_byte = match self.mode {
            ComposeMode::Fallback => 1u8,
            ComposeMode::Override => 2u8,
        };
        h = (h ^ u64::from(mode_byte)).wrapping_mul(0x0000_0100_0000_01b3);
        if let Some(me) = &self.me_chunker {
            for b in me.config_hash().to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
            }
            if let Some(mref) = me.model() {
                for s in [mref.id.as_bytes(), mref.revision.as_bytes(), mref.sha256.as_bytes()] {
                    for &b in s {
                        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
            }
        }
        h
    }
}
