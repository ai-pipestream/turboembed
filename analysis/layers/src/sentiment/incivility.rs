use turbo_analysis::{Anchor, Ix, IxRange, Key, KeyId, Layer, Span};

use crate::{Chunks, Sentences, Structure, Tokens};

use super::{IncivilityLabel, IncivilityScores, check_token_range};

/// Which unit span an [`Incivility`] item scores: a sentence, a retrieval
/// chunk, or a structure block (such as a post, quote, or paragraph).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IncivilityTarget {
    Sentence(Ix<Sentences>),
    Chunk(Ix<Chunks>),
    Block(Ix<Structure>),
}

impl IncivilityTarget {
    pub const fn kind_name(self) -> &'static str {
        match self {
            IncivilityTarget::Sentence(_) => "sentence",
            IncivilityTarget::Chunk(_) => "chunk",
            IncivilityTarget::Block(_) => "block",
        }
    }

    pub fn raw_index(self) -> u32 {
        match self {
            IncivilityTarget::Sentence(i) => i.get() as u32,
            IncivilityTarget::Chunk(i) => i.get() as u32,
            IncivilityTarget::Block(i) => i.get() as u32,
        }
    }

    pub fn from_kind_and_index(kind: &str, index: u32) -> Option<IncivilityTarget> {
        match kind {
            "sentence" => Some(IncivilityTarget::Sentence(Ix::new(index))),
            "chunk" => Some(IncivilityTarget::Chunk(Ix::new(index))),
            "block" => Some(IncivilityTarget::Block(Ix::new(index))),
            _ => None,
        }
    }
}

/// One item handed to [`Incivility::try_push`] or [`Incivility::push`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IncivilityItem<'a> {
    pub target: IncivilityTarget,
    pub scores: IncivilityScores,
    pub evidence: &'a [(IncivilityLabel, IxRange<Tokens>)],
}

/// Multi-label incivility and toxicity (`turbo:incivility`) anchored to
/// sentence, chunk, or structure (post) spans, with six `f64` label scores
/// and the evidence token spans per label.
#[derive(Clone, Debug)]
pub struct Incivility {
    pub target: Vec<IncivilityTarget>,
    pub insult: Vec<f64>,
    pub profanity: Vec<f64>,
    pub threat: Vec<f64>,
    pub identity_attack: Vec<f64>,
    pub intolerance: Vec<f64>,
    pub severe_toxicity: Vec<f64>,
    /// Flat CSR column of evidence labels, sorted by `(label, tokens.start,
    /// tokens.end)` within each item.
    pub evidence_label: Vec<IncivilityLabel>,
    /// Flat CSR column of evidence token spans, parallel to
    /// `evidence_label`.
    pub evidence_tokens: Vec<IxRange<Tokens>>,
    /// End offset in `evidence_label` / `evidence_tokens` for each item
    /// `0..len`.
    pub evidence_end: Vec<u32>,
    pub tokens_key: Key<Tokens>,
    pub sentences_key: Key<Sentences>,
    pub chunks_key: Key<Chunks>,
    pub structure_key: Key<Structure>,
}

impl Default for Incivility {
    fn default() -> Self {
        Self::new()
    }
}

impl Incivility {
    /// An empty `Incivility` layer indexing the default [`Tokens`],
    /// [`Sentences`], [`Chunks`], and [`Structure`] instances.
    pub fn new() -> Incivility {
        Self::on(Key::DEFAULT, Key::DEFAULT, Key::DEFAULT, Key::DEFAULT)
    }

    /// An empty `Incivility` layer indexing the given named instances.
    pub fn on(
        tokens_key: Key<Tokens>,
        sentences_key: Key<Sentences>,
        chunks_key: Key<Chunks>,
        structure_key: Key<Structure>,
    ) -> Incivility {
        Incivility {
            target: Vec::new(),
            insult: Vec::new(),
            profanity: Vec::new(),
            threat: Vec::new(),
            identity_attack: Vec::new(),
            intolerance: Vec::new(),
            severe_toxicity: Vec::new(),
            evidence_label: Vec::new(),
            evidence_tokens: Vec::new(),
            evidence_end: Vec::new(),
            tokens_key,
            sentences_key,
            chunks_key,
            structure_key,
        }
    }

    /// Builds an `Incivility` layer from `items`, bounds-checking every
    /// target span index and every evidence token range.
    pub fn from_items<'a>(
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        chunks: Option<&Chunks>,
        structure: Option<&Structure>,
        items: impl IntoIterator<Item = IncivilityItem<'a>>,
    ) -> Result<Incivility, String> {
        Self::from_items_on(
            Key::DEFAULT,
            Key::DEFAULT,
            Key::DEFAULT,
            Key::DEFAULT,
            tokens,
            sentences,
            chunks,
            structure,
            items,
        )
    }

    /// Builds an `Incivility` layer indexing explicit keys and
    /// bounds-checking every index.
    #[allow(clippy::too_many_arguments)]
    pub fn from_items_on<'a>(
        tokens_key: Key<Tokens>,
        sentences_key: Key<Sentences>,
        chunks_key: Key<Chunks>,
        structure_key: Key<Structure>,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        chunks: Option<&Chunks>,
        structure: Option<&Structure>,
        items: impl IntoIterator<Item = IncivilityItem<'a>>,
    ) -> Result<Incivility, String> {
        let mut out = Self::on(tokens_key, sentences_key, chunks_key, structure_key);
        for item in items {
            out.try_push(tokens, sentences, chunks, structure, item)?;
        }
        Ok(out)
    }

    /// Bounds-checks `item` against the target layers and `tokens`, then
    /// appends it.
    pub fn try_push(
        &mut self,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        chunks: Option<&Chunks>,
        structure: Option<&Structure>,
        item: IncivilityItem<'_>,
    ) -> Result<Ix<Incivility>, String> {
        let i = self.target.len();
        if let Some(&prev) = self.target.last()
            && item.target <= prev
        {
            return Err(format!("incivility item {i} target is not in strictly increasing target order"));
        }
        match item.target {
            IncivilityTarget::Sentence(s) => {
                let Some(sents) = sentences else {
                    return Err(format!(
                        "incivility item {i} targets sentence {}, but no Sentences layer was given",
                        s.get()
                    ));
                };
                if s.get() >= sents.len() {
                    return Err(format!(
                        "incivility item {i} sentence index {} is out of bounds for {} sentences",
                        s.get(),
                        sents.len()
                    ));
                }
            }
            IncivilityTarget::Chunk(c) => {
                let Some(ch) = chunks else {
                    return Err(format!(
                        "incivility item {i} targets chunk {}, but no Chunks layer was given",
                        c.get()
                    ));
                };
                if c.get() >= ch.len() {
                    return Err(format!(
                        "incivility item {i} chunk index {} is out of bounds for {} chunks",
                        c.get(),
                        ch.len()
                    ));
                }
            }
            IncivilityTarget::Block(b) => {
                let Some(st) = structure else {
                    return Err(format!(
                        "incivility item {i} targets block {}, but no Structure layer was given",
                        b.get()
                    ));
                };
                if b.get() >= st.len() {
                    return Err(format!(
                        "incivility item {i} block index {} is out of bounds for {} blocks",
                        b.get(),
                        st.len()
                    ));
                }
            }
        }
        if !item.scores.is_valid() {
            return Err(format!("incivility item {i} has a score outside 0..=1 or NaN"));
        }
        for &(_, r) in item.evidence {
            check_token_range("incivility evidence", i, r, tokens.len())?;
        }
        Ok(self.push(item))
    }

    /// Appends `item`, sorting and deduplicating its evidence pairs by
    /// `(label, tokens.start, tokens.end)`.
    pub fn push(&mut self, item: IncivilityItem<'_>) -> Ix<Incivility> {
        let ix = Ix::new(self.target.len() as u32);
        self.target.push(item.target);
        self.insult.push(item.scores.insult);
        self.profanity.push(item.scores.profanity);
        self.threat.push(item.scores.threat);
        self.identity_attack.push(item.scores.identity_attack);
        self.intolerance.push(item.scores.intolerance);
        self.severe_toxicity.push(item.scores.severe_toxicity);

        let mut sorted: Vec<(IncivilityLabel, IxRange<Tokens>)> = item.evidence.to_vec();
        sorted.sort_unstable_by_key(|&(l, r)| (l, r.start, r.end));
        sorted.dedup();
        for (l, r) in sorted {
            self.evidence_label.push(l);
            self.evidence_tokens.push(r);
        }
        self.evidence_end.push(self.evidence_label.len() as u32);
        ix
    }

    /// The six multi-label scores of item `i`.
    pub fn scores(&self, i: Ix<Incivility>) -> IncivilityScores {
        let k = i.get();
        IncivilityScores {
            insult: self.insult[k],
            profanity: self.profanity[k],
            threat: self.threat[k],
            identity_attack: self.identity_attack[k],
            intolerance: self.intolerance[k],
            severe_toxicity: self.severe_toxicity[k],
        }
    }

    fn evidence_range(&self, i: Ix<Incivility>) -> std::ops::Range<usize> {
        let k = i.get();
        let start = if k == 0 { 0 } else { self.evidence_end[k - 1] as usize };
        let end = self.evidence_end[k] as usize;
        start..end
    }

    /// All `(label, tokens)` evidence entries for item `i`, in canonical
    /// `(label, start, end)` order.
    pub fn evidence_of(
        &self,
        i: Ix<Incivility>,
    ) -> impl ExactSizeIterator<Item = (IncivilityLabel, IxRange<Tokens>)> + '_ {
        let range = self.evidence_range(i);
        self.evidence_label[range.clone()].iter().copied().zip(self.evidence_tokens[range].iter().copied())
    }

    /// The contiguous slice of evidence token spans for `label` on item
    /// `i`, found by binary search with zero allocation.
    pub fn evidence_for_label(&self, i: Ix<Incivility>, label: IncivilityLabel) -> &[IxRange<Tokens>] {
        let range = self.evidence_range(i);
        let labels = &self.evidence_label[range.clone()];
        let tokens = &self.evidence_tokens[range];
        let a = labels.partition_point(|&l| l < label);
        let b = a + labels[a..].partition_point(|&l| l == label);
        &tokens[a..b]
    }

    /// The byte span of item `i`'s target in the corresponding layer.
    pub fn target_span(
        &self,
        i: Ix<Incivility>,
        sentences: Option<&Sentences>,
        chunks: Option<&Chunks>,
        structure: Option<&Structure>,
    ) -> Option<Span> {
        match self.target[i.get()] {
            IncivilityTarget::Sentence(s) => sentences.map(|l| l.spans.get(s)),
            IncivilityTarget::Chunk(c) => chunks.map(|l| l.spans.get(c)),
            IncivilityTarget::Block(b) => structure.map(|l| l.spans.get(b)),
        }
    }
}

impl Layer for Incivility {
    const ID: &'static str = "turbo:incivility";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.target.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        // Evidence spans are optional, so items scored without evidence do
        // not require a Tokens layer.
        let max_tok = self.evidence_tokens.iter().map(|r| r.end.get()).max().unwrap_or(0);
        let mut out = Vec::new();
        if max_tok > 0 || !self.tokens_key.name.is_empty() {
            out.push((self.tokens_key.id(), max_tok));
        }
        let mut max_sent = 0usize;
        let mut max_chunk = 0usize;
        let mut max_block = 0usize;
        for &t in &self.target {
            match t {
                IncivilityTarget::Sentence(s) => max_sent = max_sent.max(s.get() + 1),
                IncivilityTarget::Chunk(c) => max_chunk = max_chunk.max(c.get() + 1),
                IncivilityTarget::Block(b) => max_block = max_block.max(b.get() + 1),
            }
        }
        if max_sent > 0 || !self.sentences_key.name.is_empty() {
            out.push((self.sentences_key.id(), max_sent));
        }
        if max_chunk > 0 || !self.chunks_key.name.is_empty() {
            out.push((self.chunks_key.id(), max_chunk));
        }
        if max_block > 0 || !self.structure_key.name.is_empty() {
            out.push((self.structure_key.id(), max_block));
        }
        out
    }

    fn check(&self) -> Result<(), String> {
        let n = self.target.len();
        let cols = [
            self.insult.len(),
            self.profanity.len(),
            self.threat.len(),
            self.identity_attack.len(),
            self.intolerance.len(),
            self.severe_toxicity.len(),
            self.evidence_end.len(),
        ];
        if cols.iter().any(|&c| c != n) {
            return Err("every incivility column must have the same length".into());
        }
        if self.evidence_label.len() != self.evidence_tokens.len() {
            return Err("an evidence token range for every evidence label".into());
        }
        if !self.target.windows(2).all(|w| w[0] < w[1]) {
            return Err("incivility targets must be in strictly increasing order".into());
        }
        let mut prev_end = 0usize;
        for i in 0..n {
            if !self.scores(Ix::new(i as u32)).is_valid() {
                return Err(format!("incivility item {i} has a score outside 0..=1 or NaN"));
            }
            let end = self.evidence_end[i] as usize;
            if end < prev_end || end > self.evidence_label.len() {
                return Err(format!("incivility item {i} evidence_end {end} is out of bounds or decreases"));
            }
            for k in prev_end..end {
                let r = self.evidence_tokens[k];
                if r.is_empty() {
                    return Err(format!(
                        "incivility item {i} has an empty evidence token range {}..{}",
                        r.start.get(),
                        r.end.get()
                    ));
                }
                if k > prev_end {
                    let prev_key = (
                        self.evidence_label[k - 1],
                        self.evidence_tokens[k - 1].start,
                        self.evidence_tokens[k - 1].end,
                    );
                    if (self.evidence_label[k], r.start, r.end) <= prev_key {
                        return Err(format!(
                            "incivility item {i} evidence must be in strictly increasing (label, start, end) order"
                        ));
                    }
                }
            }
            prev_end = end;
        }
        if prev_end != self.evidence_label.len() {
            return Err("evidence_end does not reach the end of evidence_label".into());
        }
        Ok(())
    }
}
