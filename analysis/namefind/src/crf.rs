//! CRF document integration over caller-provided numerical scores. The
//! caller owns the model side: it binds the typed entity schema of
//! [`EntityDecoder`], supplies per-sentence [`CrfSentenceScores`], and
//! names the [`ModelRef`] provenance recorded on every document the
//! annotator runs on. There is no model reader in this crate; scores
//! arrive as numbers, never as archives.
//!
//! The companion alternatives layer is a union of candidate spans with
//! unit-token fallback edges. Exact ranked paths remain available through
//! the decoder's lattice; arbitrary recombinations of lattice edges are
//! not ranked CRF paths.
use std::collections::BTreeMap;
use std::sync::Arc;
use turbo_analysis::{
    Alternatives, Annotator, Error, Fnv1a, Ix, IxRange, Key, Layer, Locality, ModeSet, ModelRef, Refs, Span, Spans,
    Tag, View,
};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_sequence::crf::{CrfScores, EntityDecoder, ScoredEntity};

/// Numerical CRF parameters of one sentence, in [`CrfScores`] layout. The
/// caller produces them from its own model; the decoder never sees it.
#[derive(Clone, Debug)]
pub struct CrfSentenceScores {
    pub emissions: Vec<Vec<f64>>,
    pub transitions: Vec<f64>,
    pub start: Vec<f64>,
    pub end: Vec<f64>,
}

impl CrfSentenceScores {
    /// Borrows the parameters in the layout the typed decoder consumes.
    pub fn as_scores(&self) -> CrfScores<'_> {
        CrfScores { emissions: &self.emissions, transitions: &self.transitions, start: &self.start, end: &self.end }
    }
}

/// Computes one sentence's numerical scores from its tokens. The model
/// lives on the caller's side; this crate only decodes the numbers.
pub type CrfScoreFn = dyn Fn(&[&str]) -> Result<CrfSentenceScores, String> + Send + Sync;

/// A document annotator decoding the caller's numerical scores into
/// entities and their candidate alternatives.
pub struct CrfNameAnnotator {
    decoder: EntityDecoder,
    scores: Arc<CrfScoreFn>,
    k: usize,
    hash: u64,
    model_ref: ModelRef,
}

impl CrfNameAnnotator {
    /// Binds the schema, the candidate path count `k`, the score source
    /// and the explicit provenance recorded on every document. `scores_hash`
    /// identifies the numerical parameters `scores` runs on; the config
    /// hash folds it in next to the schema hash and `k`, and
    /// [`crate::calibrated::crf_identity`] uses the same triple for the
    /// calibration identity.
    pub fn new(
        decoder: EntityDecoder,
        k: usize,
        scores: impl Fn(&[&str]) -> Result<CrfSentenceScores, String> + Send + Sync + 'static,
        scores_hash: u64,
        provenance: ModelRef,
    ) -> Result<Self, String> {
        if k == 0 || k > 4096 {
            return Err("k must be in 1..=4096".into());
        }
        let hash = Fnv1a::new().u64(decoder.schema_hash()).u64(k as u64).u64(scores_hash).finish();
        Ok(Self { decoder, scores: Arc::new(scores), k, hash, model_ref: provenance })
    }

    pub fn decoder(&self) -> &EntityDecoder {
        &self.decoder
    }

    pub fn candidate_paths(&self) -> usize {
        self.k
    }
}

impl Annotator for CrfNameAnnotator {
    const ID: &'static str = "turbo-namefind:crf";
    const VERSION: &'static str = "3";
    type Reads = (View, Sentences, Tokens);
    type Writes = (Entities, Alternatives<Entities>);

    fn config_hash(&self) -> u64 {
        self.hash
    }

    fn locality(&self) -> Locality {
        Locality::Partition(Key::<Sentences>::DEFAULT.id())
    }
    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref.clone())
    }

    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        sentences.check().map_err(fail)?;
        tokens.check().map_err(fail)?;
        let ranges = sentence_token_ranges(sentences, tokens, view).map_err(fail)?;
        let (starts, ends) = tokens.spans.columns();
        let text = view.text();
        let mut entities = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: self.decoder.kinds().clone(),
            score: Vec::new(),
        };
        let mut alternatives = Alternatives::new(view.id());
        for range in &ranges {
            let words: Vec<&str> =
                (range.base..range.end).map(|i| &text[starts[i] as usize..ends[i] as usize]).collect();
            let scores = (self.scores)(&words).map_err(fail)?;
            let lattice = self.decoder.lattice(scores.as_scores()).map_err(|e| fail(e.to_string()))?;
            let paths = lattice.best_paths(self.k).map_err(|e| fail(e.to_string()))?;
            let mut candidates = BTreeMap::<(usize, usize, Tag), (f64, bool)>::new();
            let mut covered = vec![false; range.end - range.base];
            for (rank, path) in paths.iter().enumerate() {
                for entity in &path.entities {
                    let local = &entity.tokens;
                    let item = candidates
                        .entry((range.base + local.start.get(), range.base + local.end.get(), entity.kind))
                        .or_insert((entity.probability, false));
                    item.1 |= rank == 0;
                    if rank == 0 {
                        covered[local.start.get()..local.end.get()].fill(true);
                        push_entity(&mut entities, starts, ends, range.base, entity).map_err(fail)?;
                    }
                }
            }
            // A complete unit-token backbone keeps the union lattice valid,
            // including positions traversed only inside a multi-token entity.
            let mut union = BTreeMap::<(usize, usize), (f64, bool)>::new();
            for ((a, b, _), (prob, best)) in candidates {
                let entry = union.entry((a, b)).or_default();
                entry.0 += prob;
                entry.1 |= best;
            }
            for i in range.base..range.end {
                alternatives.push_full(
                    Span::new(starts[i], ends[i]),
                    i as u32,
                    1,
                    if !covered[i - range.base] { ModeSet::BEST } else { ModeSet::EMPTY },
                    Some(0.0),
                    None,
                );
                for (&(a, b), &(score, best)) in union.range((i, 0)..=(i, usize::MAX)) {
                    alternatives.push_full(
                        Span::new(starts[a], ends[b - 1]),
                        a as u32,
                        (b - a) as u32,
                        if best { ModeSet::BEST } else { ModeSet::EMPTY },
                        Some(score.min(1.0)),
                        None,
                    );
                }
            }
        }
        entities.check().map_err(fail)?;
        alternatives.check().map_err(fail)?;
        Ok((entities, alternatives))
    }
}

/// One sentence's tokens: the token range `[base, end)` in document token
/// order.
pub(crate) struct SentenceTokens {
    pub(crate) base: usize,
    pub(crate) end: usize,
}

/// Checks the sentences and tokens against the view and each other, then
/// groups the tokens by sentence in document order. Shared by the CRF
/// annotators; the error text names the shared requirements.
pub(crate) fn sentence_token_ranges(
    sentences: &Sentences,
    tokens: &Tokens,
    view: &View,
) -> Result<Vec<SentenceTokens>, String> {
    if tokens.spans.view() != view.id() || sentences.spans.view() != view.id() {
        return Err("input view mismatch".into());
    }
    check_ordered_spans(&tokens.spans, view.text())?;
    check_ordered_spans(&sentences.spans, view.text())?;
    let (starts, ends) = tokens.spans.columns();
    let mut out = Vec::new();
    let mut base = 0;
    for sentence in 0..sentences.len() {
        let span = sentences.spans.get(Ix::new(sentence as u32));
        let mut end = base;
        while end < tokens.len() && starts[end] < span.end {
            if starts[end] < span.start
                || ends[end] > span.end
                || tokens.sentence.as_ref().is_some_and(|s| s[end].get() != sentence)
            {
                return Err("token/sentence mismatch".into());
            }
            end += 1;
        }
        out.push(SentenceTokens { base, end });
        base = end;
    }
    if base != tokens.len() {
        return Err("tokens outside sentences".into());
    }
    Ok(out)
}

/// Requires ordered, nonempty, contiguous spans on UTF-8 char boundaries.
fn check_ordered_spans<L: Layer>(spans: &Spans<L>, text: &str) -> Result<(), String> {
    let mut last = 0;
    for i in 0..spans.len() {
        let ix = Ix::new(i as u32);
        let span = spans.get(ix);
        if spans.is_discontinuous(ix)
            || span.start < last
            || span.start >= span.end
            || text.get(span.start as usize..span.end as usize).is_none()
        {
            return Err("name finder needs ordered nonempty contiguous UTF-8 spans".into());
        }
        last = span.end;
    }
    Ok(())
}

/// Pushes one decoded entity onto `entities`, mapping sentence-local token
/// indices through the document's token columns.
pub(crate) fn push_entity(
    entities: &mut Entities,
    starts: &[u32],
    ends: &[u32],
    base: usize,
    entity: &ScoredEntity,
) -> Result<(), String> {
    let (a, b) = (base + entity.tokens.start.get(), base + entity.tokens.end.get());
    if a >= b || b > starts.len() {
        return Err("invalid decoded name span".into());
    }
    entities.spans.push(Span::new(starts[a], ends[b - 1]));
    entities.tokens.push(IxRange::new(a as u32, b as u32));
    entities.kind.push(entity.kind);
    entities.score.push(entity.probability);
    Ok(())
}
