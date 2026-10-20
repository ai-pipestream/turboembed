//! Multi-representation chunk indexing: several indexable
//! views of one chunk (its surface text, a header or summary,
//! its keyphrases, its sentence sub-spans), each carrying
//! provenance back to the canonical chunk unit and its
//! content ID, so a hit on any representation resolves to the
//! chunk a reader should actually receive. Parent expansion
//! of the resulting hits lives in [`crate::expand`].

use std::collections::BTreeMap;

use turbo_analysis::{Ix, Layer, Span};
use turbo_analysis_layers::{Level, Levels, Sentences, Unit};

use crate::hashes::{ChunkId, ContentHashes};
use crate::terms::{TermStats, Vocabulary};

/// What one indexed representation of a chunk is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RepresentationKind {
    /// The chunk's own span text.
    Surface,
    /// A caller-supplied summary of the chunk.
    Summary,
    /// A contextual header (a heading path, an antecedent
    /// header, a generated title) for the chunk.
    Header,
    /// The chunk's keyphrases, as one indexable text.
    Keyphrases,
    /// A fine-grained sub-span of the chunk (a sentence or
    /// clause inside it), mapped back to the chunk.
    ChildSpan,
}

/// One indexed representation and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkRepresentation {
    pub kind: RepresentationKind,
    /// The canonical unit this representation stands for.
    pub chunk: Ix<Levels>,
    /// The chunk's content ID, when the index was built with
    /// content hashes.
    pub chunk_id: Option<ChunkId>,
    /// The span this representation's text covers: the chunk's
    /// own span, or the sub-span for `ChildSpan`.
    pub span: Span,
    /// The text a retriever indexes for this representation.
    pub text: String,
}

/// One scored representation hit from [`MultiRepChunkIndex::rank`].
#[derive(Clone, Debug, PartialEq)]
pub struct RepHit {
    /// Index into [`MultiRepChunkIndex::entries`].
    pub rep: usize,
    pub kind: RepresentationKind,
    /// The canonical chunk the representation resolves to.
    pub chunk: Ix<Levels>,
    pub chunk_id: Option<ChunkId>,
    pub score: f64,
}

/// An index of chunk representations with provenance. Entries
/// keep registration order; every lookup and ranking is
/// deterministic.
#[derive(Clone, Debug, Default)]
pub struct MultiRepChunkIndex {
    entries: Vec<ChunkRepresentation>,
}

impl MultiRepChunkIndex {
    pub fn new() -> MultiRepChunkIndex {
        MultiRepChunkIndex { entries: Vec::new() }
    }

    /// Registers one representation and returns its entry
    /// index. Blank text is refused: an entry that indexes
    /// nothing is a wiring bug, not a representation.
    pub fn add(
        &mut self,
        kind: RepresentationKind,
        chunk: Ix<Levels>,
        chunk_id: Option<ChunkId>,
        span: Span,
        text: impl Into<String>,
    ) -> Result<usize, String> {
        let text = text.into();
        if span.start > span.end {
            return Err(format!(
                "representation {kind:?} for chunk {} has a reversed span {}..{}",
                chunk.get(),
                span.start,
                span.end
            ));
        }
        if text.trim().is_empty() {
            return Err(format!("representation {kind:?} for chunk {} has no text", chunk.get()));
        }
        self.entries.push(ChunkRepresentation { kind, chunk, chunk_id, span, text });
        Ok(self.entries.len() - 1)
    }

    pub fn entries(&self) -> &[ChunkRepresentation] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry indexes standing for `chunk`, in registration
    /// order.
    pub fn of(&self, chunk: Ix<Levels>) -> impl Iterator<Item = usize> + '_ {
        self.entries.iter().enumerate().filter_map(move |(i, e)| (e.chunk == chunk).then_some(i))
    }

    /// The entry indexes of one kind, in registration order.
    pub fn by_kind(&self, kind: RepresentationKind) -> impl Iterator<Item = usize> + '_ {
        self.entries.iter().enumerate().filter_map(move |(i, e)| (e.kind == kind).then_some(i))
    }

    /// Where entry `rep` resolves: its canonical chunk and
    /// content ID.
    pub fn resolve(&self, rep: usize) -> Option<(Ix<Levels>, Option<ChunkId>)> {
        self.entries.get(rep).map(|e| (e.chunk, e.chunk_id))
    }

    /// One `Surface` representation per chunk unit of `levels`,
    /// its text sliced from `text` and its content ID from
    /// `hashes` when given.
    pub fn surface(
        text: &str,
        levels: &Levels,
        hashes: Option<&ContentHashes<Levels>>,
    ) -> Result<MultiRepChunkIndex, String> {
        levels.check()?;
        check_hashes(levels, hashes)?;
        let mut index = MultiRepChunkIndex::new();
        for unit in levels.of(Level::Chunk) {
            let span = levels.spans.get(unit);
            // The chunk's text is its fragments' text, never
            // the bounding envelope's: a discontinuous chunk
            // must not index the gap it excludes. Fragments
            // join with a space so tokens across a gap stay
            // separate words.
            let mut surface = String::new();
            for fragment in levels.spans.fragments(unit) {
                if !surface.is_empty() {
                    surface.push(' ');
                }
                surface.push_str(slice(text, fragment, unit)?);
            }
            index.add(RepresentationKind::Surface, unit, id_of(hashes, unit), span, surface)?;
        }
        Ok(index)
    }

    /// Adds one `ChildSpan` representation per sentence inside
    /// each chunk unit: the small end of small-to-big
    /// retrieval, each sentence resolving to its chunk. A
    /// sentence crossing a chunk boundary belongs wholly to
    /// neither side and gets no representation.
    pub fn add_sentence_spans(
        &mut self,
        text: &str,
        levels: &Levels,
        sentences: &Sentences,
        hashes: Option<&ContentHashes<Levels>>,
    ) -> Result<usize, String> {
        levels.check()?;
        sentences.check()?;
        check_hashes(levels, hashes)?;
        if sentences.spans.view() != levels.view() {
            return Err("sentences and levels are in different views".into());
        }
        let (starts, ends) = sentences.spans.columns();
        let mut added = 0;
        for unit in levels.of(Level::Chunk) {
            let span = levels.spans.get(unit);
            let lo = ends.partition_point(|&e| e <= span.start);
            let hi = starts.partition_point(|&s| s < span.end).max(lo);
            for s in lo..hi {
                let child = Span::new(starts[s], ends[s]);
                // Contained in one fragment of the chunk, not
                // merely inside its envelope: a sentence in a
                // discontinuous chunk's gap is not its child.
                // Whitespace-only slices index nothing and are
                // skipped rather than aborting the build.
                if child.start < child.end && levels.spans.fragments(unit).any(|f| f.contains(child)) {
                    let surface = slice(text, child, unit)?;
                    if surface.trim().is_empty() {
                        continue;
                    }
                    self.add(RepresentationKind::ChildSpan, unit, id_of(hashes, unit), child, surface)?;
                    added += 1;
                }
            }
        }
        Ok(added)
    }

    /// Adds one `Keyphrases` representation per chunk unit: its
    /// `top_n` terms by tf-idf over `stats`, joined in score
    /// order. Units with no terms get no representation.
    pub fn add_keyphrases(
        &mut self,
        levels: &Levels,
        stats: &TermStats<Levels>,
        vocabulary: &Vocabulary,
        hashes: Option<&ContentHashes<Levels>>,
        top_n: usize,
    ) -> Result<usize, String> {
        levels.check()?;
        stats.check()?;
        vocabulary.check()?;
        check_hashes(levels, hashes)?;
        if stats.len() != levels.len() {
            return Err(format!("term stats for {} units over {} levels", stats.len(), levels.len()));
        }
        if stats.df.len() != vocabulary.len() {
            return Err(format!("{} df counts for a vocabulary of {}", stats.df.len(), vocabulary.len()));
        }
        if top_n == 0 {
            return Err("keyphrases need at least one term".into());
        }
        let units = levels.len() as f64;
        let mut added = 0;
        for unit in levels.of(Level::Chunk) {
            let i = unit.get();
            let (lo, hi) = (stats.row[i] as usize, stats.row[i + 1] as usize);
            let mut scored: Vec<(f64, &str)> = (lo..hi)
                .map(|at| {
                    let term = stats.term[at];
                    let idf = ((units + 1.0) / (f64::from(stats.df[term.get()]) + 1.0)).ln() + 1.0;
                    (f64::from(stats.tf[at]) * idf, vocabulary.text.get(term.get()))
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
            scored.truncate(top_n);
            if scored.is_empty() {
                continue;
            }
            let text = scored.iter().map(|(_, term)| *term).collect::<Vec<_>>().join(" ");
            self.add(RepresentationKind::Keyphrases, unit, id_of(hashes, unit), levels.spans.get(unit), text)?;
            added += 1;
        }
        Ok(added)
    }

    /// Ranks the representations against `query` terms with a
    /// deterministic tf-idf sum over the representations
    /// themselves as the collection: score desc, then entry
    /// index asc. Representations scoring zero are omitted.
    pub fn rank(&self, query: &[&str]) -> Result<Vec<RepHit>, String> {
        let terms: Vec<String> = query.iter().flat_map(|q| tokens(q)).collect();
        if terms.is_empty() {
            return Err("a query with no terms ranks nothing".into());
        }
        let docs: Vec<BTreeMap<String, usize>> = self
            .entries
            .iter()
            .map(|e| {
                let mut counts = BTreeMap::new();
                for token in tokens(&e.text) {
                    *counts.entry(token).or_insert(0usize) += 1;
                }
                counts
            })
            .collect();
        let n = docs.len() as f64;
        let mut idf: BTreeMap<&str, f64> = BTreeMap::new();
        for term in &terms {
            idf.entry(term.as_str()).or_insert_with(|| {
                let df = docs.iter().filter(|d| d.contains_key(term)).count() as f64;
                ((n + 1.0) / (df + 1.0)).ln() + 1.0
            });
        }
        let mut hits = Vec::new();
        for (rep, (entry, counts)) in self.entries.iter().zip(&docs).enumerate() {
            let mut score = 0.0f64;
            for term in &terms {
                if let Some(&tf) = counts.get(term) {
                    score += (tf as f64 / (tf as f64 + 1.0)) * idf[term.as_str()];
                }
            }
            if score > 0.0 {
                if !score.is_finite() {
                    return Err("ranking produced a non-finite score".into());
                }
                hits.push(RepHit { rep, kind: entry.kind, chunk: entry.chunk, chunk_id: entry.chunk_id, score });
            }
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.rep.cmp(&b.rep)));
        Ok(hits)
    }
}

fn check_hashes(levels: &Levels, hashes: Option<&ContentHashes<Levels>>) -> Result<(), String> {
    if let Some(hashes) = hashes {
        hashes.check()?;
        if hashes.len() != levels.len() {
            return Err(format!("content hashes for {} units over {} levels", hashes.len(), levels.len()));
        }
    }
    Ok(())
}

fn id_of(hashes: Option<&ContentHashes<Levels>>, unit: Ix<Levels>) -> Option<ChunkId> {
    hashes.map(|h| h.get(unit))
}

fn slice(text: &str, span: Span, unit: Ix<Levels>) -> Result<&str, String> {
    text.get(span.range())
        .ok_or_else(|| format!("unit {} span {}..{} does not slice the text", unit.get(), span.start, span.end))
}

/// Query and representation tokens: alphanumeric runs, folded.
/// Deliberately the same fold on both sides of a ranking.
fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            current.extend(c.to_lowercase());
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// The chunk unit a `Levels` unit names, for callers mapping
/// between the level tree and the `Chunks` layer.
pub fn chunk_of(levels: &Levels, unit: Ix<Levels>) -> Option<Ix<turbo_analysis_layers::Chunks>> {
    match levels.unit.get(unit.get()) {
        Some(Unit::Chunk(c)) => Some(*c),
        _ => None,
    }
}
