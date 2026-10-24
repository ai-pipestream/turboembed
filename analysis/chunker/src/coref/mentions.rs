//! Mention layer types (`Mentions`, `MentionKind`, `MentionOrigin`, `Agreement`)
//! re-exported from [`turbo_coref`] and per-chunk antecedent header context
//! (`ChunkAntecedents`, `ChunkAntecedentBuilder`) for coreference-aware chunking.

use turbo_analysis::{Anchor, Annotator, Edges, Error, Groups, Ix, Key, KeyId, Layer, Refs, Span, View};
use turbo_analysis_layers::{Chunks, Entities, Sentences};
pub use turbo_coref::{Agreement, Animacy, Gender, MentionKind, Mentions, Number, Origin as MentionOrigin, Person};

use super::{ResolvedMention, extract_chains, is_pronoun_text};

/// One unresolved mention in a chunk paired with its canonical antecedent in
/// an earlier sentence before the chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AntecedentLink {
    /// Span of the anaphoric/pronominal mention inside the chunk.
    pub mention_span: Span,
    /// Sentence of the mention inside the chunk.
    pub mention_sentence: Ix<Sentences>,
    /// Span of the canonical antecedent in an earlier sentence (`< chunk.sentences.start`).
    pub antecedent_span: Span,
    /// Sentence holding the canonical antecedent.
    pub antecedent_sentence: Ix<Sentences>,
    /// Surface text of the mention when `View` was provided.
    pub mention_text: String,
    /// Surface text of the antecedent when `View` was provided.
    pub antecedent_text: String,
}

/// Per-chunk antecedent context (`turbo:chunk-antecedents`), aligned to [`Chunks`].
///
/// When a chunk cuts across an active coreference chain, leaving a pronominal
/// or anaphoric mention inside the chunk whose canonical antecedent lies in an
/// earlier sentence before the chunk, this layer records the resolved
/// [`AntecedentLink`]s and a formatted context header for the chunk.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkAntecedents {
    pub row: Vec<u32>,
    pub links: Vec<AntecedentLink>,
    pub headers: Vec<String>,
}

impl ChunkAntecedents {
    /// Returns the antecedent links for chunk `c`.
    pub fn links_of(&self, c: Ix<Chunks>) -> &[AntecedentLink] {
        let i = c.get();
        if i + 1 >= self.row.len() {
            return &[];
        }
        let (a, b) = (self.row[i] as usize, self.row[i + 1] as usize);
        &self.links[a..b]
    }

    /// Returns the formatted antecedent context header for chunk `c`, or
    /// `None` if chunk `c` has no out-of-chunk antecedents.
    pub fn header_of(&self, c: Ix<Chunks>) -> Option<&str> {
        let s = self.headers.get(c.get())?.as_str();
        (!s.is_empty()).then_some(s)
    }
}

impl Layer for ChunkAntecedents {
    const ID: &'static str = "turbo:chunk-antecedents";

    fn anchor() -> Anchor {
        Anchor::aligned::<Chunks>()
    }

    fn len(&self) -> usize {
        self.headers.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let bound =
            self.links.iter().map(|l| l.mention_sentence.get().max(l.antecedent_sentence.get()) + 1).max().unwrap_or(0);
        if bound > 0 { vec![(Key::<Sentences>::DEFAULT.id(), bound)] } else { Vec::new() }
    }

    fn check(&self) -> Result<(), String> {
        let n = self.headers.len();
        if self.row.len() != n + 1 {
            return Err(format!("{} row offsets for {n} chunks", self.row.len()));
        }
        if *self.row.last().unwrap_or(&0) as usize != self.links.len() {
            return Err("chunk antecedent row offsets do not end at links.len()".into());
        }
        if !self.row.windows(2).all(|w| w[0] <= w[1]) {
            return Err("chunk antecedent row offsets are not monotonic".into());
        }
        Ok(())
    }
}

/// Builds [`ChunkAntecedents`] for every chunk in `chunks`.
#[allow(clippy::too_many_arguments)]
pub fn chunk_antecedents(
    sentences: &Sentences,
    chunks: &Chunks,
    view: Option<&View>,
    mentions: Option<&Mentions>,
    mention_groups: Option<&Groups<Mentions>>,
    mention_edges: Option<&Edges<Mentions, Mentions>>,
    entities: Option<&Entities>,
    entity_groups: Option<&Groups<Entities>>,
    entity_edges: Option<&Edges<Entities, Entities>>,
) -> ChunkAntecedents {
    let chains =
        extract_chains(sentences, view, mentions, mention_groups, mention_edges, entities, entity_groups, entity_edges);
    let n_chunks = chunks.spans.len();
    let mut row = Vec::with_capacity(n_chunks + 1);
    let mut links = Vec::new();
    let mut headers = Vec::with_capacity(n_chunks);
    row.push(0);

    let slice_text = |s: Span| -> String {
        view.and_then(|v| v.text().get(s.range())).map(|t| t.trim().to_owned()).unwrap_or_default()
    };

    for c_idx in 0..n_chunks {
        let sent_range = chunks.sentences[c_idx];
        let (s_start, s_end) = (sent_range.start.get(), sent_range.end.get());
        let chunk_span = chunks.spans.get(Ix::new(c_idx as u32));
        let start_links = links.len();

        for cluster in &chains.clusters {
            // Find the canonical antecedent strictly before `s_start` (preferring non-pronoun).
            let antecedent: Option<&ResolvedMention> = cluster
                .iter()
                .find(|m| m.sentence < s_start && !m.is_pronoun)
                .or_else(|| cluster.iter().find(|m| m.sentence < s_start));
            let Some(ant) = antecedent else {
                continue;
            };
            // Check if this chain has an anaphoric/pronominal mention inside the chunk
            // (or any mention inside the chunk when no non-pronoun mention of this chain
            // has appeared earlier inside the chunk).
            let has_in_chunk_non_pronoun = cluster.iter().any(|m| {
                m.sentence >= s_start
                    && m.sentence < s_end
                    && m.span.start >= chunk_span.start
                    && m.span.end <= chunk_span.end
                    && !m.is_pronoun
            });
            if let Some(in_chunk) = cluster.iter().find(|m| {
                m.sentence >= s_start
                    && m.sentence < s_end
                    && m.span.start >= chunk_span.start
                    && m.span.end <= chunk_span.end
                    && (m.is_pronoun || !has_in_chunk_non_pronoun)
            }) {
                let mention_text = slice_text(in_chunk.span);
                let antecedent_text = slice_text(ant.span);
                if in_chunk.is_pronoun || is_pronoun_text(&mention_text) || mention_text != antecedent_text {
                    links.push(AntecedentLink {
                        mention_span: in_chunk.span,
                        mention_sentence: Ix::new(in_chunk.sentence as u32),
                        antecedent_span: ant.span,
                        antecedent_sentence: Ix::new(ant.sentence as u32),
                        mention_text,
                        antecedent_text,
                    });
                }
            }
        }

        row.push(links.len() as u32);
        if links.len() == start_links {
            headers.push(String::new());
        } else {
            let parts: Vec<String> = links[start_links..]
                .iter()
                .map(|l| {
                    if !l.antecedent_text.is_empty() && !l.mention_text.is_empty() {
                        format!("{} ({})", l.antecedent_text, l.mention_text)
                    } else if !l.antecedent_text.is_empty() {
                        l.antecedent_text.clone()
                    } else {
                        format!(
                            "{}..{} -> {}..{}",
                            l.antecedent_span.start, l.antecedent_span.end, l.mention_span.start, l.mention_span.end
                        )
                    }
                })
                .collect();
            headers.push(format!("[Context: {}]", parts.join("; ")));
        }
    }

    ChunkAntecedents { row, links, headers }
}

/// Annotator that writes [`ChunkAntecedents`] (`turbo:chunk-antecedents`).
#[derive(Copy, Clone, Debug, Default)]
pub struct ChunkAntecedentBuilder;

impl Annotator for ChunkAntecedentBuilder {
    const ID: &'static str = "turbo:chunk-antecedent-builder";
    type Reads = (
        Sentences,
        Chunks,
        Option<View>,
        Option<Mentions>,
        Option<Groups<Mentions>>,
        Option<Edges<Mentions, Mentions>>,
        Option<Entities>,
        Option<Groups<Entities>>,
    );
    type Writes = (ChunkAntecedents,);

    fn annotate(
        &self,
        (sentences, chunks, view, mentions, m_groups, m_edges, entities, e_groups): Refs<'_, Self>,
    ) -> Result<(ChunkAntecedents,), Error> {
        let v = sentences.spans.view();
        if chunks.spans.view() != v
            || view.is_some_and(|vw| vw.id() != v)
            || mentions.is_some_and(|m| m.spans.view() != v)
            || entities.is_some_and(|e| e.spans.view() != v)
        {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        Ok((chunk_antecedents(sentences, chunks, view, mentions, m_groups, m_edges, entities, e_groups, None),))
    }
}
