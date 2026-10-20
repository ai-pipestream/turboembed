//! Mention layer types (`Mentions`, `MentionKind`, `MentionOrigin`, `Agreement`)
//! and per-chunk antecedent header context (`ChunkAntecedents`,
//! `ChunkAntecedentBuilder`) for coreference-aware chunking.

use turbo_analysis::{
    Anchor, Annotator, Document, Edges, Error, Groups, Ix, IxRange, Key, KeyId, Layer, Overlap, Refs, Span, Spans,
    TextSpans, View,
};
use turbo_analysis_layers::{Chunks, Entities, Sentences, Tokens};

use super::{ResolvedMention, extract_chains, is_pronoun_text};

/// Grammatical number (`Mention.Number`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Number {
    Singular,
    Plural,
    Unknown,
}

/// Grammatical gender (`Mention.Gender`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Gender {
    Male,
    Female,
    Neutral,
    Unknown,
}

/// Grammatical animacy (`Mention.Animacy`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Animacy {
    Animate,
    Inanimate,
    Unknown,
}

/// Grammatical person (`Mention.Person`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Person {
    First,
    Second,
    Third,
    Unknown,
}

/// What a mention is by its syntactic form.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MentionKind {
    Proper,
    Nominal,
    Pronominal,
    Demonstrative,
}

/// Where a mention came from (`entity`, `nominal`, `pronoun`, `gold`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MentionOrigin {
    Entity,
    NounPhrase,
    Pronoun,
    Gold,
}

/// Packed agreement features (number, gender, animacy, person in 2 bits each).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Agreement(u8);

impl Agreement {
    pub fn new(number: Number, gender: Gender, animacy: Animacy, person: Person) -> Agreement {
        Agreement((number as u8) | ((gender as u8) << 2) | ((animacy as u8) << 4) | ((person as u8) << 6))
    }

    pub fn unknown() -> Agreement {
        Agreement::new(Number::Unknown, Gender::Unknown, Animacy::Unknown, Person::Unknown)
    }

    pub fn bits(self) -> u8 {
        self.0
    }
}

/// Coreference mentions (`opennlp:mentions`), one row per mention.
///
/// Chains over mentions are `Groups<Mentions>` (`turbo:groups<opennlp:mentions>`)
/// and resolved antecedent links are `Edges<Mentions, Mentions>`
/// (`turbo:edges<opennlp:mentions, opennlp:mentions>`).
#[derive(Clone, Debug)]
pub struct Mentions {
    pub spans: Spans<Mentions>,
    pub sentence: Vec<Ix<Sentences>>,
    pub tokens: Vec<IxRange<Tokens>>,
    pub head: Vec<Ix<Tokens>>,
    pub entity: Option<Vec<Option<Ix<Entities>>>>,
    pub kind: Vec<MentionKind>,
    pub origin: Vec<MentionOrigin>,
    pub agreement: Vec<Agreement>,
}

impl Mentions {
    /// Creates an empty `Mentions` layer in `view`.
    pub fn new(view: turbo_analysis::ViewId) -> Self {
        Self {
            spans: Spans::new(view),
            sentence: Vec::new(),
            tokens: Vec::new(),
            head: Vec::new(),
            entity: None,
            kind: Vec::new(),
            origin: Vec::new(),
            agreement: Vec::new(),
        }
    }

    /// Pushes one mention with default `Agreement::unknown()`.
    pub fn push(
        &mut self,
        span: Span,
        sentence: Ix<Sentences>,
        tokens: IxRange<Tokens>,
        head: Ix<Tokens>,
        kind: MentionKind,
    ) -> Ix<Mentions> {
        let origin = match kind {
            MentionKind::Proper => MentionOrigin::Entity,
            MentionKind::Nominal => MentionOrigin::NounPhrase,
            MentionKind::Pronominal | MentionKind::Demonstrative => MentionOrigin::Pronoun,
        };
        self.sentence.push(sentence);
        self.tokens.push(tokens);
        self.head.push(head);
        self.kind.push(kind);
        self.origin.push(origin);
        self.agreement.push(Agreement::unknown());
        self.spans.push(span)
    }

    /// The span of mention `i` in source bytes, through `doc`'s views.
    pub fn source_span(&self, doc: &Document, i: Ix<Mentions>) -> Span {
        doc.to_source_bytes(self.spans.view(), self.spans.get(i))
    }
}

impl Layer for Mentions {
    const ID: &'static str = "opennlp:mentions";
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let tokens =
            self.tokens.iter().map(|r| r.end.get()).chain(self.head.iter().map(|h| h.get() + 1)).max().unwrap_or(0);
        let sentences = self.sentence.iter().map(|s| s.get() + 1).max().unwrap_or(0);
        let mut out = vec![(Key::<Tokens>::DEFAULT.id(), tokens), (Key::<Sentences>::DEFAULT.id(), sentences)];
        if let Some(entity) = &self.entity {
            let bound = entity.iter().flatten().map(|e| e.get() + 1).max().unwrap_or(0);
            out.push((Key::<Entities>::DEFAULT.id(), bound));
        }
        out
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.sentence.len() != n
            || self.tokens.len() != n
            || self.head.len() != n
            || self.kind.len() != n
            || self.origin.len() != n
            || self.agreement.len() != n
        {
            return Err("a sentence, tokens, a head, a kind, an origin and an agreement for every mention".into());
        }
        if self.entity.as_ref().is_some_and(|e| e.len() != n) {
            return Err("an entity slot for every mention".into());
        }
        for i in 0..n {
            let r = self.tokens[i];
            if r.start >= r.end {
                return Err(format!("mention {i} has no token"));
            }
            if self.head[i] < r.start || self.head[i] >= r.end {
                return Err(format!("mention {i} has head token outside its token range"));
            }
        }
        self.spans.check(Self::OVERLAP)
    }
}

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
