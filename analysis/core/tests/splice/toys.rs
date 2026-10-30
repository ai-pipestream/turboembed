//! The toy annotators of the splice tests: a sentence splitter and a
//! paragraph structurer (document-wide), sentence-local tokens, tags,
//! lemmas, dependencies and names, one section-local tokenizer, a
//! document-wide count, and a step whose local declaration the pipeline
//! cannot bear.

use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_layers::{
    Block, Dependencies, Entities, Language, Lemmas, PosTags, Sentences, Strs, Structure, Tag, TagSet, Tokens,
};

/// Sentences end at a full stop; a trailing stretch without one is a
/// sentence too.
pub struct Sentencer;

impl Annotator for Sentencer {
    const ID: &'static str = "test:sentences";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut start = None;
        for (i, &c) in text.as_bytes().iter().enumerate() {
            if start.is_none() && !c.is_ascii_whitespace() {
                start = Some(i);
            }
            if c == b'.'
                && let Some(s) = start.take()
            {
                spans.push(Span::new(s as u32, i as u32 + 1));
            }
        }
        if let Some(s) = start {
            spans.push(Span::new(s as u32, text.len() as u32));
        }
        Ok((Sentences { spans },))
    }
}

/// One section per paragraph.
pub struct Structurer;

impl Annotator for Structurer {
    const ID: &'static str = "test:structure";
    type Reads = (View,);
    type Writes = (Structure,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Structure,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut at = 0usize;
        for para in text.split("\n\n") {
            spans.push(Span::new(at as u32, (at + para.len()) as u32));
            at += para.len() + 2;
        }
        let n = spans.len();
        Ok((Structure { spans, kind: vec![Block::Section; n], parent: vec![None; n], depth: vec![1; n] },))
    }
}

/// Runs of non-space bytes inside each item of the partition layer.
pub fn words_in<P: Layer>(view: &View, parts: &Spans<P>, sentence_column: bool) -> Tokens {
    let text = view.text();
    let mut spans = Spans::new(view.id());
    let mut sentence = Vec::new();
    for (p, span) in parts.iter() {
        let mut at = span.start as usize;
        for word in text[span.range()].split(' ') {
            if !word.is_empty() {
                spans.push(Span::new(at as u32, (at + word.len()) as u32));
                sentence.push(Ix::<Sentences>::new(p.get() as u32));
            }
            at += word.len() + 1;
        }
    }
    Tokens { spans, sentence: sentence_column.then_some(sentence) }
}

pub struct Tokenizer;

impl Annotator for Tokenizer {
    const ID: &'static str = "test:tokens";
    type Reads = (View, Sentences);
    type Writes = (Tokens,);
    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        Ok((words_in(view, &sentences.spans, true),))
    }
    fn locality(&self) -> Locality {
        Locality::Partition(Key::<Sentences>::DEFAULT.id())
    }
}

/// Tokens per section, as a second `Tokens` instance.
pub struct SectionTokens;

impl Annotator for SectionTokens {
    const ID: &'static str = "test:section-tokens";
    type Reads = (View, Structure);
    type Writes = (Tokens,);
    fn annotate(&self, (view, structure): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        Ok((words_in(view, &structure.spans, false),))
    }
    fn locality(&self) -> Locality {
        Locality::Partition(Key::<Structure>::DEFAULT.id())
    }
}

pub fn tagset() -> Arc<TagSet> {
    TagSet::new("toy", &["NNP", "VBZ", "X"])
}

/// A capitalized word is NNP, a word ending in "s" VBZ, the rest X. `work`
/// rounds of hashing per token stand in for a model's cost in the timing.
pub struct Tagger {
    pub tagset: Arc<TagSet>,
    /// Rounds of hashing per token, to give the step a cost; part of
    /// the configuration.
    pub work: u32,
    pub cacheable: bool,
}

impl Annotator for Tagger {
    const ID: &'static str = "test:tagger";
    type Reads = (View, Tokens);
    type Writes = (PosTags,);
    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(PosTags,), Error> {
        let text = view.text();
        let (mut tags, mut prob) = (Vec::new(), Vec::new());
        for (_, s) in tokens.spans.iter() {
            let w = &text[s.range()];
            let t = if w.starts_with(|c: char| c.is_uppercase()) {
                0
            } else if w.trim_end_matches('.').ends_with('s') {
                1
            } else {
                2
            };
            let mut h = w.len() as u64;
            for _ in 0..self.work {
                h = fnv1a(&h.to_le_bytes());
            }
            tags.push(Tag(t));
            prob.push((h % 10) as f64 / 10.0);
        }
        Ok((PosTags { tagset: self.tagset.clone(), tags, prob: Some(prob) },))
    }
    fn config_hash(&self) -> u64 {
        u64::from(self.work)
    }
    fn is_cacheable(&self) -> bool {
        self.cacheable
    }
    fn locality(&self) -> Locality {
        Locality::Partition(Key::<Sentences>::DEFAULT.id())
    }
}

/// Lowercase, a VBZ without its final "s".
pub struct Lemmatizer;

impl Annotator for Lemmatizer {
    const ID: &'static str = "test:lemmas";
    type Reads = (View, Tokens, PosTags);
    type Writes = (Lemmas,);
    fn annotate(&self, (view, tokens, tags): Refs<'_, Self>) -> Result<(Lemmas,), Error> {
        let text = view.text();
        let mut lemma = Strs::default();
        for ((_, s), &t) in tokens.spans.iter().zip(&tags.tags) {
            let w = text[s.range()].trim_end_matches('.').to_lowercase();
            lemma.push(if t == Tag(1) { w.trim_end_matches('s') } else { &w });
        }
        Ok((Lemmas { lemma },))
    }
    fn locality(&self) -> Locality {
        Locality::Partition(Key::<Sentences>::DEFAULT.id())
    }
}

/// Each token's head is the token before it in its sentence.
pub struct Parser {
    pub relations: Arc<TagSet>,
}

impl Annotator for Parser {
    const ID: &'static str = "test:parser";
    type Reads = (Tokens,);
    type Writes = (Dependencies,);
    fn annotate(&self, (tokens,): Refs<'_, Self>) -> Result<(Dependencies,), Error> {
        let mut head = vec![None; tokens.len()];
        let mut relation = vec![Tag(0); tokens.len()];
        for (_, range) in tokens.sentence_ranges() {
            for i in range.start.get() + 1..range.end.get() {
                head[i] = Some(Ix::new(i as u32 - 1));
                relation[i] = Tag(1);
            }
        }
        Ok((Dependencies { head, relation, relations: self.relations.clone() },))
    }
    fn locality(&self) -> Locality {
        Locality::Partition(Key::<Sentences>::DEFAULT.id())
    }
}

/// Runs of capitalized tokens after the first token of a sentence.
pub struct NameFinder {
    pub kinds: Arc<TagSet>,
}

impl Annotator for NameFinder {
    const ID: &'static str = "test:names";
    type Reads = (View, Sentences, Tokens);
    type Writes = (Entities,);
    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<(Entities,), Error> {
        let text = view.text();
        let mut out = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: self.kinds.clone(),
            score: Vec::new(),
        };
        for (s, _) in sentences.spans.iter() {
            let range = tokens.of(s);
            let mut run: Option<(usize, usize)> = None;
            let flush = |run: &mut Option<(usize, usize)>, out: &mut Entities| {
                if let Some((a, b)) = run.take() {
                    let (start, end) =
                        (tokens.spans.get(Ix::new(a as u32)).start, tokens.spans.get(Ix::new(b as u32 - 1)).end);
                    out.spans.push(Span::new(start, end));
                    out.tokens.push(IxRange::new(a as u32, b as u32));
                    out.kind.push(Tag(0));
                    out.score.push(0.5 + 0.1 * (b - a) as f64);
                }
            };
            for i in range.start.get() + 1..range.end.get() {
                let w = &text[tokens.spans.get(Ix::new(i as u32)).range()];
                if w.starts_with(|c: char| c.is_uppercase()) {
                    run = Some(run.map_or((i, i + 1), |(a, _)| (a, i + 1)));
                } else {
                    flush(&mut run, &mut out);
                }
            }
            flush(&mut run, &mut out);
        }
        Ok((out,))
    }
    fn locality(&self) -> Locality {
        Locality::Partition(Key::<Sentences>::DEFAULT.id())
    }
}

/// Document-wide: the token count as a language score.
pub struct Counter;

impl Annotator for Counter {
    const ID: &'static str = "test:language";
    type Reads = (Tokens,);
    type Writes = (Language,);
    fn annotate(&self, (tokens,): Refs<'_, Self>) -> Result<(Language,), Error> {
        let mut code = Strs::default();
        code.push("en");
        Ok((Language { code, score: vec![tokens.len() as f64] },))
    }
}

/// Declares itself sentence-local but reads the document-wide count.
pub struct Scaled {
    pub tagset: Arc<TagSet>,
}

impl Annotator for Scaled {
    const ID: &'static str = "test:scaled";
    type Reads = (Tokens, Language);
    type Writes = (PosTags,);
    fn annotate(&self, (tokens, language): Refs<'_, Self>) -> Result<(PosTags,), Error> {
        let n = language.score[0];
        let tags = (0..tokens.len()).map(|i| Tag(if (i as f64) < n / 2.0 { 0 } else { 1 })).collect();
        Ok((PosTags { tagset: self.tagset.clone(), tags, prob: None },))
    }
    fn locality(&self) -> Locality {
        Locality::Partition(Key::<Sentences>::DEFAULT.id())
    }
}

pub fn analyzer() -> Analyzer {
    analyzer_with(0)
}

/// The pipeline with a tagger of `work` rounds.
pub fn analyzer_with(work: u32) -> Analyzer {
    analyzer_tagging(Tagger { tagset: tagset(), work, cacheable: true })
}

/// The pipeline with a tagger whose output may not be reused.
pub fn analyzer_uncached() -> Analyzer {
    analyzer_tagging(Tagger { tagset: tagset(), work: 0, cacheable: false })
}

fn analyzer_tagging(tagger: Tagger) -> Analyzer {
    Analyzer::builder()
        .add(Sentencer)
        .add(Structurer)
        .add(Tokenizer)
        .step(Bound::new(SectionTokens).write(0, "section"))
        .add(tagger)
        .add(Lemmatizer)
        .add(Parser { relations: TagSet::new("deps", &["root", "dep"]) })
        .add(NameFinder { kinds: TagSet::new("kinds", &["NAME"]) })
        .add(Counter)
        .step(Bound::new(Scaled { tagset: TagSet::new("halves", &["first", "second"]) }).write(0, "scaled"))
        .build()
        .unwrap()
}
