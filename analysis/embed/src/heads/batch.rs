//! Shared preparation and batching for span classification heads.
use super::annotators::*;
use crate::vectors::{Input, VectorAnnotator};
use turbo_analysis::{Error, Ix, IxRange, Layer, Span, Spans, ViewId};
use turbo_analysis_layers::{
    AspectTarget, Aspects, Categories, CategoryScope, Chunks, Opinions, Sentences, Subwords, Tokens,
};

pub(super) struct Prepared<'a> {
    sub: &'a Subwords,
    sentences: Option<&'a Spans<Sentences>>,
    targets: Spans<Chunks>,
}

pub(super) fn prepare<'a>(
    sub: &'a Subwords,
    sentences: Option<&'a Sentences>,
    view: ViewId,
    spans: impl Iterator<Item = Span>,
) -> Prepared<'a> {
    let mut targets = Spans::new(view);
    for s in spans {
        targets.push(s);
    }
    Prepared { sub, sentences: sentences.map(|s| &s.spans), targets }
}

pub(super) fn batch<L: CategoryScope>(
    vectors: &VectorAnnotator,
    head: &ScoringHead,
    caller: &'static str,
    docs: Vec<Result<Prepared<'_>, Error>>,
) -> Vec<Result<(Categories<L>,), Error>> {
    let inputs: Vec<_> = docs
        .iter()
        .filter_map(|d| d.as_ref().ok())
        .map(|d| Input { subwords: d.sub, sentences: d.sentences, chunks: Some(&d.targets), levels: None })
        .collect();
    let mut outputs = vectors.embed(caller, &inputs).into_iter();
    docs.into_iter()
        .map(|doc| {
            doc?;
            let c = outputs.next().expect("one output per valid input")?.chunks.expect("targets supplied");
            let pooled = turbo_analysis_layers::Vectors::<L>::new(
                c.dim,
                c.data,
                c.empty.into_iter().map(|i| Ix::new(i.get() as u32)).collect(),
                c.model,
            );
            Ok((head.categorize_vectors(caller, &pooled)?,))
        })
        .collect()
}

fn check_range(caller: &'static str, tokens: &Tokens, range: IxRange<Tokens>) -> Result<(), Error> {
    if range.start.get() >= range.end.get() || range.end.get() > tokens.len() {
        return Err(Error::Annotator { annotator: caller, detail: "span head token range exceeds Tokens".into() });
    }
    Ok(())
}

pub(super) fn opinion_spans(caller: &'static str, tokens: &Tokens, opinions: &Opinions) -> Result<Vec<Span>, Error> {
    opinions.check().map_err(|detail| Error::Annotator { annotator: caller, detail })?;
    let mut spans = Vec::with_capacity(opinions.len());
    for i in 0..opinions.len() {
        let ix = Ix::new(i as u32);
        check_range(caller, tokens, opinions.get(ix).tokens)?;
        spans.push(opinions.span(ix, tokens));
    }
    Ok(spans)
}

pub(super) fn aspect_spans(
    caller: &'static str,
    tokens: &Tokens,
    sentences: Option<&Sentences>,
    aspects: &Aspects,
) -> Result<Vec<Span>, Error> {
    let fail = |detail| Error::Annotator { annotator: caller, detail };
    aspects.check().map_err(fail)?;
    let mut spans = Vec::with_capacity(aspects.len());
    for i in 0..aspects.len() {
        let ix = Ix::new(i as u32);
        match aspects.target(ix) {
            AspectTarget::Term { tokens: r, .. } => check_range(caller, tokens, r)?,
            AspectTarget::Implicit { sentence } => {
                if sentences.is_none_or(|s| sentence.get() >= s.len() || s.spans.view() != tokens.spans.view()) {
                    return Err(fail("implicit aspect requires a valid sentence in the token view".into()));
                }
            }
        }
        spans.push(aspects.span(ix, tokens, sentences).expect("validated aspect target"));
    }
    Ok(spans)
}
