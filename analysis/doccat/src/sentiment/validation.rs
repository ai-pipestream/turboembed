//! Input contracts for scorers whose evidence is a contiguous token range.
use turbo_analysis::{Error, Ix, Layer, Spans};
use turbo_analysis_layers::{Sentences, Tokens, Vectors};

pub(super) fn checked<L: Layer>(id: &'static str, layer: &L) -> Result<(), Error> {
    layer.check().map_err(|detail| Error::Annotator { annotator: id, detail })
}

pub(super) fn contiguous<L>(id: &'static str, spans: &Spans<L>, text: &str) -> Result<(), Error> {
    crate::layer::validate_spans(id, "input", spans, text)?;
    if spans.iter().any(|(ix, _)| spans.is_discontinuous(ix)) {
        return Err(Error::Annotator {
            annotator: id,
            detail: "sentiment scoring requires contiguous input spans; split discontinuous units before scoring"
                .into(),
        });
    }
    Ok(())
}

pub(super) fn tokens(
    id: &'static str,
    tokens: &Tokens,
    sentences: Option<&Sentences>,
    text: &str,
) -> Result<(), Error> {
    checked(id, tokens)?;
    contiguous(id, &tokens.spans, text)?;
    let fail = |detail: &str| Error::Annotator { annotator: id, detail: detail.into() };
    let (starts, ends) = tokens.spans.columns();
    if !starts.windows(2).all(|w| w[0] <= w[1]) || !ends.windows(2).all(|w| w[0] <= w[1]) {
        return Err(fail("tokens must be in text order"));
    }
    if let Some(sentences) = sentences {
        checked(id, sentences)?;
        contiguous(id, &sentences.spans, text)?;
        if let Some(assignments) = &tokens.sentence {
            for (i, &sentence) in assignments.iter().enumerate() {
                if sentence.get() >= sentences.len()
                    || !sentences.spans.get(sentence).contains(tokens.spans.get(Ix::new(i as u32)))
                {
                    return Err(fail("token sentence assignment does not contain its token"));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn vectors<L: Layer>(id: &'static str, vectors: &Vectors<L>, count: usize) -> Result<(), Error> {
    checked(id, vectors)?;
    if vectors.len() != count || vectors.data.iter().any(|x| !x.is_finite()) {
        return Err(Error::Annotator { annotator: id, detail: "vectors must have one finite row per target".into() });
    }
    Ok(())
}
