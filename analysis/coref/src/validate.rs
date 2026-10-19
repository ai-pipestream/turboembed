//! The structural validation of a [`CorefInput`] before anything indexes
//! it: a `CorefInput` is public and can be built outside a `Document`, so
//! each layer's own invariants, the alignment of the part-of-speech tags,
//! every token and sentence index a column holds, and every span and
//! fragment against the text are checked here, each failure an `Err`.

use turbo_analysis::{IxRange, Layer, Spans};
use turbo_analysis_layers::{Entities, PhraseChunks, Phrases, Sentences, Tokens};

use crate::input::CorefInput;
use crate::layer::Speakers;

/// `Err` naming the layer whose own invariants fail.
fn checked<L: Layer>(layer: &L) -> Result<(), String> {
    layer.check().map_err(|e| format!("{}: {e}", L::ID))
}

/// `Err` unless every span and fragment of `spans` lies inside `text` on
/// character boundaries.
fn slices<L>(text: &str, id: &str, spans: &Spans<L>) -> Result<(), String> {
    for (i, span) in spans.iter() {
        for s in std::iter::once(span).chain(spans.fragments(i)) {
            if s.start > s.end || text.get(s.range()).is_none() {
                return Err(format!("{id} item {} [{}..{}) does not slice the text", i.get(), s.start, s.end));
            }
        }
    }
    Ok(())
}

/// `Err` unless every token range of a column ends inside the tokens.
fn within(id: &str, ranges: &[IxRange<Tokens>], tokens: usize) -> Result<(), String> {
    for (i, r) in ranges.iter().enumerate() {
        if r.start.get() > r.end.get() || r.end.get() > tokens {
            return Err(format!("{id} item {i} has tokens {}..{} past {tokens} tokens", r.start.get(), r.end.get()));
        }
    }
    Ok(())
}

/// Validates the layers of `input` so that flattening them cannot panic.
pub(crate) fn input(input: &CorefInput<'_>) -> Result<(), String> {
    checked(input.sentences)?;
    checked(input.tokens)?;
    checked(input.tags)?;
    checked(input.entities)?;
    if let Some(c) = input.chunks {
        checked(c)?;
    }
    if let Some(p) = input.phrases {
        checked(p)?;
    }
    if let Some(s) = input.speakers {
        checked(s)?;
    }
    let tokens = input.tokens.spans.len();
    let sentences = input.sentences.spans.len();
    if input.tags.tags.len() != tokens {
        return Err("document needs aligned opennlp:tokens and opennlp:pos layers".into());
    }
    if let Some(of) = &input.tokens.sentence {
        if let Some((t, s)) = of.iter().enumerate().find(|(_, s)| s.get() >= sentences) {
            return Err(format!("token {t} names sentence {} past {sentences} sentences", s.get()));
        }
    }
    within(<Entities as Layer>::ID, &input.entities.tokens, tokens)?;
    if let Some(c) = input.chunks {
        within(<PhraseChunks as Layer>::ID, &c.tokens, tokens)?;
    }
    if let Some(p) = input.phrases {
        // A phrase's head lies inside its token range (`Phrases::check`).
        within(<Phrases as Layer>::ID, &p.tokens, tokens)?;
    }
    let text = input.text;
    slices(text, <Sentences as Layer>::ID, &input.sentences.spans)?;
    slices(text, <Tokens as Layer>::ID, &input.tokens.spans)?;
    slices(text, <Entities as Layer>::ID, &input.entities.spans)?;
    if let Some(c) = input.chunks {
        slices(text, <PhraseChunks as Layer>::ID, &c.spans)?;
    }
    if let Some(p) = input.phrases {
        slices(text, <Phrases as Layer>::ID, &p.spans)?;
    }
    if let Some(s) = input.speakers {
        slices(text, <Speakers as Layer>::ID, &s.spans)?;
    }
    Ok(())
}
