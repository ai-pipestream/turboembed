//! The detector as an analysis step: definitions sentence by sentence,
//! then every later use of each short form, written as
//! `turbo:abbreviations` with the links beside it.

use turbo_analysis::{Annotator, Edge, Edges, Error, Ix, Refs, Span, Tag, View, ViewId};
use turbo_analysis_layers::{AbbreviationForm, Abbreviations, Sentences, Tokens, abbreviation_links};
use turbo_unicode::java::is_letter_or_digit;

use crate::detect::{Definition, definitions_in};

/// The edge tags of [`abbreviation_links`].
const DEFINITION_TAG: Tag = Tag(0);
const MENTION_TAG: Tag = Tag(1);

/// Detects abbreviation definitions with the Schwartz and Hearst algorithm
/// and links each later use of a short form to its long form. It reads a
/// text view and its sentences, and the word tokens when the document has
/// them, and writes `turbo:abbreviations` and
/// `turbo:edges<turbo:abbreviations, turbo:abbreviations>`.
#[derive(Copy, Clone, Debug, Default)]
pub struct AbbreviationAnnotator;

impl Annotator for AbbreviationAnnotator {
    const ID: &'static str = "turbo-abbrev:schwartz-hearst";
    type Reads = (View, Sentences, Option<Tokens>);
    type Writes = (Abbreviations, Edges<Abbreviations, Abbreviations>);

    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: &str| Error::Annotator { annotator: Self::ID, detail: detail.into() };
        if sentences.spans.view() != view.id() {
            return Err(fail("the sentences are in another view"));
        }
        let bounds = match tokens {
            Some(t) if t.spans.view() != view.id() => return Err(fail("the tokens are in another view")),
            Some(t) if !t.spans.is_ordered() => return Err(fail("the tokens are not in text order")),
            Some(t) => Some(t.spans.columns()),
            None => None,
        };
        let sentence_spans = sentences.spans.iter().map(|(_, s)| s);
        Ok(detect(view.text(), view.id(), sentence_spans, bounds))
    }
}

/// The definitions in `sentences` (spans of `text`, in text order) and the
/// later mentions of their short forms, as the layer and its links.
///
/// A mention is an exact, case-sensitive occurrence of a short form after
/// its definition's closing parenthesis, with no letter or digit right
/// before or after it. It overlaps no defining short form, is not itself a
/// long form and is not inside a longer mention. When `tokens` (their start
/// and end columns, in text order) are given, it must also start where a
/// token starts and end where a token ends. It refers to the latest
/// definition of that short form before it, so a short form defined again
/// takes the new long form from there on.
pub fn detect(
    text: &str,
    view: ViewId,
    sentences: impl IntoIterator<Item = Span>,
    tokens: Option<(&[u32], &[u32])>,
) -> (Abbreviations, Edges<Abbreviations, Abbreviations>) {
    let mut definitions = Vec::new();
    for sentence in sentences {
        definitions_in(text, sentence, &mut definitions);
    }
    // Each item: its span, its form, and for a short form or mention the
    // item of its long form.
    let mut items: Vec<(Span, AbbreviationForm, Option<usize>)> = Vec::with_capacity(definitions.len() * 2);
    for d in &definitions {
        items.push((d.long, AbbreviationForm::Long, None));
        items.push((d.short, AbbreviationForm::Short, Some(items.len() - 1)));
    }
    let mut longs: Vec<Span> = definitions.iter().map(|d| d.long).collect();
    longs.sort_unstable();
    // Each mention with the item of its long form.
    let mut mentions: Vec<(Span, usize)> = Vec::new();
    for (i, d) in definitions.iter().enumerate() {
        let short = &text[d.short.range()];
        let until = definitions[i + 1..]
            .iter()
            .find(|later| text[later.short.range()] == *short)
            .map_or(text.len(), |later| later.start() as usize);
        let from = d.end as usize;
        if from >= until {
            continue;
        }
        for (at, _) in text[from..until].match_indices(short) {
            let span = Span::new((from + at) as u32, (from + at + short.len()) as u32);
            if is_mention(text, span, &definitions, &longs, tokens) {
                mentions.push((span, 2 * i));
            }
        }
    }
    // A mention inside a longer one ("IL" in "IL-2", both defined) is part
    // of that one, not a mention of its own.
    mentions.sort_unstable_by_key(|&(s, _)| (s.start, std::cmp::Reverse(s.end)));
    let mut reach = 0;
    for (span, long) in mentions {
        if span.end <= reach {
            continue;
        }
        reach = span.end;
        items.push((span, AbbreviationForm::Mention, Some(long)));
    }
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_unstable_by_key(|&i| (items[i].0.start, items[i].0.end));
    let mut position = vec![0u32; items.len()];
    let mut layer = Abbreviations::new(view);
    for (to, &from) in order.iter().enumerate() {
        position[from] = to as u32;
        layer.push(items[from].0, items[from].1);
    }
    let rows = items.iter().enumerate().filter_map(|(i, &(_, form, long))| {
        let tag = if form == AbbreviationForm::Mention { MENTION_TAG } else { DEFINITION_TAG };
        long.map(|l| Edge::new(Ix::new(position[i]), Ix::new(position[l]), tag, 1.0))
    });
    let links = Edges::new(abbreviation_links(), rows);
    (layer, links)
}

fn is_mention(
    text: &str,
    span: Span,
    definitions: &[Definition],
    longs: &[Span],
    tokens: Option<(&[u32], &[u32])>,
) -> bool {
    let (start, end) = (span.start as usize, span.end as usize);
    if text[..start].chars().next_back().is_some_and(is_letter_or_digit)
        || text[end..].chars().next().is_some_and(is_letter_or_digit)
    {
        return false;
    }
    if let Some((starts, ends)) = tokens
        && (starts.binary_search(&span.start).is_err() || ends.binary_search(&span.end).is_err())
    {
        return false;
    }
    // Definitions are in text order, so their short forms are too.
    let first = definitions.partition_point(|d| d.short.end <= span.start);
    if definitions.get(first).is_some_and(|d| d.short.start < span.end) {
        return false;
    }
    longs.binary_search(&span).is_err()
}
