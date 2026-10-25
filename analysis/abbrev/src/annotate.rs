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
    const VERSION: &'static str = "2";
    type Reads = (View, Sentences, Option<Tokens>);
    type Writes = (Abbreviations, Edges<Abbreviations, Abbreviations>);

    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let invalid = |detail| Error::Annotator { annotator: Self::ID, detail };
        sentences.check_contiguous_in(view).map_err(invalid)?;
        if let Some(t) = tokens {
            t.check_contiguous_in(view).map_err(invalid)?;
        }
        let bounds = tokens.map(|t| t.spans.columns());
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
    let mentions = find_mentions(text, &definitions, tokens);
    // A mention inside or partially crossing an earlier one ("IL" in "IL-2",
    // or "CD-EF" crossing "AB-CD" in "AB-CD-EF", both defined) is skipped so
    // mentions stay disjoint in text order.
    let mut reach = 0;
    for (span, long) in mentions {
        if span.start < reach {
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

/// Each mention with the item of its long form, sorted by start and then
/// by end, longest first. One pass over the text from the first
/// definition's end: a position is tried only when its first two bytes
/// start some short form, and then only against those short forms.
fn find_mentions(text: &str, definitions: &[Definition], tokens: Option<(&[u32], &[u32])>) -> Vec<(Span, usize)> {
    let mut mentions = Vec::new();
    let Some(first) = definitions.first() else { return mentions };
    let bytes = text.as_bytes();
    let short = |i: u32| &bytes[definitions[i as usize].short.range()];
    let prefix = |b: &[u8]| usize::from(b[0]) << 8 | usize::from(b[1]);
    // The definitions grouped by short form, each group in text order.
    let mut by_short: Vec<u32> = (0..definitions.len() as u32).collect();
    by_short.sort_by(|&a, &b| short(a).cmp(short(b)).then(a.cmp(&b)));
    // Each group as (its short form's first two bytes, its range in
    // `by_short`), sorted by those bytes; a short form has two characters
    // or more, so two bytes or more.
    let mut groups: Vec<(usize, usize, usize)> = Vec::new();
    let mut seen = vec![0u64; 1 << 10];
    let mut g = 0;
    while g < by_short.len() {
        let end = g + by_short[g..].partition_point(|&i| short(i) == short(by_short[g]));
        let key = prefix(short(by_short[g]));
        seen[key >> 6] |= 1 << (key & 63);
        groups.push((key, g, end));
        g = end;
    }
    groups.sort_unstable();
    // Where the group's latest mention ends, so that matches of one short
    // form do not overlap.
    let mut free = vec![0usize; groups.len()];
    for at in first.end as usize..bytes.len().saturating_sub(1) {
        let key = prefix(&bytes[at..]);
        if seen[key >> 6] & (1 << (key & 63)) == 0
            || (bytes[at] & 0xC0) == 0x80
            || text[..at].chars().next_back().is_some_and(is_letter_or_digit)
        {
            continue;
        }
        let lo = groups.partition_point(|g| g.0 < key);
        for (k, &(_, g_start, g_end)) in groups.iter().enumerate().skip(lo).take_while(|(_, g)| g.0 == key) {
            let group = &by_short[g_start..g_end];
            let len = short(group[0]).len();
            if at < free[k] || !bytes[at..].starts_with(short(group[0])) {
                continue;
            }
            // The latest definition that closes at or before `at`, and
            // the mention must end before the short form is defined again.
            let n = group.partition_point(|&i| definitions[i as usize].end as usize <= at);
            if n == 0 || group.get(n).is_some_and(|&i| (definitions[i as usize].start() as usize) < at + len) {
                continue;
            }
            let span = Span::new(at as u32, (at + len) as u32);
            if is_mention(text, span, definitions, tokens) {
                free[k] = at + len;
                mentions.push((span, 2 * group[n - 1] as usize));
            }
        }
    }
    mentions.sort_unstable_by_key(|&(s, _)| (s.start, std::cmp::Reverse(s.end)));
    mentions
}

/// Whether a match of a short form at `span` is a mention: no letter or
/// digit on either side, whole tokens when there are tokens, no overlap
/// with a defining short form, and neither equal to nor partially crossing
/// a defining long form.
fn is_mention(text: &str, span: Span, definitions: &[Definition], tokens: Option<(&[u32], &[u32])>) -> bool {
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
    // Definitions are in text order, so their short forms and their long
    // forms are too.
    let first = definitions.partition_point(|d| d.short.end <= span.start);
    if definitions.get(first).is_some_and(|d| d.short.start < span.end) {
        return false;
    }
    let first_long = definitions.partition_point(|d| d.long.end <= span.start);
    for d in definitions[first_long..].iter().take_while(|d| d.long.start < span.end) {
        if d.long == span || !(d.long.contains(span) || span.contains(d.long)) {
            return false;
        }
    }
    true
}
