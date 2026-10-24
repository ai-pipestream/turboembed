//! The spelling-correcting normalizer, a port of OpenNLP's
//! `SpellCheckingCharSequenceNormalizer` that writes its corrections
//! through a `ViewBuilder`, so the corrected view aligns exactly with the
//! text it was read from.

use std::ops::Range;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Refs, Span, View, ViewBuilder};
use turbo_normalize_layers::{Corrections, Placeholders};
use turbo_unicode::java;

use super::casing;
use super::compound::{Compound, tokens};
use super::guards::{core, is_number_like, is_url_like};
use super::suggestion::Verbosity;
use super::symspell::SymSpell;
use crate::{Normalizer, hash_config};

/// How [`SpellCorrect`] corrects.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum SpellCorrectMode {
    /// Each whitespace-delimited token on its own; the whitespace is kept.
    #[default]
    PerToken,
    /// The whole text as a phrase, which also splits words run together
    /// and merges words split apart; whitespace runs become single spaces
    /// and the ends are trimmed.
    Compound,
}

/// Corrects spelling with a [`SymSpell`] engine (id
/// `turbo:normalize:spell-correct`).
///
/// In per-token mode the text is split at Unicode White_Space; each token
/// loses its leading and trailing code points that are neither letters nor
/// numbers, and the core left is corrected when it passes the guards: at
/// least `min_token_length` code points, not a number (`skip_numbers`),
/// not URL- or e-mail-like (`skip_urls`), at least one letter. The lookup
/// is on the core lowercased (`String.toLowerCase(Locale.ROOT)`); a core
/// the dictionary knows stays as it is, and a correction takes the
/// capitalization of the core (all uppercase, or a capital first code
/// point). Each corrected core is one replacement in the view.
///
/// In compound mode the text goes to `SymSpell::lookup_compound`. Each
/// output word that differs from the input it stands for is one
/// replacement of that input (one token, or two tokens and the whitespace
/// between them for a merge); each whitespace run between output words
/// that is not a single space is replaced by one; whitespace at the ends
/// is deleted. A text that is empty, blank as Java's `String.isBlank`
/// defines it, or whose correction is empty, is left as it is.
///
/// As an `Annotator` it reads a `View` and, when present, the
/// `Placeholders` of that view, and writes the corrected `View` and a
/// `Corrections` layer with a row for each replaced stretch of a word.
#[derive(Clone, Debug)]
pub struct SpellCorrect {
    engine: Arc<SymSpell>,
    mode: SpellCorrectMode,
    min_token_length: usize,
    max_edit_distance: usize,
    skip_numbers: bool,
    skip_urls: bool,
    keep_unknown_proper_nouns: bool,
    use_placeholders: bool,
}

/// OpenNLP's default minimum token length.
pub const DEFAULT_MIN_TOKEN_LENGTH: usize = 4;

impl SpellCorrect {
    /// Per-token correction with OpenNLP's defaults: tokens of at least 4
    /// code points, edit distance 2 (or the engine's maximum when that is
    /// smaller), numbers and URLs skipped.
    pub fn new(engine: Arc<SymSpell>) -> SpellCorrect {
        let max_edit_distance = 2.min(engine.max_edit_distance());
        SpellCorrect {
            engine,
            mode: SpellCorrectMode::PerToken,
            min_token_length: DEFAULT_MIN_TOKEN_LENGTH,
            max_edit_distance,
            skip_numbers: true,
            skip_urls: true,
            keep_unknown_proper_nouns: false,
            use_placeholders: false,
        }
    }

    pub fn mode(mut self, mode: SpellCorrectMode) -> SpellCorrect {
        self.mode = mode;
        self
    }

    /// Tokens shorter than `n` code points are left alone.
    ///
    /// Panics if `n` is 0.
    pub fn min_token_length(mut self, n: usize) -> SpellCorrect {
        assert!(n >= 1, "minTokenLength must be >= 1: {n}");
        self.min_token_length = n;
        self
    }

    /// The largest edit distance a correction may have, lowered to the
    /// engine's maximum.
    pub fn max_edit_distance(mut self, n: usize) -> SpellCorrect {
        self.max_edit_distance = n.min(self.engine.max_edit_distance());
        self
    }

    pub fn skip_numbers(mut self, on: bool) -> SpellCorrect {
        self.skip_numbers = on;
        self
    }

    pub fn skip_urls(mut self, on: bool) -> SpellCorrect {
        self.skip_urls = on;
        self
    }

    /// In per-token mode, leaves a token that starts with an uppercase
    /// letter alone unless a correction exists at edit distance 1, so names
    /// the dictionary does not know are not rewritten into common words
    /// two edits away. Off by default; OpenNLP has no such option.
    pub fn keep_unknown_proper_nouns(mut self, on: bool) -> SpellCorrect {
        self.keep_unknown_proper_nouns = on;
        self
    }

    /// As an `Annotator`, skips every token that overlaps a placeholder of
    /// the view read (a URL, an e-mail address, a hashtag, an asset)
    /// instead of guessing URLs from prefixes and top-level domains; the
    /// URL guard is off while placeholders are present. In compound mode
    /// such a token is kept as it is and the phrases between them are
    /// corrected apart. Without a `Placeholders` layer, and in the
    /// `Normalizer` form, OpenNLP's guards apply. Off by default.
    pub fn use_placeholders(mut self, on: bool) -> SpellCorrect {
        self.use_placeholders = on;
        self
    }

    pub fn engine(&self) -> &Arc<SymSpell> {
        &self.engine
    }

    /// Corrects `text` into `out`, skipping tokens that overlap `skip`
    /// (placeholder spans in text order) and reporting each replaced
    /// stretch of a word with its replacement and edit distance.
    fn run(&self, text: &str, out: &mut ViewBuilder<'_>, skip: Option<&[Span]>, report: &mut Report<'_>) {
        match self.mode {
            SpellCorrectMode::PerToken => self.per_token(text, out, skip, report),
            SpellCorrectMode::Compound => self.compound(text, out, skip, report),
        }
    }

    fn per_token(&self, text: &str, out: &mut ViewBuilder<'_>, skip: Option<&[Span]>, report: &mut Report<'_>) {
        let mut skip = Skip::new(skip);
        let mut lower = String::new();
        let mut corrected = String::new();
        for token in tokens(text) {
            if skip.overlaps(&token) {
                continue;
            }
            let (s, e) = core(&text[token.clone()]);
            let core = token.start + s..token.start + e;
            let Some(distance) = self.correct(&text[core.clone()], skip.active(), &mut lower, &mut corrected) else {
                continue;
            };
            out.keep(core.start);
            out.replace(core.end, &corrected);
            report(span(core), &corrected, distance);
        }
    }

    /// The correction of one core into `corrected`, with its edit
    /// distance, or `None` when the core stays as it is.
    fn correct(&self, core: &str, placeholders: bool, lower: &mut String, corrected: &mut String) -> Option<usize> {
        if !self.correctable(core, placeholders) {
            return None;
        }
        lower.clear();
        java::to_lowercase(core, lower);
        let found = self.engine.lookup_within(lower, Verbosity::Top, self.max_edit_distance);
        let best = found.first()?;
        if best.edit_distance == 0 {
            return None;
        }
        if self.keep_unknown_proper_nouns && best.edit_distance > 1 && core.chars().next().is_some_and(casing::is_upper)
        {
            return None;
        }
        corrected.clear();
        casing::apply(core, best.term, corrected);
        (corrected != core).then_some(best.edit_distance)
    }

    /// OpenNLP's guards. The length is in code points.
    fn correctable(&self, core: &str, placeholders: bool) -> bool {
        if core.chars().nth(self.min_token_length - 1).is_none() {
            return false;
        }
        if self.skip_urls && !placeholders && is_url_like(core) {
            return false;
        }
        if self.skip_numbers && is_number_like(core) {
            return false;
        }
        core.chars().any(java::is_letter)
    }

    fn compound(&self, text: &str, out: &mut ViewBuilder<'_>, skip: Option<&[Span]>, report: &mut Report<'_>) {
        if java::is_blank(text) {
            return;
        }
        let mut pieces: Vec<Piece> = Vec::new();
        let mut phrases: Vec<(usize, Compound)> = Vec::new();
        match skip {
            None => phrases.push((0, self.engine.compound_within(text, self.max_edit_distance))),
            Some(spans) => {
                // The tokens in placeholders stay; the runs of tokens between
                // them are corrected as phrases of their own.
                let mut skip = Skip::new(Some(spans));
                let mut run: Option<Range<usize>> = None;
                for token in tokens(text) {
                    if skip.overlaps(&token) {
                        if let Some(r) = run.take() {
                            phrases.push((r.start, self.engine.compound_within(&text[r], self.max_edit_distance)));
                        }
                        phrases.push((token.start, verbatim(&text[token])));
                    } else {
                        run = Some(run.map_or(token.clone(), |r| r.start..token.end));
                    }
                }
                if let Some(r) = run {
                    phrases.push((r.start, self.engine.compound_within(&text[r], self.max_edit_distance)));
                }
            }
        }
        for (base, phrase) in &phrases {
            for part in &phrase.parts {
                pieces.push(Piece {
                    source: base + part.source.start..base + part.source.end,
                    output: &phrase.term[part.output.clone()],
                    edit_distance: part.edit_distance,
                });
            }
        }
        if pieces.iter().all(|p| p.output.is_empty()) {
            return;
        }
        write_pieces(text, &pieces, out, report);
    }

    fn options(&self) -> u64 {
        hash_config(&(
            self.mode,
            self.min_token_length,
            self.max_edit_distance,
            self.skip_numbers,
            self.skip_urls,
            self.keep_unknown_proper_nouns,
            self.use_placeholders,
            self.engine.fingerprint(),
        ))
    }
}

/// Receives each replaced stretch of a word: its span in the text read,
/// what it became, and the edit distance of the correction.
type Report<'r> = dyn FnMut(Span, &str, usize) + 'r;

fn span(r: Range<usize>) -> Span {
    Span::new(r.start as u32, r.end as u32)
}

/// A token in a placeholder, kept as it is in compound mode.
fn verbatim(token: &str) -> Compound {
    Compound {
        term: token.to_owned(),
        edit_distance: 0,
        count: 0,
        parts: vec![super::compound::CompoundPart {
            source: 0..token.len(),
            output: 0..token.len(),
            edit_distance: 0,
            count: 0,
        }],
    }
}

/// One output word of compound mode and the input it stands for.
struct Piece<'a> {
    source: Range<usize>,
    output: &'a str,
    edit_distance: usize,
}

/// Writes the pieces in order: a piece that differs from its input
/// replaces it; the whitespace before a piece becomes one space once
/// something has been written and nothing before that, as OpenNLP joins
/// the parts; the whitespace after the last piece is deleted.
fn write_pieces(text: &str, pieces: &[Piece<'_>], out: &mut ViewBuilder<'_>, report: &mut Report<'_>) {
    let mut written = false;
    for piece in pieces {
        let separator = if written { " " } else { "" };
        write_gap(text, piece.source.start, separator, out);
        if &text[piece.source.clone()] != piece.output {
            out.replace(piece.source.end, piece.output);
            report(span(piece.source.clone()), piece.output, piece.edit_distance);
        } else {
            out.keep(piece.source.end);
        }
        written |= !separator.is_empty() || !piece.output.is_empty();
    }
    write_gap(text, text.len(), "", out);
}

/// Turns the whitespace from the builder's position to `end` into
/// `separator` ("" or " "). A space already in the run is kept and the
/// rest deleted, rather than the run replaced as a whole, so the edit is
/// as small as it can be and a corrected word next to the run maps back
/// to that word alone, not to the whitespace beside it.
fn write_gap(text: &str, end: usize, separator: &str, out: &mut ViewBuilder<'_>) {
    let start = out.position();
    let gap = &text[start..end];
    if gap == separator {
        out.keep(end);
        return;
    }
    match gap.find(' ').filter(|_| !separator.is_empty()) {
        Some(space) => {
            if space > 0 {
                out.replace(start + space, "");
            }
            out.keep(start + space + 1);
            if start + space + 1 < end {
                out.replace(end, "");
            }
        }
        None => out.replace(end, separator),
    }
}

/// Placeholder spans in text order, walked once alongside the tokens.
struct Skip<'s> {
    spans: Option<&'s [Span]>,
    next: usize,
}

impl<'s> Skip<'s> {
    fn new(spans: Option<&'s [Span]>) -> Skip<'s> {
        Skip { spans, next: 0 }
    }

    fn active(&self) -> bool {
        self.spans.is_some()
    }

    /// Whether `token` shares a byte with a placeholder. Tokens come in
    /// text order.
    fn overlaps(&mut self, token: &Range<usize>) -> bool {
        let Some(spans) = self.spans else { return false };
        while self.next < spans.len() && (spans[self.next].end as usize) <= token.start {
            self.next += 1;
        }
        spans[self.next..]
            .iter()
            .take_while(|s| (s.start as usize) < token.end)
            .any(|s| !s.is_empty() && (s.end as usize) > token.start)
    }
}

impl Normalizer for SpellCorrect {
    const ID: &'static str = "turbo:normalize:spell-correct";
    const VERSION: &'static str = "1";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        self.run(text, out, None, &mut |_, _, _| {});
    }

    fn config_hash(&self) -> u64 {
        self.options()
    }
}

impl Annotator for SpellCorrect {
    const ID: &'static str = <SpellCorrect as Normalizer>::ID;
    const VERSION: &'static str = <SpellCorrect as Normalizer>::VERSION;
    type Reads = (View, Option<Placeholders>);
    type Writes = (View, Corrections);

    /// Corrects the view, writing a row of `Corrections` for each replaced
    /// stretch. With `use_placeholders`, the placeholders must be in the
    /// view read and within the text bounds.
    fn annotate(&self, (src, placeholders): Refs<'_, Self>) -> Result<(View, Corrections), Error> {
        let text = src.text();
        let skip = match placeholders {
            Some(p) if self.use_placeholders => {
                if p.spans.view() != src.id() {
                    return Err(Error::Annotator {
                        annotator: <Self as Annotator>::ID,
                        detail: format!(
                            "the placeholders are in view {}, the text read is view {}",
                            p.spans.view().0,
                            src.id().0
                        ),
                    });
                }
                let mut spans = Vec::with_capacity(p.spans.len());
                for (i, s) in p.spans.iter() {
                    let (start, end) = (s.start as usize, s.end as usize);
                    if start > end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                        return Err(Error::Annotator {
                            annotator: <Self as Annotator>::ID,
                            detail: format!(
                                "placeholder {} span {}..{} is out of bounds for text of {} bytes",
                                i.get(),
                                s.start,
                                s.end,
                                text.len()
                            ),
                        });
                    }
                    if !s.is_empty() {
                        spans.push(s);
                    }
                }
                spans.sort_unstable();
                let mut joined: Vec<Span> = Vec::with_capacity(spans.len());
                for s in spans {
                    match joined.last_mut() {
                        Some(last) if s.start < last.end => last.end = last.end.max(s.end),
                        _ => joined.push(s),
                    }
                }
                Some(joined)
            }
            _ => None,
        };
        let mut out = ViewBuilder::new(text);
        let mut corrections = Corrections::new(src.id());
        self.run(text, &mut out, skip.as_deref(), &mut |s, with, distance| {
            corrections.spans.push(s);
            corrections.suggestion.push(with);
            corrections.edit_distance.push(u8::try_from(distance).unwrap_or(u8::MAX));
        });
        let (normalized, alignment) = out.finish();
        Ok((View::derived(src, normalized, alignment), corrections))
    }

    fn config_hash(&self) -> u64 {
        self.options()
    }
}
