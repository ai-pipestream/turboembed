//! `SymbolJoiner`: joins multi-token symbol sequences from a table and
//! applies the ten-symbol word map to standalone symbol tokens.

use turbo_analysis::{Annotator, Error, Refs, Span, Spans, View, ViewBuilder};
use turbo_analysis_layers::Tokens;

use super::whole_symbol_word;
use crate::hash_config;

/// Default table of common symbol sequences joined when a tokenizer splits
/// them across touching token spans.
pub const DEFAULT_SYMBOL_SEQUENCES: &[&str] = &[
    "!=",
    "!==",
    "%=",
    "&&",
    "&=",
    "*=",
    "**",
    "++",
    "+=",
    "--",
    "-=",
    "->",
    "..",
    "...",
    "..=",
    ".NET",
    ".net",
    "//",
    "/=",
    "::",
    ":=",
    "<-",
    "<->",
    "<<",
    "<<=",
    "<=",
    "<=>",
    "<>",
    "<|",
    "<~",
    "==",
    "===",
    "=>",
    ">=",
    ">>",
    ">>=",
    ">>>",
    "ASP.NET",
    "AT&T",
    "A&E",
    "B&B",
    "C#",
    "C++",
    "C/O",
    "D3.js",
    "F#",
    "I/O",
    "M&A",
    "N/A",
    "Next.js",
    "Node.js",
    "Nuxt.js",
    "Objective-C",
    "P&L",
    "Q#",
    "Q&A",
    "R&D",
    "React.js",
    "S&L",
    "S&P",
    "Three.js",
    "UTF-16",
    "UTF-32",
    "UTF-8",
    "VB.NET",
    "Vue.js",
    "W/O",
    "^=",
    "a/k/a",
    "asp.net",
    "c#",
    "c++",
    "c/o",
    "d/b/a",
    "d3.js",
    "f#",
    "i/o",
    "n/a",
    "next.js",
    "node.js",
    "nuxt.js",
    "objective-c",
    "q#",
    "r&d",
    "react.js",
    "three.js",
    "utf-16",
    "utf-32",
    "utf-8",
    "vb.net",
    "vue.js",
    "w/",
    "w/o",
    "|=",
    "|>",
    "||",
    "~>",
];

/// Joins touching token spans that match a symbol sequence in its table
/// (`C++`, `.NET`, `->`, `R&D`, etc.) into single tokens and replaces every
/// remaining token that is a single mapped symbol with its English word:
/// with tokens `Dungeons`, `&`, `Dragons` the view reads
/// `Dungeons and Dragons`, while `C`, `+`, `+` is joined into one `C++`
/// token and left as `C++` in the view.
///
/// It reads a `View` and the `Tokens` made on it, and writes both the
/// normalized `View` and the `Tokens` layer on that view with joined
/// symbol-sequence tokens and mapped byte offsets.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymbolJoiner {
    sequences: Vec<String>,
}

impl Default for SymbolJoiner {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolJoiner {
    /// Creates a `SymbolJoiner` with [`DEFAULT_SYMBOL_SEQUENCES`].
    pub fn new() -> SymbolJoiner {
        Self::with_sequences(DEFAULT_SYMBOL_SEQUENCES.iter().copied())
    }

    /// Creates a `SymbolJoiner` with the given table of symbol sequences.
    pub fn with_sequences<I, S>(sequences: I) -> SymbolJoiner
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut list: Vec<String> = sequences.into_iter().map(Into::into).filter(|s| !s.is_empty()).collect();
        list.sort();
        list.dedup();
        SymbolJoiner { sequences: list }
    }

    /// Creates a `SymbolJoiner` with an empty sequence table (only replaces
    /// lone mapped-symbol tokens with words).
    pub fn without_sequences() -> SymbolJoiner {
        SymbolJoiner { sequences: Vec::new() }
    }

    /// The configured symbol sequences, sorted lexicographically.
    pub fn sequences(&self) -> &[String] {
        &self.sequences
    }

    fn contains_sequence(&self, candidate: &str) -> bool {
        self.sequences.binary_search_by(|s| s.as_str().cmp(candidate)).is_ok()
    }

    /// Joins touching token spans in `tokens` whose combined slice in
    /// `src.text()` matches a sequence in `self.sequences` (longest match
    /// first), returning a new `Tokens` layer on `src.id()`.
    pub fn join_tokens(&self, src: &View, tokens: &Tokens) -> Result<Tokens, Error> {
        if tokens.spans.view() != src.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("tokens are in view {}, the text in view {}", tokens.spans.view().0, src.id().0),
            });
        }
        let text = src.text();
        let raw: Vec<Span> = tokens.spans.iter().map(|(_, s)| s).collect();
        if let Some(&oob) = raw.iter().find(|s| s.start > s.end || s.end as usize > text.len()) {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("token span {}..{} out of bounds for {}-byte view", oob.start, oob.end, text.len()),
            });
        }
        if let Some(ref sent) = tokens.sentence
            && sent.len() != raw.len()
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("tokens has {} spans but {} sentence indices", raw.len(), sent.len()),
            });
        }
        let mut joined_spans = Spans::new(src.id());
        let mut joined_sent = tokens.sentence.as_ref().map(|s| Vec::with_capacity(s.len()));
        let mut i = 0;
        while i < raw.len() {
            let mut best_j = i;
            if !self.sequences.is_empty() {
                let start = raw[i].start as usize;
                let mut prev_end = raw[i].end;
                let mut j = i + 1;
                while j < raw.len() && raw[j].start == prev_end {
                    if let Some(ref sent) = tokens.sentence
                        && sent[j] != sent[i]
                    {
                        break;
                    }
                    let end = raw[j].end as usize;
                    if let Some(slice) = text.get(start..end)
                        && self.contains_sequence(slice)
                    {
                        best_j = j;
                    }
                    prev_end = raw[j].end;
                    j += 1;
                }
            }
            joined_spans.push(Span::new(raw[i].start, raw[best_j].end));
            if let (Some(out_s), Some(in_s)) = (&mut joined_sent, &tokens.sentence) {
                out_s.push(in_s[i]);
            }
            i = best_j + 1;
        }
        Ok(Tokens { spans: joined_spans, sentence: joined_sent })
    }
}

impl Annotator for SymbolJoiner {
    const ID: &'static str = "turbo:normalize:symbol-joiner";
    type Reads = (View, Tokens);
    type Writes = (View, Tokens);

    fn annotate(&self, (src, tokens): Refs<'_, Self>) -> Result<(View, Tokens), Error> {
        let joined = self.join_tokens(src, tokens)?;
        let text = src.text();
        // Only one-symbol tokens (after sequence joining) are replaced, so
        // overlapping or unordered tokens elsewhere cannot disturb the order
        // of the writes.
        let mut symbols: Vec<(usize, usize, &'static str)> = joined
            .spans
            .iter()
            .filter_map(|(_, s)| whole_symbol_word(&text[s.range()]).map(|w| (s.start as usize, s.end as usize, w)))
            .collect();
        symbols.sort_unstable_by_key(|&(start, _, _)| start);
        symbols.dedup_by_key(|&mut (start, _, _)| start);
        let mut out = ViewBuilder::new(text);
        for (start, end, word) in symbols {
            out.keep(start);
            out.replace(end, word);
        }
        let (normalized, alignment) = out.finish();
        let view = View::derived(src, normalized, alignment);
        let mut mapped_spans = Spans::new(view.id());
        for (_, span) in joined.spans.iter() {
            mapped_spans.push(view.alignment().span_to_child(span));
        }
        Ok((view, Tokens { spans: mapped_spans, sentence: joined.sentence }))
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.sequences)
    }
}
