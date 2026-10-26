//! GLiNER2's input for a schema: one element sequence per task, joined by
//! `[SEP_STRUCT]`, then `[SEP_TEXT]` and the text tokens, each element
//! tokenized on its own with no `[CLS]` or `[SEP]` around the whole.
//!
//! Tasks come in GLiNER2's order: record types, then the entity types as
//! one task, then each classification. A task is
//! `( [P] <prompt> ( <marker> <name> ... ) )`, the marker `[C]` for a
//! record's fields, `[E]` for entity types and `[L]` for labels. The prompt
//! is the task's name (`entities` for the entity types), followed by
//! ` [DESCRIPTION] <name>: <description>` for each child that has one.
//!
//! The text tokens are a prefix that lists every closed-choice field, then
//! the text's words (GLiNER2's word splitter, lower-cased). The prefix for a
//! record type is `( <record>: <field> ( <choice> | <choice> ) , ... )`.
//! The span head reads every text token at its first subword, so a choice
//! is scored where it stands in the prefix.

use std::ops::Range;

use turbo::sentencepiece::{Scratch, SentencePiece};
use turbo_relation::gliner2::words::{Word, split};

use crate::schema::ExtractionSchema;

/// The ids of GLiNER2's added tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Specials {
    pub sep_struct: u32,
    pub sep_text: u32,
    pub p: u32,
    pub c: u32,
    pub e: u32,
    pub l: u32,
    pub description: u32,
}

/// What a task asks for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The record type at this index of the schema.
    Record(usize),
    Entities,
    /// The classification task at this index of the schema.
    Classification(usize),
}

/// One task's place in the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    pub kind: Kind,
    /// Its `[P]` position.
    pub p: usize,
    /// Each child marker's position, in the schema's order.
    pub markers: Vec<usize>,
}

/// The encoder input for one text and schema.
#[derive(Debug)]
pub struct Measurement {
    pub overhead: usize,
    pub period: usize,
    pub words: Vec<(std::ops::Range<usize>, usize)>,
    /// Cost of each word when it ends a window requiring final punctuation.
    pub endings: Vec<usize>,
}

#[derive(Debug)]
pub struct Prompt<'a> {
    pub ids: Vec<u32>,
    /// The caller's unchanged text. Virtual prompt punctuation may extend a
    /// word beyond its end; emitted source spans must be clipped to this text.
    pub text: &'a str,
    pub words: Vec<Word<'a>>,
    /// The choice prefix's tokens, before the words.
    pub prefix: Vec<String>,
    /// Per record type and field, the index in `prefix` of a closed-choice
    /// field's first choice; its choice `i` is at that index plus `2 * i`.
    pub choice_at: Vec<Vec<Option<usize>>>,
    /// The first subword of each text token, prefix then words; a token
    /// with no subwords takes the position the next element starts at.
    pub token_first: Vec<usize>,
    pub tasks: Vec<Task>,
    /// The ids of the `.` added when the text lacks sentence punctuation.
    pub period: Vec<u32>,
    window_words: Vec<(std::ops::Range<usize>, usize, usize)>,
    /// Per URL word inside the text, by word index: its ids with the `.`
    /// joined, for a window that ends on it.
    url_ids: Vec<(usize, Vec<u32>)>,
}

/// Whether GLiNER2 reads `text` with a `.` added after it.
pub fn needs_period(text: &str) -> bool {
    !text.ends_with(['.', '!', '?'])
}

impl<'a> Prompt<'a> {
    /// How many text tokens precede the words.
    pub fn prefix_len(&self) -> usize {
        self.prefix.len()
    }

    pub fn measure(&self, text_len: usize) -> Measurement {
        let lengths = self.words.iter().map(|w| (w.start..w.end, w.subword_len));
        let subwords: usize = lengths.clone().map(|(_, n)| n).sum();
        Measurement {
            overhead: self.ids.len() - subwords,
            period: self.period.len(),
            words: self
                .window_words
                .iter()
                .filter(|(r, _, _)| r.start < text_len)
                .map(|(r, n, _)| (r.clone(), *n))
                .collect(),
            endings: self.window_words.iter().filter(|(r, _, _)| r.start < text_len).map(|(_, _, n)| *n).collect(),
        }
    }

    /// The input for the window `at` of the text, cut from this one with no
    /// tokenization: the schema and prefix ids as they are, then the words
    /// inside `at` with their own ids and offsets moved to the window, then
    /// the `.` when the window lacks sentence punctuation (joined to a URL
    /// that ends the window, as the reference splits it). Every word is
    /// tokenized on its own, so this is the input built over the window's
    /// text from the same words.
    ///
    /// # Panics
    /// When `at` is not a range of the text on character boundaries.
    pub fn window(&self, at: Range<usize>) -> Prompt<'a> {
        let (len, text) = (self.text.len(), &self.text[at.clone()]);
        let period = needs_period(text);
        let n = self.prefix.len();
        let words_at = self.token_first.get(n).copied().unwrap_or(self.ids.len());
        let mut ids = self.ids[..words_at].to_vec();
        let mut token_first = self.token_first[..n].to_vec();
        let (mut words, mut window_words, mut url_ids) = (Vec::new(), Vec::new(), Vec::new());
        for (k, (w, &first)) in self.words.iter().zip(&self.token_first[n..]).enumerate() {
            let end = w.end.min(len);
            if w.start >= len || w.start < at.start || end > at.end {
                continue;
            }
            let mut word = Word { start: w.start - at.start, end: w.end - at.start, ..w.clone() };
            let mut own = &self.ids[first..first + w.subword_len];
            if let Ok(u) = self.url_ids.binary_search_by_key(&k, |(i, _)| *i) {
                if period && end == at.end {
                    word.text.to_mut().push('.');
                    word.end += 1;
                    own = &self.url_ids[u].1;
                } else {
                    url_ids.push((words.len(), self.url_ids[u].1.clone()));
                }
            }
            let (r, base, ending) = &self.window_words[k];
            window_words.push((r.start - at.start..r.end - at.start, *base, *ending));
            token_first.push(ids.len());
            ids.extend_from_slice(own);
            word.subword_len = own.len();
            words.push(word);
        }
        if period && words.last().is_none_or(|w| w.end <= text.len()) {
            token_first.push(ids.len());
            ids.extend_from_slice(&self.period);
            let (start, subword_len) = (text.len(), self.period.len());
            words.push(Word { text: ".".into(), start, end: start + 1, subword_len });
        }
        Prompt {
            ids,
            text,
            words,
            prefix: self.prefix.clone(),
            choice_at: self.choice_at.clone(),
            token_first,
            tasks: self.tasks.clone(),
            period: self.period.clone(),
            window_words,
            url_ids,
        }
    }
}

/// `s` tokenized as the reference's tokenizer does, with `[DESCRIPTION]`
/// kept as its one id between the pieces around it.
fn encode(sp: &SentencePiece, specials: &Specials, s: &str, scratch: &mut Scratch, ids: &mut Vec<u32>) {
    let mut rest = s;
    while let Some(at) = rest.find("[DESCRIPTION]") {
        sp.encode_ids(&rest[..at], scratch, ids);
        ids.push(specials.description);
        rest = &rest[at + "[DESCRIPTION]".len()..];
    }
    sp.encode_ids(rest, scratch, ids);
}

/// The prompt string of a task: its name and its children's descriptions.
fn prompt_text<'a>(name: &str, described: impl Iterator<Item = (&'a str, Option<&'a String>)>) -> String {
    let mut s = name.to_owned();
    for (child, d) in described {
        if let Some(d) = d {
            s.push_str(&format!(" [DESCRIPTION] {child}: {d}"));
        }
    }
    s
}

/// The choice prefix: every closed-choice field of every record type, and
/// where each field's first choice stands in it.
fn prefix(schema: &ExtractionSchema) -> (Vec<String>, Vec<Vec<Option<usize>>>) {
    let mut out = Vec::new();
    let mut at = Vec::with_capacity(schema.structures.len());
    for s in &schema.structures {
        let mut inner: Vec<String> = Vec::new();
        let mut first = Vec::with_capacity(s.fields.len());
        for f in &s.fields {
            let Some(choices) = &f.choices else {
                first.push(None);
                continue;
            };
            // After `(`, `<record>:`, the earlier fields and `<field> (`.
            first.push(Some(out.len() + 2 + inner.len() + 2));
            inner.extend([f.name.clone(), "(".into()]);
            for (i, c) in choices.iter().enumerate() {
                if i > 0 {
                    inner.push("|".into());
                }
                inner.push(c.clone());
            }
            inner.extend([")".into(), ",".into()]);
        }
        if inner.pop().is_some() {
            out.extend(["(".into(), format!("{}:", s.name)]);
            out.extend(inner);
            out.push(")".into());
        }
        at.push(first);
    }
    (out, at)
}

/// The input for `schema` over `text`. The schema is assumed validated.
pub fn build<'a>(sp: &SentencePiece, specials: &Specials, text: &'a str, schema: &ExtractionSchema) -> Prompt<'a> {
    build_words(sp, specials, text, schema, split(text))
}

pub fn build_words<'a>(
    sp: &SentencePiece,
    specials: &Specials,
    text: &'a str,
    schema: &ExtractionSchema,
    mut words: Vec<Word<'a>>,
) -> Prompt<'a> {
    let mut tasks: Vec<(Kind, String, u32, Vec<&str>)> = Vec::new();
    for (i, s) in schema.structures.iter().enumerate() {
        let p = prompt_text(&s.name, s.fields.iter().map(|f| (f.name.as_str(), f.description.as_ref())));
        tasks.push((Kind::Record(i), p, specials.c, s.fields.iter().map(|f| f.name.as_str()).collect()));
    }
    if !schema.entities.is_empty() {
        let p = prompt_text("entities", schema.entities.iter().map(|e| (e.name.as_str(), e.description.as_ref())));
        tasks.push((Kind::Entities, p, specials.e, schema.entities.iter().map(|e| e.name.as_str()).collect()));
    }
    for (i, c) in schema.classifications.iter().enumerate() {
        let p = prompt_text(&c.name, c.labels.iter().map(|l| (l.name.as_str(), l.description.as_ref())));
        tasks.push((Kind::Classification(i), p, specials.l, c.labels.iter().map(|l| l.name.as_str()).collect()));
    }
    let mut scratch = Scratch::default();
    let mut ids = Vec::new();
    let mut out_tasks = Vec::with_capacity(tasks.len());
    for (k, (kind, p_text, marker, children)) in tasks.into_iter().enumerate() {
        if k > 0 {
            ids.push(specials.sep_struct);
        }
        sp.encode_ids("(", &mut scratch, &mut ids);
        let p = ids.len();
        ids.push(specials.p);
        encode(sp, specials, &p_text, &mut scratch, &mut ids);
        sp.encode_ids("(", &mut scratch, &mut ids);
        let mut markers = Vec::with_capacity(children.len());
        for child in children {
            markers.push(ids.len());
            ids.push(marker);
            encode(sp, specials, child, &mut scratch, &mut ids);
        }
        sp.encode_ids(")", &mut scratch, &mut ids);
        sp.encode_ids(")", &mut scratch, &mut ids);
        out_tasks.push(Task { kind, p, markers });
    }
    ids.push(specials.sep_text);
    turbo_relation::gliner2::words::append_period(text, &mut words);
    let (prefix, choice_at) = prefix(schema);
    let mut token_first = Vec::with_capacity(prefix.len() + words.len());
    for t in &prefix {
        token_first.push(ids.len());
        encode(sp, specials, t, &mut scratch, &mut ids);
    }
    let mut period = Vec::new();
    encode(sp, specials, ".", &mut scratch, &mut period);
    let mut window_words = Vec::new();
    let mut url_ids = Vec::new();
    let mut boundary_ids = Vec::new();
    for (k, w) in words.iter_mut().enumerate() {
        token_first.push(ids.len());
        let at = ids.len();
        encode(sp, specials, &w.text, &mut scratch, &mut ids);
        w.subword_len = ids.len() - at;
        if w.start >= text.len() {
            continue;
        }
        let (base, ending) = if turbo_relation::gliner2::words::is_url(&w.text) {
            // A terminal URL absorbs the virtual period. Cache both variants
            // once so every window still uses prefix sums without tokenization.
            boundary_ids.clear();
            if w.end > text.len() {
                encode(sp, specials, &w.text[..w.text.len() - 1], &mut scratch, &mut boundary_ids);
                (boundary_ids.len(), w.subword_len)
            } else {
                encode(sp, specials, &format!("{}.", w.text), &mut scratch, &mut boundary_ids);
                url_ids.push((k, boundary_ids.clone()));
                (w.subword_len, boundary_ids.len())
            }
        } else {
            (w.subword_len, w.subword_len + period.len())
        };
        window_words.push((w.start..w.end.min(text.len()), base, ending));
    }
    Prompt { ids, text, words, prefix, choice_at, token_first, tasks: out_tasks, period, window_words, url_ids }
}
