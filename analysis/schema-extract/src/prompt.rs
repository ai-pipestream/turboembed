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

use turbo::sentencepiece::{Scratch, SentencePiece};
use turbo_relation::gliner2::prompt::normalize;
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
}

#[derive(Debug)]
pub struct Prompt<'a> {
    pub ids: Vec<u32>,
    /// The text as GLiNER2 reads it: with a final `.` when it lacks
    /// sentence punctuation. Word offsets index it.
    pub text: String,
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
    pub period: usize,
}

impl Prompt<'_> {
    /// How many text tokens precede the words.
    pub fn prefix_len(&self) -> usize {
        self.prefix.len()
    }

    pub fn measure(&self, text_len: usize) -> Measurement {
        let lengths = self.words.iter().map(|w| (w.start..w.end, w.subword_len));
        let subwords: usize = lengths.clone().map(|(_, n)| n).sum();
        Measurement {
            overhead: self.ids.len() - subwords,
            period: self.period,
            words: lengths.filter(|(r, _)| r.start < text_len).collect(),
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
    let normalized = normalize(text);
    if normalized.len() != text.len() {
        words.push(Word { text: ".".into(), start: text.len(), end: text.len() + 1, subword_len: 0 });
    }
    let (prefix, choice_at) = prefix(schema);
    let mut token_first = Vec::with_capacity(prefix.len() + words.len());
    for t in &prefix {
        token_first.push(ids.len());
        encode(sp, specials, t, &mut scratch, &mut ids);
    }
    for w in &mut words {
        token_first.push(ids.len());
        let at = ids.len();
        encode(sp, specials, &w.text, &mut scratch, &mut ids);
        w.subword_len = ids.len() - at;
    }
    let mut period = Vec::new();
    encode(sp, specials, ".", &mut scratch, &mut period);
    Prompt { ids, text: normalized, words, prefix, choice_at, token_first, tasks: out_tasks, period: period.len() }
}
