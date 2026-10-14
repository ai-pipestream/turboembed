//! The name samples of brat documents: `BratDocumentParser`.

use std::collections::{BTreeSet, HashMap};

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use super::annotation::{BratAnnotation, Fragment};
use super::document::BratDocument;
use crate::error::FormatError;
use crate::ner::{NerError, SentenceSpans, TokenSpans};
use crate::sample::{Name, NameSample};
use crate::source::{Field, split_fields};

/// The UTF-16 offsets of a text beside its byte offsets, as brat's
/// offsets count in Java.
struct Utf16 {
    /// The UTF-16 offset of each byte offset that starts a character, and
    /// of the end.
    units_at: Vec<u32>,
    /// The byte offset of each UTF-16 offset, `None` between the two
    /// halves of a surrogate pair.
    bytes_at: Vec<Option<u32>>,
}

impl Utf16 {
    fn new(text: &str) -> Utf16 {
        let mut units_at = vec![0; text.len() + 1];
        let mut bytes_at = Vec::with_capacity(text.len() + 1);
        let mut unit = 0;
        for (i, c) in text.char_indices() {
            units_at[i] = unit;
            bytes_at.push(Some(i as u32));
            if c.len_utf16() == 2 {
                bytes_at.push(None);
            }
            unit += c.len_utf16() as u32;
        }
        units_at[text.len()] = unit;
        bytes_at.push(Some(text.len() as u32));
        Utf16 { units_at, bytes_at }
    }

    fn units(&self, byte: u32) -> i64 {
        i64::from(self.units_at[byte as usize])
    }

    fn byte(&self, unit: i64) -> Option<u32> {
        usize::try_from(unit).ok().and_then(|u| self.bytes_at.get(u).copied().flatten())
    }

    /// Whether the UTF-16 unit at `unit` is whitespace, as
    /// `StringUtil.isWhitespace(text.charAt(unit))`: half of a surrogate
    /// pair never is.
    fn is_whitespace(&self, text: &str, unit: i64, mode: WhitespaceMode) -> bool {
        self.byte(unit)
            .and_then(|b| text[b as usize..].chars().next())
            .is_some_and(|c| c.len_utf16() == 1 && mode.is_whitespace(c as u32))
    }

    /// `Span.trim(text)` of units `start..end`.
    fn trim(&self, text: &str, start: i64, end: i64, mode: WhitespaceMode) -> (i64, i64) {
        let mut new_start = start;
        while new_start < end && self.is_whitespace(text, new_start, mode) {
            new_start += 1;
        }
        let mut new_end = end;
        while new_end > start && self.is_whitespace(text, new_end - 1, mode) {
            new_end -= 1;
        }
        if new_start > new_end { (start, start) } else { (new_start, new_end) }
    }
}

/// Turns brat documents into name samples as `BratDocumentParser` does
/// (`BratNameSampleStream` runs it over each document of a
/// `BratDocumentStream`): the text is split into sentences and each
/// sentence into tokens, and each text-bound annotation of a type kept
/// is a name over the tokens its fragments start and end on.
///
/// - A sentence that starts inside a fragment that starts before it is
///   joined to the sentence before. The fragment asked is the last one
///   put in OpenNLP's map of offsets, of the annotations in the order
///   [`BratDocument::annotations`] gives and each one's fragments in
///   order.
/// - A fragment is trimmed of whitespace and must start where a token of
///   its sentence starts and end where one ends; one that does not, or
///   that is not inside one sentence, is dropped (OpenNLP logs it).
///   Fragments of an annotation that meet are joined into one name.
/// - The first sample of a document clears adaptive data.
/// - Overlapping names fail the whole document
///   ([`NerError::BratOverlappingNames`]): OpenNLP's stream throws and
///   goes on with the next document, the document's samples lost.
///
/// Offsets count UTF-16 code units, as Java's do; brat counts code
/// points, so the offsets of a text with characters outside the Basic
/// Multilingual Plane are read as OpenNLP reads them, not as brat wrote
/// them.
pub struct BratParser<S, T> {
    sentences: S,
    tokenizer: T,
    name_types: Option<BTreeSet<String>>,
    whitespace: WhitespaceMode,
}

impl<S: SentenceSpans, T: TokenSpans> BratParser<S, T> {
    /// A parser that splits sentences with `sentences` and tokens with
    /// `tokenizer`, keeps the names of `name_types` (every type when
    /// `None`), and trims fragments of whitespace in `whitespace`.
    ///
    /// # Errors
    ///
    /// [`NerError::BratNoNameTypes`] for an empty set of name types, which
    /// `BratDocumentParser`'s constructor refuses.
    pub fn new(
        sentences: S,
        tokenizer: T,
        name_types: Option<BTreeSet<String>>,
        whitespace: WhitespaceMode,
    ) -> Result<BratParser<S, T>, FormatError> {
        if name_types.as_ref().is_some_and(BTreeSet::is_empty) {
            return Err(NerError::BratNoNameTypes.into());
        }
        Ok(BratParser { sentences, tokenizer, name_types, whitespace })
    }

    /// The name types `BratNameSampleStreamFactory` takes from its
    /// `nameTypes` parameter: the values between commas, trimmed
    /// (`String.trim`), or every type (`None`) when the parameter is
    /// empty or only commas.
    pub fn name_types_from_param(names: &str) -> Option<BTreeSet<String>> {
        let piece = crate::source::Piece { text: names, span: Span::new(0, names.len() as u32) };
        let parts = split_fields(piece, ',');
        if parts.is_empty() {
            return None;
        }
        Some(parts.iter().map(|p| p.text.trim_matches(|c| c <= ' ').to_owned()).collect())
    }

    /// Whether an annotation is a name the parser keeps
    /// (`isSpanAnnotation`).
    fn keeps(&self, a: &BratAnnotation) -> bool {
        matches!(a.value, super::AnnotationValue::Span { .. })
            && self.name_types.as_ref().is_none_or(|t| t.contains(&a.kind))
    }

    /// The name samples of `doc`, a sample per sentence.
    ///
    /// # Errors
    ///
    /// [`NerError::BratOverlappingNames`] when two names of a sentence
    /// overlap.
    pub fn parse<'d>(&self, doc: &'d BratDocument) -> Result<Vec<NameSample<'d>>, FormatError> {
        let text = doc.text();
        let units = Utf16::new(text);
        let names: Vec<&BratAnnotation> = doc.annotations().iter().filter(|a| self.keeps(a)).collect();
        let fragments = || names.iter().flat_map(|a| a.fragments());

        let mut detected = Vec::new();
        self.sentences.sentence_spans(text, &mut detected);
        let mut sentences: Vec<(i64, i64)> = Vec::new();
        for s in &detected {
            let (start, end) = (units.units(s.start), units.units(s.end));
            let covering = fragments().rfind(|f| i64::from(f.start) <= start && start < i64::from(f.end));
            match covering {
                Some(f) if !sentences.is_empty() && i64::from(f.start) < start => {
                    let last = sentences.pop().expect("a sentence before");
                    sentences.push((last.0, end));
                }
                _ => sentences.push((start, end)),
            }
        }

        let mut samples = Vec::with_capacity(sentences.len());
        let mut spans = Vec::new();
        for &(start, end) in &sentences {
            let from = units.byte(start).expect("a sentence starts on a character") as usize;
            let to = units.byte(end).expect("a sentence ends on a character") as usize;
            spans.clear();
            self.tokenizer.token_spans(&text[from..to], &mut spans);
            // a token's start offset, negated, and its end offset, to the
            // index of the token it starts and of the one after it ends;
            // a token starting at offset 0 is under key 0, as in Java
            let mut index: HashMap<i64, u32> = HashMap::new();
            let mut tokens = Vec::with_capacity(spans.len());
            for (i, t) in spans.iter().enumerate() {
                let span = Span::new(from as u32 + t.start, from as u32 + t.end);
                index.insert(-units.units(span.start), i as u32);
                index.insert(units.units(span.end), i as u32 + 1);
                tokens.push(Field::of(text, span));
            }
            let mut sample_names = Vec::new();
            for a in &names {
                let mut mapped: Vec<Option<Name<'d>>> = Vec::new();
                for &Fragment { start: f_start, end: f_end } in a.fragments() {
                    let (f_start, f_end) = (i64::from(f_start), i64::from(f_end));
                    if !(start <= f_start && f_end <= end) {
                        continue;
                    }
                    let (t_start, t_end) = units.trim(text, f_start, f_end, self.whitespace);
                    if let (Some(&b), Some(&e)) = (index.get(&-t_start), index.get(&t_end)) {
                        mapped.push(Some(Name::new(b, e, Some(a.kind.as_str()))?));
                    }
                }
                mapped.sort_by(|x, y| x.as_ref().unwrap().java_cmp(y.as_ref().unwrap()));
                for i in 1..mapped.len() {
                    let (prev, cur) = (mapped[i - 1].unwrap(), mapped[i].unwrap());
                    if prev.end == cur.start {
                        mapped[i] = Some(Name::new(prev.start, cur.end, cur.kind)?);
                        mapped[i - 1] = None;
                    }
                }
                sample_names.extend(mapped.into_iter().flatten());
            }
            let source = Span::new(from as u32, to as u32);
            let sample = NameSample::new(tokens, sample_names, samples.is_empty(), source, 0).map_err(|e| match e {
                FormatError::OverlappingNames { first, second, sentence, .. } => {
                    NerError::BratOverlappingNames { file: doc.id().to_owned(), first, second, sentence }.into()
                }
                e => e,
            })?;
            samples.push(sample);
        }
        Ok(samples)
    }
}

/// The name samples of brat documents read one after the other, as
/// `BratNameSampleStream` reads a `BratDocumentStream`: a document that
/// could not be parsed, or whose samples could not be made, gives its
/// error and the next document is read.
pub struct BratSamples<'d, 'p, S, T> {
    documents: std::slice::Iter<'d, Result<BratDocument, FormatError>>,
    parser: &'p BratParser<S, T>,
    current: std::vec::IntoIter<NameSample<'d>>,
}

impl<'d, 'p, S: SentenceSpans, T: TokenSpans> BratSamples<'d, 'p, S, T> {
    /// The samples of `documents` ([`super::read_brat_corpus`]) as `parser`
    /// makes them.
    pub fn new(documents: &'d [Result<BratDocument, FormatError>], parser: &'p BratParser<S, T>) -> Self {
        BratSamples { documents: documents.iter(), parser, current: Vec::new().into_iter() }
    }
}

impl<'d, S: SentenceSpans, T: TokenSpans> Iterator for BratSamples<'d, '_, S, T> {
    type Item = Result<NameSample<'d>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(s) = self.current.next() {
                return Some(Ok(s));
            }
            match self.documents.next()? {
                Err(e) => return Some(Err(e.clone())),
                Ok(doc) => match self.parser.parse(doc) {
                    Err(e) => return Some(Err(e)),
                    Ok(samples) => self.current = samples.into_iter(),
                },
            }
        }
    }
}
