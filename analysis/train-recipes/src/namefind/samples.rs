use turbo_analysis::Span;
use turbo_formats::masc::{MascCorpus, MascDocument, MascSentence, Unplaced};
use turbo_formats::{Field, Name, NameSample};
use turbo_zoo::Sha256;

use crate::card::NamesCard;
use crate::error::{Error, Result};
use crate::spans::SpanTypes;

/// The name samples read from one side of the split, and what reading
/// them counted.
#[derive(Debug, Clone)]
pub struct NameSamples<'c> {
    pub(super) samples: Vec<NameSample<'c>>,
    pub(super) counts: NamesCard,
    pub(super) skipped: usize,
}

impl<'c> NameSamples<'c> {
    /// The samples, a unit of the gate each.
    pub fn samples(&self) -> &[NameSample<'c>] {
        &self.samples
    }

    /// What reading the samples counted.
    pub fn counts(&self) -> &NamesCard {
        &self.counts
    }

    /// Sentences skipped for having no token.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The SHA-256 of the samples as scored for `types`: for each sample,
    /// its number of tokens as a little endian `u64`, each token's text
    /// as its length in bytes (a little endian `u64`) and its bytes, its
    /// number of names as a `u64`, each name's start and end as `u64`s and
    /// its type as its length and its bytes, and a byte, 1 when the sample
    /// clears the adaptive data and 0 otherwise; after the samples, the
    /// number of types scored as a `u64` and each type, then the number of
    /// renamed types and each one's held-out and scored type, each as its
    /// length and its bytes.
    pub fn sha256(&self, types: &SpanTypes) -> String {
        let mut bytes = Vec::new();
        let put = |bytes: &mut Vec<u8>, s: &str| {
            bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
            bytes.extend_from_slice(s.as_bytes());
        };
        for sample in &self.samples {
            bytes.extend_from_slice(&(sample.tokens().len() as u64).to_le_bytes());
            for token in sample.tokens() {
                put(&mut bytes, token.as_str());
            }
            bytes.extend_from_slice(&(sample.names().len() as u64).to_le_bytes());
            for name in sample.names() {
                bytes.extend_from_slice(&u64::from(name.start).to_le_bytes());
                bytes.extend_from_slice(&u64::from(name.end).to_le_bytes());
                put(&mut bytes, name.kind.unwrap_or_default());
            }
            bytes.push(u8::from(sample.clear_adaptive_data()));
        }
        bytes.extend_from_slice(&(types.types.len() as u64).to_le_bytes());
        for t in types.types {
            put(&mut bytes, t);
        }
        bytes.extend_from_slice(&(types.renamed.len() as u64).to_le_bytes());
        for (held_out, scored) in types.renamed {
            put(&mut bytes, held_out);
            put(&mut bytes, scored);
        }
        Sha256::of(&bytes).to_string()
    }
}

/// Reads the name samples of `corpus` for `types`, giving each to
/// `each` (see the module documentation): a sample per sentence of the
/// documents with a named entity file, in the corpus's order, but for
/// sentences of no tokens. Returns what reading counted, and how many
/// sentences were skipped.
///
/// # Errors
///
/// What reading a sentence's tokens fails on (a document without Penn
/// tokens, an offset outside the text), or a name past the tokens, which
/// no entity read from the file is.
pub fn read_samples<'c>(
    corpus: &'c MascCorpus,
    types: &SpanTypes,
    mut each: impl FnMut(NameSample<'c>) -> Result<()>,
) -> Result<(NamesCard, usize)> {
    let mut counts = NamesCard::default();
    let mut skipped = 0;
    for doc in corpus.documents().iter().filter(|d| d.has_named_entities()) {
        counts.documents += 1;
        for entity in doc.unplaced_entities() {
            if entity.kind.as_deref().and_then(|k| types.held_out(k)).is_some() {
                match entity.cause {
                    Unplaced::OutsideSentences => counts.outside_sentences += 1,
                    Unplaced::DroppedSentence => counts.in_dropped_sentences += 1,
                }
            }
        }
        let mut first = true;
        for sentence in doc.sentences() {
            match sample(doc, sentence, types, first, &mut counts)? {
                Some(s) => {
                    first = false;
                    each(s)?;
                }
                None => skipped += 1,
            }
        }
    }
    Ok((counts, skipped))
}

/// The sample of `sentence`, which clears the adaptive data when `first`,
/// or `None` for a sentence of no tokens; adds what it read to `counts`.
fn sample<'c>(
    doc: &'c MascDocument,
    sentence: &MascSentence,
    types: &SpanTypes,
    first: bool,
    counts: &mut NamesCard,
) -> Result<Option<NameSample<'c>>> {
    let tokens: Vec<Field<'c>> =
        doc.token_spans(sentence)?.into_iter().map(|span| Field::of(doc.text(), span)).collect();
    let (Some(start), Some(end)) = (tokens.iter().map(|t| t.span.start).min(), tokens.iter().map(|t| t.span.end).max())
    else {
        return Ok(None);
    };
    let mut names: Vec<(Name<'static>, bool)> = Vec::new();
    for entity in sentence.entities() {
        match entity.kind.as_deref().and_then(|k| types.held_out(k)) {
            Some(kind) => names.push((
                Name { start: entity.start, end: entity.end, kind: Some(kind), source: Span::default() },
                !entity.is_contiguous(),
            )),
            None => counts.other_types += 1,
        }
    }
    names.sort_by(|a, b| a.0.java_cmp(&b.0));
    let mut kept: Vec<Name<'c>> = Vec::with_capacity(names.len());
    for (name, gapped) in names {
        if kept.last().is_some_and(|k| name.start < k.end) {
            counts.overlapping += 1;
            continue;
        }
        counts.names += 1;
        counts.gapped += u64::from(gapped);
        kept.push(name);
    }
    let source = Span::new(start, end);
    let sample = NameSample::new(tokens, kept, first, source, 0)
        .map_err(|e| Error::Sample { path: doc.path().to_owned(), message: e.to_string() })?;
    Ok(Some(sample))
}
