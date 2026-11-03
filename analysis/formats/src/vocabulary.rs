//! Deterministic domain vocabulary from document samples.

use std::collections::{BTreeMap, BTreeSet};

use turbo_stem::stemmer::Stemmer;
use turbo_stopword::StopwordList;

use crate::sample::DocumentSample;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabularyEntry {
    pub id: String,
    pub term: String,
    pub frequency: u64,
    pub document_frequency: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vocabulary {
    entries: Vec<VocabularyEntry>,
}

impl Vocabulary {
    pub fn build<'a>(
        documents: impl IntoIterator<Item = DocumentSample<'a>>,
        stopwords: &StopwordList,
        stemmer: &impl Stemmer,
    ) -> Vocabulary {
        let mut counts: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for document in documents {
            let mut seen = BTreeSet::new();
            for token in document.tokens {
                let mut lower = String::new();
                turbo_unicode::java::to_lowercase(token.as_str(), &mut lower);
                if stopwords.contains(&lower) || !lower.chars().any(char::is_alphanumeric) {
                    continue;
                }
                let stem = stemmer.stem(&lower);
                if stem.is_empty() {
                    continue;
                }
                counts.entry(stem.clone()).or_default().0 += 1;
                seen.insert(stem);
            }
            for stem in seen {
                counts.entry(stem).or_default().1 += 1;
            }
        }
        let entries = counts
            .into_iter()
            .map(|(term, (frequency, document_frequency))| VocabularyEntry {
                id: term.clone(),
                term,
                frequency,
                document_frequency,
            })
            .collect();
        Vocabulary { entries }
    }

    pub fn entries(&self) -> &[VocabularyEntry] {
        &self.entries
    }

    /// A headered TSV term list. The two count columns are retained for the
    /// corpus receipt and ignored by the glossary's two-column reader.
    pub fn write_tsv(&self) -> String {
        let mut out = String::from("id\tterm\tfrequency\tdocument_frequency\n");
        for entry in &self.entries {
            field(&entry.id, &mut out);
            out.push('\t');
            field(&entry.term, &mut out);
            out.push('\t');
            out.push_str(&entry.frequency.to_string());
            out.push('\t');
            out.push_str(&entry.document_frequency.to_string());
            out.push('\n');
        }
        out
    }
}

fn field(text: &str, out: &mut String) {
    if text.contains(['\t', '\r', '\n', '"']) {
        out.push('"');
        out.push_str(&text.replace('"', "\"\""));
        out.push('"');
    } else {
        out.push_str(text);
    }
}
