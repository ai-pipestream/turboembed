//! The analysis stage: a document's text as a `turbo_analysis` document
//! with the layers the chunkers read. Sentences come first; then, when
//! asked for, the structure; then the embedder's subword pieces, which
//! every chunker that counts tokens counts with.
//!
//! The text is analyzed as it was loaded, in the source view, so every
//! chunk's span is a byte range of the set's own text, the offsets span
//! judgements use.

use std::path::Path;

use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, Structure};
use turbo_maxent::ModelFile;
use turbo_sentdetect::{SentenceDetector, SentenceModel, WhitespaceMode, newline_spans};
use turbo_structure::Syntax;

use crate::Result;
use crate::embedder::Embedder;

/// Where the sentences come from.
pub enum Sentencer {
    /// One sentence per line that is not blank (OpenNLP's
    /// `NewlineSentenceDetector`).
    Newline,
    /// An OpenNLP maxent sentence model (`SentenceDetectorME`).
    Maxent {
        detector: SentenceDetector,
        /// The model file's name, without its directory.
        file: String,
        sha256: String,
    },
}

impl Sentencer {
    /// The maxent detector with the model file at `path`, in OpenNLP
    /// 3.x's whitespace mode.
    pub fn maxent(path: &Path) -> Result<Sentencer> {
        let at = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
        let file = ModelFile::read(path).map_err(|e| at(&e))?;
        let sha256 = file.sha256().to_hex();
        let model = SentenceModel::open(file).map_err(|e| at(&e))?;
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        Ok(Sentencer::Maxent { detector: SentenceDetector::new(model, WhitespaceMode::Unicode), file: name, sha256 })
    }

    fn name(&self) -> String {
        match self {
            Sentencer::Newline => "opennlp:sentences newline".into(),
            Sentencer::Maxent { file, sha256, .. } => format!("opennlp:sentences maxent {file} sha256 {sha256}"),
        }
    }

    fn spans(&self, text: &str) -> Spans<Sentences> {
        let mut spans = Spans::new(ViewId::SOURCE);
        match self {
            Sentencer::Newline => {
                let mut out = Vec::new();
                newline_spans(text, WhitespaceMode::Unicode, &mut out);
                out.into_iter().for_each(|s| {
                    spans.push(s);
                });
            }
            Sentencer::Maxent { detector, .. } => {
                let mut out = Vec::new();
                detector.detect(text, &mut out);
                out.into_iter().for_each(|(s, _)| {
                    spans.push(s);
                });
            }
        }
        spans
    }
}

/// What the analysis stage writes on each document.
pub struct Analysis {
    pub sentences: Sentencer,
    /// The syntax the structure is read in; None for no structure.
    pub structure: Option<Syntax>,
}

impl Analysis {
    /// The layers it writes, by name, in the order it writes them, as a
    /// record lists them.
    pub fn names(&self) -> Vec<String> {
        let mut out = vec![self.sentences.name()];
        match self.structure {
            Some(Syntax::Markdown) => out.push("turbo:structure markdown".into()),
            Some(Syntax::Plain) => out.push("turbo:structure plain".into()),
            None => {}
        }
        out.push("opennlp:subwords".into());
        out
    }

    /// The document of `text`, with its sentences, structure and subwords.
    pub fn analyze(&self, text: &str, embedder: &dyn Embedder) -> Result<Document> {
        if u32::try_from(text.len()).is_err() {
            return Err(format!("a document of {} bytes is over u32::MAX", text.len()));
        }
        let mut doc = Document::new(text);
        let fail = |e: turbo_analysis::Error| e.to_string();
        doc.add(&Key::DEFAULT, Sentences { spans: self.sentences.spans(text) }, Provenance::by("turbo-eval"))
            .map_err(fail)?;
        if let Some(syntax) = self.structure {
            let s: Structure = turbo_structure::detect(text, syntax, ViewId::SOURCE);
            doc.add(&Key::DEFAULT, s, Provenance::by("turbo-eval")).map_err(fail)?;
        }
        doc.add(&Key::DEFAULT, embedder.subwords(text)?, Provenance::by("turbo-eval")).map_err(fail)?;
        Ok(doc)
    }
}

/// The span of sentences `[i, j)`, from the first's start to the last's
/// end.
pub(crate) fn sentence_run(sentences: &Sentences, i: usize, j: usize) -> Span {
    let (first, last) = (
        sentences.spans.get(turbo_analysis::Ix::new(i as u32)),
        sentences.spans.get(turbo_analysis::Ix::new(j as u32 - 1)),
    );
    Span::new(first.start, last.end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::stub::Stub;

    #[test]
    fn the_maxent_model_splits_sentences_and_is_named_by_its_hash() {
        let Some(path) = turbo_goldens::published_model("opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin") else {
            return;
        };
        let a = Analysis { sentences: Sentencer::maxent(&path).unwrap(), structure: Some(Syntax::Plain) };
        let text = "Cats purr all day. Dogs bark at night.";
        let doc = a.analyze(text, &Stub::new(&["cats"])).unwrap();
        let s: Vec<&str> = doc.get::<Sentences>().unwrap().spans.iter().map(|(_, s)| &text[s.range()]).collect();
        assert_eq!(s, ["Cats purr all day.", "Dogs bark at night."]);
        assert!(doc.get::<Structure>().is_some());
        // The SHA-256 of the file in analysis/sentdetect/tests/models/README.md.
        let names = a.names();
        assert_eq!(
            names[0],
            "opennlp:sentences maxent opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin sha256 \
             4de58d12b09421003c8b1ba13e8b58ec533b28e1f22a65272119dc0993d333e8"
        );
        assert_eq!(names[1..], ["turbo:structure plain", "opennlp:subwords"]);
    }
}
