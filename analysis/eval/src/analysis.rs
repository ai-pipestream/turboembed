//! The analysis stage: a document's text as a `turbo_analysis` document
//! with the layers the chunkers read. Sentences come first; then, when
//! asked for, the structure; then the embedder's subword pieces, which
//! every chunker that counts tokens counts with.
//!
//! The text is analyzed as it was loaded, in the source view, so every
//! chunk's span is a byte range of the set's own text, the offsets span
//! judgements use.

use std::path::Path;

use turbo_analysis::{Annotator, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, Structure};
use turbo_sat::{SatAnnotator, SatOptions};
use turbo_sentdetect::{RuleSentenceDetector, newline_spans};
use turbo_structure::Syntax;

use crate::Result;
use crate::embedder::Embedder;

/// The rule splitter with no unpunctuated-script fallback; the field is
/// private so no model or device can be attached to it.
pub struct RuleSentences(RuleSentenceDetector);

/// Where the sentences come from.
pub enum Sentencer {
    /// The rule splitter of `turbo-sentdetect` for a language: UAX #29
    /// with the language's tailorings and abbreviation inventory. It
    /// needs no model file, and it is the default. Built only by
    /// [`Sentencer::rules`], so it never has a model-backed fallback and
    /// its record name (language, version, configuration hash) says all
    /// that decides its spans.
    Rules(RuleSentences),
    /// One sentence per line that is not blank (OpenNLP's
    /// `NewlineSentenceDetector`).
    Newline,
    /// The SaT model tier over a sealed bundle, named explicitly with
    /// `--sentences sat`; never the default and never a fallback.
    Sat { annotator: Box<SatAnnotator> },
}

impl Sentencer {
    /// The rule splitter for the BCP 47 language tag `tag`.
    pub fn rules(tag: &str) -> Sentencer {
        Sentencer::Rules(RuleSentences(RuleSentenceDetector::new(tag)))
    }

    /// The SaT annotator over the bundle directory at `path`.
    pub fn sat(path: &Path, options: &SatOptions) -> Result<Sentencer> {
        let annotator = SatAnnotator::open(path, options).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Sentencer::Sat { annotator: Box::new(annotator) })
    }

    fn name(&self) -> String {
        match self {
            Sentencer::Rules(RuleSentences(d)) => format!(
                "{} {} version {} config {:016x}",
                RuleSentenceDetector::ID,
                d.splitter.language().rules_code(),
                RuleSentenceDetector::VERSION,
                d.config_hash()
            ),
            Sentencer::Newline => "opennlp:sentences newline".into(),
            Sentencer::Sat { annotator } => {
                let model = annotator.model();
                let execution = annotator.execution();
                match (model, execution) {
                    (Some(model), Some(execution)) => format!(
                        "turbo:sat {} revision {} sha256 {} backend {} precision {} config {:016x}",
                        model.id,
                        model.revision,
                        model.sha256,
                        execution.backend,
                        execution.precision,
                        annotator.config_hash()
                    ),
                    _ => "turbo:sat unknown".into(),
                }
            }
        }
    }

    fn spans(&self, text: &str) -> Result<Spans<Sentences>> {
        let mut spans = Spans::new(ViewId::SOURCE);
        match self {
            Sentencer::Rules(RuleSentences(d)) => {
                let mut out = Vec::new();
                d.splitter.try_split(text, &mut out).map_err(|e| e.to_string())?;
                out.into_iter().for_each(|s| {
                    spans.push(s);
                });
            }
            Sentencer::Newline => {
                let mut out = Vec::new();
                newline_spans(text, &mut out);
                out.into_iter().for_each(|s| {
                    spans.push(s);
                });
            }
            Sentencer::Sat { annotator } => {
                let (token_spans, scores) = annotator
                    .scores(&[text])
                    .into_iter()
                    .next()
                    .ok_or("turbo:sat scored no text")?
                    .map_err(|e| e.to_string())?;
                let mut out = Vec::new();
                annotator.sentences(text, &token_spans, &scores, &mut out);
                out.into_iter().for_each(|s| {
                    spans.push(s);
                });
            }
        }
        Ok(spans)
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
        doc.add(&Key::DEFAULT, Sentences { spans: self.sentences.spans(text)? }, Provenance::by("turbo-eval"))
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
    fn the_rule_splitter_splits_sentences_and_is_named_by_its_configuration() {
        let a = Analysis { sentences: Sentencer::rules("en"), structure: Some(Syntax::Plain) };
        let text = "Dr. Smith arrived at 5 p.m. on Monday. Dogs bark at night.";
        let doc = a.analyze(text, &Stub::new(&["dogs"])).unwrap();
        let s: Vec<&str> = doc.get::<Sentences>().unwrap().spans.iter().map(|(_, s)| &text[s.range()]).collect();
        assert_eq!(s, ["Dr. Smith arrived at 5 p.m. on Monday.", "Dogs bark at night."]);
        assert!(doc.get::<Structure>().is_some());
        let names = a.names();
        let config = RuleSentenceDetector::new("en").config_hash();
        assert_eq!(
            names[0],
            format!("turbo-sentdetect:rules en version {} config {config:016x}", RuleSentenceDetector::VERSION)
        );
        assert_ne!(names[0], Sentencer::rules("de").name());
        assert_eq!(names[0], Sentencer::rules("en").name());
        assert_eq!(names[1..], ["turbo:structure plain", "opennlp:subwords"]);
    }

    #[test]
    fn the_sat_tier_splits_with_the_tiny_bundle_and_names_its_backend() {
        let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-sat-bundle");
        let a = Analysis {
            sentences: Sentencer::sat(&bundle, &SatOptions::default()).unwrap(),
            structure: Some(Syntax::Plain),
        };
        let text = "Cats purr all day. Dogs bark at night.";
        let doc = a.analyze(text, &Stub::new(&["cats"])).unwrap();
        let sentences = doc.get::<Sentences>().unwrap();
        assert!(!sentences.spans.is_empty());
        for (_, s) in sentences.spans.iter() {
            assert!(s.start < s.end && s.end as usize <= text.len());
            assert!(text.is_char_boundary(s.start as usize) && text.is_char_boundary(s.end as usize));
        }
        let names = a.names();
        assert!(names[0].contains("turboembed/tiny-sat"), "{}", names[0]);
        assert!(names[0].contains("backend cpu"), "{}", names[0]);
    }
}
