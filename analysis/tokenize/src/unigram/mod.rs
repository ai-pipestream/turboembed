//! Frequency segmentation for Chinese, with a trained unknown-word model.
//!
//! [`UnigramMode::Chinese`] groups ASCII alphanumerics and applies jieba's
//! BMES model to unresolved Han runs. The default [`UnigramMode::Dictionary`] groups
//! ASCII without the HMM. [`UnigramMode::OpenNlp`] keeps the original
//! dictionary-only segmentation and tie rules for JVM goldens.
//! All modes write byte spans into the unchanged input view.

mod hmm;
mod lexicon;
mod segmenter;
mod trie;

pub use lexicon::{DEFAULT_MAX_ENTRIES, LexiconError, UnigramLexicon};
pub use segmenter::{Scratch, UnigramMode, UnigramSegmenter};

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Sentences, Tokens};

use crate::unit::tokenize_units;
use crate::{WHITESPACE_DEFINITION, fnv1a};

/// The OpenNLP class the segmenter ports, the id of its lexicon.
pub const COMPONENT_NAME: &str = "UnigramSegmenter";

impl Annotator for UnigramSegmenter {
    const ID: &'static str = "turbo-tokenize:unigram";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens,);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        let mut scratch = Scratch::new();
        let tokens =
            tokenize_units(Self::ID, view, sentences, |text, _, out| self.segment_with(text, &mut scratch, out))?;
        Ok((tokens,))
    }

    /// The lexicon file: `opennlp:UnigramSegmenter` as the id, no
    /// revision (a lexicon file carries none), and the file's SHA-256.
    fn model(&self) -> Option<ModelRef> {
        Some(ModelRef {
            id: format!("opennlp:{COMPONENT_NAME}"),
            revision: String::new(),
            sha256: self.lexicon.sha256().to_hex(),
        })
    }

    /// The whitespace definition, once for the text and once for the
    /// lexicon, and the segmentation policy. Chinese mode also includes the
    /// packed HMM SHA-256.
    fn config_hash(&self) -> u64 {
        let whitespace = [WHITESPACE_DEFINITION, WHITESPACE_DEFINITION];
        if self.mode == UnigramMode::OpenNlp {
            return fnv1a(&whitespace);
        }
        let policy = match self.mode {
            UnigramMode::Chinese => hmm::SHA256.as_bytes(),
            UnigramMode::Dictionary => b"ascii-alphanumeric-v1",
            UnigramMode::OpenNlp => unreachable!(),
        };
        let mut bytes = whitespace.to_vec();
        bytes.extend_from_slice(b"unigram-policy-v1:");
        bytes.extend_from_slice(policy);
        fnv1a(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Document, Span, Spans, ViewId};

    fn segmenter() -> UnigramSegmenter {
        let lexicon = UnigramLexicon::parse("\u{5317}\u{4eac} 3\n\u{5927}\u{5b66} 2\n".as_bytes()).unwrap();
        UnigramSegmenter::new(lexicon).with_mode(UnigramMode::OpenNlp)
    }

    #[test]
    fn writes_tokens_per_sentence() {
        let doc = Document::new("\u{5317}\u{4eac}\u{5927}\u{5b66}\u{3002} \u{5927}\u{5b66}");
        let view = doc.get::<View>().unwrap();
        let s = segmenter();
        let (tokens,) = s.annotate((view, None)).unwrap();
        let got: Vec<(u32, u32)> = tokens.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
        assert_eq!(got, [(0, 6), (6, 12), (12, 15), (16, 22)]);
        assert!(tokens.sentence.is_none());

        let mut spans = Spans::new(view.id());
        spans.push(Span::new(0, 15));
        spans.push(Span::new(16, 22));
        let (tokens,) = s.annotate((view, Some(&Sentences { spans }))).unwrap();
        let got: Vec<(u32, u32)> = tokens.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
        assert_eq!(got, [(0, 6), (6, 12), (12, 15), (16, 22)]);
        let column: Vec<usize> = tokens.sentence.unwrap().iter().map(|i| i.get()).collect();
        assert_eq!(column, [0, 0, 0, 1]);
    }

    #[test]
    fn sentences_in_another_view_are_refused() {
        let doc = Document::new("ab");
        let sentences = Sentences { spans: Spans::new(ViewId(3)) };
        let s = segmenter();
        let e = s.annotate((doc.get::<View>().unwrap(), Some(&sentences))).err().unwrap();
        assert_eq!(
            e,
            Error::Annotator {
                annotator: "turbo-tokenize:unigram",
                detail: "the sentences are in view 3, not in view 0 that is tokenized".into()
            }
        );
    }

    #[test]
    fn the_model_is_the_lexicon_file() {
        let s = segmenter();
        let m = s.model().unwrap();
        assert_eq!(m.id, "opennlp:UnigramSegmenter");
        assert_eq!(m.sha256, turbo_maxent::sha256("\u{5317}\u{4eac} 3\n\u{5927}\u{5b66} 2\n".as_bytes()).to_hex());
    }

    #[test]
    fn config_hashes_differ_per_mode() {
        let modes = [UnigramMode::OpenNlp, UnigramMode::Dictionary, UnigramMode::Chinese];
        let hashes: Vec<u64> = modes.iter().map(|&m| segmenter().with_mode(m).config_hash()).collect();
        for (i, a) in hashes.iter().enumerate() {
            assert!(hashes[i + 1..].iter().all(|b| a != b));
        }
        // Pinned, so a recorded provenance hash stays valid.
        assert_eq!(hashes[0], fnv1a(&[1, 1]));
    }
}
