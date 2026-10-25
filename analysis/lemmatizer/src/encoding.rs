//! Versioned lemma classes. Published OpenNLP files keep their original
//! decoder; newly trained models keep case and delimit edit indices.

use turbo_maxent::JavaString;

use crate::EditScriptError;
use crate::edit_script::{DecodeScratch, encode_with_case};

/// A separate factory prevents an OpenNLP reader from silently decoding
/// native classes with its one-digit, lowercase format.
pub const CASE_SENSITIVE_FACTORY: &str = "turbo.lemmatizer.CaseSensitiveV1";

/// The encoding belongs to the model, not an inference-time guess about
/// the token or its POS tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LemmaEncoding {
    /// OpenNLP's lowercase classes and one-digit decoder, for old models
    /// and exact JVM goldens.
    OpenNlp,
    /// A learned `K:` (keep) or `L:` (lowercase) base, followed by
    /// case-sensitive edits with `:` after each decimal UTF-16 index.
    CaseSensitiveV1,
}

impl LemmaEncoding {
    /// The edit class of a token and its annotated lemma.
    pub fn encode(self, token: &str, lemma: &str) -> JavaString {
        if self == Self::OpenNlp {
            let units = encode_with_case(token, lemma, false);
            return if units.is_empty() { JavaString::new("_") } else { JavaString::from_utf16(&units) };
        }
        let mut lower = String::new();
        turbo_unicode::java::to_lowercase(lemma, &mut lower);
        // The classifier learns whether to keep the token's case or to
        // lowercase it. Identity classes then generalize to unseen names
        // and sentence-initial common words without a POS heuristic.
        let keep = lower != lemma;
        let base = if keep {
            token
        } else {
            lower.clear();
            turbo_unicode::java::to_lowercase(token, &mut lower);
            &lower
        };
        let mut units = vec![u16::from(if keep { b'K' } else { b'L' }), u16::from(b':')];
        units.extend(encode_with_case(base, lemma, true));
        JavaString::from_utf16(&units)
    }

    /// Applies a class according to the model's format. An empty lemma
    /// is `_`, as in the lemma layer's existing convention.
    pub fn decode(self, token: &str, class: &JavaString) -> Result<String, EditScriptError> {
        DecodeScratch::default().decode_with_case(token, class, self == Self::CaseSensitiveV1).map(str::to_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annotated_names_and_labels_keep_case_and_long_indices_round_trip() {
        // Names from UD Italian MarkIT r2.16 train, sent_id 714 and 725;
        // the punctuation label is MarkIT's. The longer words exercise
        // index widths that the old OpenNLP decoder cannot represent.
        let native = LemmaEncoding::CaseSensitiveV1;
        for (token, lemma) in [
            ("Bobbio", "Bobbio"),
            ("Viroli", "Viroli"),
            ("Roma-Bari", "Roma-Bari"),
            ("Romeo", "Romeo"),
            (".", "[PUNCT]"),
            ("The", "the"),
            ("Ultramicroscopic", "ultramicroscopic"),
            ("ultramicroscopic", "Ultramicroscopic"),
            ("\u{10400}abc", "\u{10428}abc"),
            ("cafés", "café"),
            ("", "Romeo"),
            ("Romeo", ""),
        ] {
            let class = native.encode(token, lemma);
            assert_eq!(native.decode(token, &class).unwrap(), if lemma.is_empty() { "_" } else { lemma });
        }
        assert_eq!(native.decode("Roma-Bari", &native.encode("Roma-Bari", "Roma-Bari")).unwrap(), "Roma-Bari");
        let old = LemmaEncoding::OpenNlp;
        assert_eq!(old.decode("Roma-Bari", &old.encode("Roma-Bari", "Roma-Bari")).unwrap(), "roma-bari");
        assert_eq!(old.encode("Ultramicroscopic", "ultramicroscopic"), "O");
        assert_ne!(native.encode("Ultramicroscopic", "ultramicroscopic"), "O");
    }

    #[test]
    fn malformed_native_classes_refuse_instead_of_using_the_legacy_parser() {
        for script in ["D0s", "K:R:ab", "K:I0:", "K:R0:a", "K:X0:a", "K:D999999999999999999999999999999999999999:s"] {
            assert!(LemmaEncoding::CaseSensitiveV1.decode("Bobbio", &JavaString::new(script)).is_err(), "{script}");
        }
    }
}
