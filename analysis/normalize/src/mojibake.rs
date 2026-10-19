//! Repairs mojibake: text that was encoded in one encoding and decoded
//! in another.

use turbo_analysis::ViewBuilder;

use crate::Normalizer;

/// Undoes mojibake with the port of ftfy's encoding fix in
/// `turbo_encoding::repair` (ftfy by Robyn Speer, Apache-2.0): "cafÃ©"
/// becomes "café", "donâ€™t" becomes "don’t". The text is
/// repaired one line at a time, as ftfy's `fix_text` does, so a line of
/// mojibake is fixed even when the line beside it is fine. Each fix is
/// one edit over the source characters it came from; text that looks
/// fine is kept unchanged.
///
/// The decision follows ftfy's badness heuristic, so it can be wrong in
/// both directions: rare mojibake goes unfixed, and rare legitimate text
/// that looks like mojibake is changed. Run it on text known to have
/// passed through a wrong decode, or check its changes.
///
/// No OpenNLP counterpart.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Mojibake;

impl Normalizer for Mojibake {
    const ID: &'static str = "turbo:normalize:mojibake";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        turbo_encoding::repair::repair_lines(text, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize;
    use turbo_analysis::Span;

    #[test]
    fn each_fix_is_one_edit_over_its_source() {
        let src = "plain line\ncafÃ© and naÃ¯ve\n";
        let (out, alignment) = normalize(&Mojibake, src);
        assert_eq!(out, "plain line\ncafé and naïve\n");
        let edits = alignment.edits();
        assert_eq!(edits.len(), 2);
        let e = src.find("Ã©").unwrap() as u32;
        assert_eq!(edits[0].parent, Span::new(e, e + 4));
        assert_eq!(&out[edits[0].child.range()], "é");
    }

    #[test]
    fn the_documented_examples_hold() {
        assert_eq!(normalize(&Mojibake, "cafÃ©").0, "café");
        assert_eq!(normalize(&Mojibake, "donâ€™t").0, "don’t");
    }

    #[test]
    fn a_fix_maps_through_the_document_to_the_file_bytes() {
        use turbo_analysis::{Key, Provenance, View};
        use turbo_encoding::{Encoding, open};
        // A UTF-8 file opened as windows-1252: view 0 is mojibake.
        let bytes = "café au lait".as_bytes();
        let mut doc = open(bytes, Some(Encoding::Windows1252)).unwrap();
        assert_eq!(doc.source(), "cafÃ© au lait");
        let (text, alignment) = normalize(&Mojibake, doc.source());
        let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
        doc.add(&Key::<View>::named("repaired"), view, Provenance::by(Mojibake::ID)).unwrap();
        let v = doc.get_key(&Key::<View>::named("repaired")).unwrap().id();
        assert_eq!(doc.text(v), "café au lait");
        // The repaired "é" is the two bytes of its UTF-8 in the file, by
        // way of the four bytes of "Ã©" in view 0.
        assert_eq!(doc.to_source(v, Span::new(3, 5)), Span::new(3, 7));
        assert_eq!(doc.to_source_bytes(v, Span::new(3, 5)), Span::new(3, 5));
        assert_eq!(doc.to_source_bytes(v, Span::new(6, 8)), Span::new(6, 8));
        assert_eq!(doc.from_source_bytes(v, Span::new(0, 5)), Span::new(0, 5));
    }

    #[test]
    fn fine_text_is_left_alone() {
        for text in ["", "ASCII only", "café crème", "日本語のテキスト", "Ελληνικά"] {
            let (out, alignment) = normalize(&Mojibake, text);
            assert_eq!(out, text);
            assert!(alignment.edits().is_empty(), "{text}");
        }
    }
}
