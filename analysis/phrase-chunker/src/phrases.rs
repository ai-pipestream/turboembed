//! `ChunkSample.phrasesAsSpanList`: the phrases a sequence of chunk tags
//! makes, as `ChunkerME.chunkAsSpans` returns them.

use crate::error::PhraseError;

/// A phrase: the tokens `start..end` of the sequence and its kind, the
/// text of the tag that started it after its first two UTF-16 units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phrase<'a> {
    pub start: usize,
    pub end: usize,
    pub kind: &'a str,
}

/// A tag's `substring(2)`: the kind of phrase it names. `None` where Java
/// throws (fewer than two units) or where the cut falls inside a
/// surrogate pair.
pub fn kind_of(tag: &str) -> Option<&str> {
    let mut units = 0;
    for (at, c) in tag.char_indices() {
        if units == 2 {
            return Some(&tag[at..]);
        }
        units += c.len_utf16();
    }
    (units == 2).then_some("")
}

/// `phrasesAsSpanList(sentence, tags, preds)` over the chunk tags of one
/// sentence, in order.
///
/// A tag starts a phrase when it begins with `B-`, or when it is neither
/// `O` nor `I-` followed by the kind of the phrase that is open; the
/// phrase's kind is the tag after its first two units, whatever those
/// are. So `I-NP` after `O` or at the start opens a noun phrase, `I-VP`
/// after `B-NP` closes the noun phrase and opens a verb phrase, and a tag
/// with another prefix, say `E-NP`, opens a phrase of its own. `I-<kind>`
/// continues the open phrase and `O` closes it. A phrase that is open at
/// the end ends with the sequence. The chunker's validator never lets an
/// `I-` tag follow `O` or another kind, so the tags `ChunkerME` returns
/// only make the plain `B-`/`I-` phrases; the rest matters for tags from
/// elsewhere.
pub fn phrases<'a>(tags: &[&'a str]) -> Result<Vec<Phrase<'a>>, PhraseError> {
    let mut out = Vec::new();
    let mut open: Option<(usize, &'a str)> = None;
    for (i, &tag) in tags.iter().enumerate() {
        let continues = |kind: &str| tag.strip_prefix("I-") == Some(kind);
        let starts = tag.starts_with("B-") || (!continues(open.map_or("", |(_, k)| k)) && tag != "O");
        if starts {
            if let Some((start, kind)) = open.take() {
                out.push(Phrase { start, end: i, kind });
            }
            let kind = kind_of(tag).ok_or_else(|| PhraseError { tag: tag.into(), index: i })?;
            open = Some((i, kind));
        } else if open.is_some_and(|(_, kind)| continues(kind)) {
            // The middle of the open phrase.
        } else if let Some((start, kind)) = open.take() {
            out.push(Phrase { start, end: i, kind });
        }
    }
    if let Some((start, kind)) = open {
        out.push(Phrase { start, end: tags.len(), kind });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(start: usize, end: usize, kind: &str) -> Phrase<'_> {
        Phrase { start, end, kind }
    }

    #[test]
    fn b_and_i_tags_make_phrases_and_o_ends_them() {
        let tags = ["B-NP", "I-NP", "B-VP", "O", "B-NP", "I-NP", "I-NP"];
        assert_eq!(phrases(&tags).unwrap(), [p(0, 2, "NP"), p(2, 3, "VP"), p(4, 7, "NP")]);
        assert_eq!(phrases(&[]).unwrap(), []);
        assert_eq!(phrases(&["O", "O"]).unwrap(), []);
        assert_eq!(phrases(&["B-NP"]).unwrap(), [p(0, 1, "NP")]);
    }

    #[test]
    fn an_inside_tag_of_another_kind_or_after_o_starts_a_phrase() {
        assert_eq!(phrases(&["I-NP", "I-NP"]).unwrap(), [p(0, 2, "NP")]);
        assert_eq!(phrases(&["O", "I-NP", "O"]).unwrap(), [p(1, 2, "NP")]);
        assert_eq!(phrases(&["B-NP", "I-VP", "I-VP"]).unwrap(), [p(0, 1, "NP"), p(1, 3, "VP")]);
        // Two B tags in a row are two phrases.
        assert_eq!(phrases(&["B-NP", "B-NP"]).unwrap(), [p(0, 1, "NP"), p(1, 2, "NP")]);
    }

    #[test]
    fn any_other_prefix_starts_a_phrase_named_after_its_first_two_units() {
        assert_eq!(phrases(&["E-NP", "E-NP", "I-NP"]).unwrap(), [p(0, 1, "NP"), p(1, 3, "NP")]);
        assert_eq!(phrases(&["XYNP", "I-NP"]).unwrap(), [p(0, 2, "NP")]);
        // An empty kind: `I-` continues it.
        assert_eq!(phrases(&["B-", "I-", "O"]).unwrap(), [p(0, 2, "")]);
        // Two units may be one character outside the Basic Multilingual Plane.
        assert_eq!(phrases(&["\u{1F600}NP"]).unwrap(), [p(0, 1, "NP")]);
        assert_eq!(phrases(&["\u{00E9}\u{00E9}NP"]).unwrap(), [p(0, 1, "NP")]);
    }

    #[test]
    fn a_short_tag_that_starts_a_phrase_is_refused_as_java_throws() {
        assert_eq!(phrases(&["B"]).unwrap_err(), PhraseError { tag: "B".into(), index: 0 });
        assert_eq!(phrases(&["O", "X"]).unwrap_err(), PhraseError { tag: "X".into(), index: 1 });
        assert_eq!(phrases(&[""]).unwrap_err(), PhraseError { tag: "".into(), index: 0 });
        // The cut would fall inside a surrogate pair.
        assert_eq!(phrases(&["a\u{1F600}"]).unwrap_err(), PhraseError { tag: "a\u{1F600}".into(), index: 0 });
        assert_eq!(kind_of("ab"), Some(""));
        assert_eq!(kind_of("a"), None);
    }
}
