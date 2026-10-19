//! `EnglishContractionsWithPos`: contraction expansion that resolves `'s`
//! and `'d` from part-of-speech tags.

use turbo_analysis::{Annotator, Error, Refs, View, ViewBuilder};
use turbo_analysis_layers::{PosTags, Tokens};

use super::rules::{Ambiguous, Reading};
use super::scan;

/// [`EnglishContractions`](super::EnglishContractions), and `'s` and `'d`
/// resolved by the Penn Treebank tags of the tokens:
///
/// - `'s` becomes ` has` when the token holding the apostrophe is tagged
///   `VBZ` and the next token `VBN` (`he's gone`), and ` is` when the
///   token is tagged `VBZ` and the next token is not `VBN` (`he's here`).
///   Any other tag (`POS` for a possessive) leaves it alone.
/// - `'d` becomes ` would` when the next token is tagged `VB` (`I'd go`)
///   and ` had` when it is tagged `VBN` (`I'd gone`). Anything else leaves
///   it alone.
///
/// The token holding the apostrophe is `'s` itself under Penn Treebank
/// tokenization, or the whole word under a tokenizer that keeps it whole;
/// either works. The same word-boundary and ASCII-letter rules apply as
/// for the plain normalizer.
///
/// It reads the `View`, and `Tokens` and `PosTags` when the document has
/// them; without both it writes what the plain normalizer writes. The
/// tokens must be in the view it reads.
///
/// OpenNLP has no counterpart: its normalizer leaves every `'s` and `'d`
/// alone.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EnglishContractionsWithPos;

/// The tokens and their tags, as the resolver reads them.
struct Tagged<'d> {
    tokens: &'d Tokens,
    tags: &'d PosTags,
}

impl Tagged<'_> {
    /// The index of the token whose span holds byte `at`.
    fn token_at(&self, at: usize) -> Option<usize> {
        let (start, end) = self.tokens.spans.columns();
        let at = at as u32;
        let k = end.partition_point(|&e| e <= at);
        (k < start.len() && start[k] <= at).then_some(k)
    }

    fn label(&self, k: usize) -> Option<&str> {
        self.tags
            .tags
            .get(k)
            .and_then(|&t| (usize::from(t.0) < self.tags.tagset.len()).then(|| self.tags.tagset.label(t)))
    }

    fn resolve(&self, apostrophe: usize, which: Ambiguous) -> Option<Reading> {
        let k = self.token_at(apostrophe)?;
        let next = self.label(k + 1);
        match which {
            Ambiguous::S if self.label(k) == Some("VBZ") => {
                Some(if next == Some("VBN") { Reading::Has } else { Reading::Is })
            }
            Ambiguous::S => None,
            Ambiguous::D => match next {
                Some("VB") => Some(Reading::Would),
                Some("VBN") => Some(Reading::Had),
                _ => None,
            },
        }
    }
}

impl Annotator for EnglishContractionsWithPos {
    const ID: &'static str = "turbo:normalize:english-contractions-pos";
    type Reads = (View, Option<Tokens>, Option<PosTags>);
    type Writes = (View,);

    fn annotate(&self, (src, tokens, tags): Refs<'_, Self>) -> Result<(View,), Error> {
        let text = src.text();
        let mut out = ViewBuilder::new(text);
        match (tokens, tags) {
            (Some(tokens), Some(tags)) => {
                if tokens.spans.view() != src.id() {
                    return Err(Error::Annotator {
                        annotator: Self::ID,
                        detail: format!(
                            "tokens are in view {}, the text in view {}",
                            tokens.spans.view().0,
                            src.id().0
                        ),
                    });
                }
                if tags.tags.len() != tokens.spans.len() {
                    return Err(Error::Annotator {
                        annotator: Self::ID,
                        detail: format!("received {} POS tags for {} tokens", tags.tags.len(), tokens.spans.len()),
                    });
                }
                let mut prev_end = 0u32;
                for (i, span) in tokens.spans.iter() {
                    let (start, end) = (span.start as usize, span.end as usize);
                    if span.start > span.end
                        || end > text.len()
                        || !text.is_char_boundary(start)
                        || !text.is_char_boundary(end)
                    {
                        return Err(Error::Annotator {
                            annotator: Self::ID,
                            detail: format!(
                                "token {} span {}..{} is out of bounds for text of {} bytes",
                                i.get(),
                                span.start,
                                span.end,
                                text.len()
                            ),
                        });
                    }
                    if i.get() > 0 && span.start < prev_end {
                        return Err(Error::Annotator {
                            annotator: Self::ID,
                            detail: format!(
                                "token {} span {}..{} is not in non-overlapping text order",
                                i.get(),
                                span.start,
                                span.end
                            ),
                        });
                    }
                    prev_end = span.end;
                }
                for (i, &tag) in tags.tags.iter().enumerate() {
                    if usize::from(tag.0) >= tags.tagset.len() {
                        return Err(Error::Annotator {
                            annotator: Self::ID,
                            detail: format!(
                                "POS tag {} has tag id {} outside tagset of length {}",
                                i,
                                tag.0,
                                tags.tagset.len()
                            ),
                        });
                    }
                }
                let tagged = Tagged { tokens, tags };
                scan::expand(text, &mut out, Some(&mut |at, which| tagged.resolve(at, which)));
            }
            _ => scan::expand(text, &mut out, None),
        }
        let (normalized, alignment) = out.finish();
        Ok((View::derived(src, normalized, alignment),))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Document, Span, Spans, Tag, TagSet};

    #[test]
    fn rejects_mismatched_unordered_or_out_of_bounds_inputs() {
        let doc = Document::new("he's gone");
        let view = doc.get::<View>().unwrap();
        let tagset = TagSet::new("ptb", &["PRP", "VBZ", "VBN"]);
        let vbz = tagset.tag("VBZ").unwrap();
        let vbn = tagset.tag("VBN").unwrap();

        let mut spans = Spans::new(view.id());
        spans.push(Span::new(0, 4));
        spans.push(Span::new(5, 9));

        // Mismatched tag count.
        let err = EnglishContractionsWithPos
            .annotate((
                view,
                Some(&Tokens { spans: spans.clone(), sentence: None }),
                Some(&PosTags { tagset: tagset.clone(), tags: vec![vbz], prob: None }),
            ))
            .expect_err("mismatched tag count must fail");
        assert!(err.to_string().contains("received 1 POS tags for 2 tokens"));

        // Out-of-bounds Tag id.
        let err = EnglishContractionsWithPos
            .annotate((
                view,
                Some(&Tokens { spans: spans.clone(), sentence: None }),
                Some(&PosTags { tagset: tagset.clone(), tags: vec![vbz, Tag(99)], prob: None }),
            ))
            .expect_err("out-of-bounds tag id must fail");
        assert!(err.to_string().contains("outside tagset"));

        // Overlapping / unordered token spans.
        let mut unordered = Spans::new(view.id());
        unordered.push(Span::new(5, 9));
        unordered.push(Span::new(0, 4));
        let err = EnglishContractionsWithPos
            .annotate((
                view,
                Some(&Tokens { spans: unordered, sentence: None }),
                Some(&PosTags { tagset: tagset.clone(), tags: vec![vbn, vbz], prob: None }),
            ))
            .expect_err("unordered token spans must fail");
        assert!(err.to_string().contains("non-overlapping text order"));

        // Out-of-bounds token span.
        let mut oob = Spans::new(view.id());
        oob.push(Span::new(0, 99));
        let err = EnglishContractionsWithPos
            .annotate((
                view,
                Some(&Tokens { spans: oob, sentence: None }),
                Some(&PosTags { tagset, tags: vec![vbz], prob: None }),
            ))
            .expect_err("out-of-bounds token span must fail");
        assert!(err.to_string().contains("out of bounds"));
    }
}
