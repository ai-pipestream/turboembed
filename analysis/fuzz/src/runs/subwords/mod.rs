//! The WordPiece run: the port's encoder with the tiny BERT bundle, whose
//! vocabulary is the one the JVM's `WordpieceEncoder` ran with.
//!
//! The port follows upstream `tokenizers`; `WordpieceEncoder` follows the
//! reference BERT, and keeps source offsets its own way. The differences
//! are documented and tested in `analysis/embed/tests/goldens.rs`. To tell
//! a documented difference from a divergence on any text, this module
//! models how `WordpieceEncoder` maps its normalized text back to the
//! source (control characters dropped, whitespace made spaces, ideographs
//! and punctuation isolated, simple lowercase with final sigma, accents
//! stripped with the mark's end given to the character before it, and
//! every piece given the whole run when canonical reordering crossed
//! characters) and lays the port's pieces over that mapping. A golden is
//! a documented difference when its spans are exactly what the model
//! gives for the port's pieces.

mod model;
#[cfg(test)]
mod tests;

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_embed::SubwordAnnotator;
use turbo_analysis_layers::Subwords;
use turbo_goldens::expect::{only_stripped, without_trailing_marks};
use turbo_goldens::{Golden, Tolerance, matches_golden};

use self::model::{
    MAX_WORD_CODE_POINTS, PortSpan, SPECIAL_TOKENS, Unit, expected_port_spans, jvm_only_cjk, map, port_span_misfit,
    port_spans_fit, sigma_words, special_occurrences,
};
use super::{Outcome, Run, RustError, analyze};
use crate::resources::{Resources, Unavailable};

/// The run name of the vocabulary the goldens were made with.
pub(crate) const VOCABULARY: &str = "all-minilm-l6-v2-vocab";

pub(crate) struct WordPiece {
    analyzer: Analyzer,
    /// The length in characters of each piece of the vocabulary, the
    /// continuation prefix left out, by id.
    lengths: Vec<usize>,
    unknown: u32,
    /// The special token strings the vocabulary has, with their ids.
    special: Vec<(&'static str, u32)>,
}

impl WordPiece {
    pub(crate) fn open(res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        Ok(Box::new(WordPiece::new(res)?))
    }

    fn new(res: &Resources) -> Result<WordPiece, Unavailable> {
        let dir = res.bert_bundle()?;
        let annotator =
            SubwordAnnotator::open(&dir).map_err(|e| Unavailable::Broken(format!("{}: {e}", dir.display())))?;
        let analyzer = Analyzer::builder().add(annotator).build().map_err(|e| Unavailable::Broken(e.to_string()))?;
        let vocabulary = res.fixture(&format!("{VOCABULARY}.txt"))?;
        let text = std::fs::read_to_string(&vocabulary)
            .map_err(|e| Unavailable::Broken(format!("{}: {e}", vocabulary.display())))?;
        let pieces: Vec<&str> = text.strip_suffix('\n').unwrap_or(&text).split('\n').collect();
        let unknown = pieces
            .iter()
            .position(|p| *p == "[UNK]")
            .ok_or_else(|| Unavailable::Broken(format!("{}: no [UNK] piece", vocabulary.display())))?
            as u32;
        let lengths = pieces.iter().map(|p| p.strip_prefix("##").unwrap_or(p).chars().count()).collect();
        let special = SPECIAL_TOKENS
            .iter()
            .filter_map(|t| pieces.iter().position(|p| p == t).map(|id| (*t, id as u32)))
            .collect();
        Ok(WordPiece { analyzer, lengths, unknown, special })
    }
}

/// A piece: its id and byte span.
pub(super) type Piece = (u32, [u32; 2]);

/// The golden's pieces.
fn jvm_pieces(golden: &Golden) -> Option<Vec<Piece>> {
    let l = golden.layer("opennlp:subwords")?;
    let (start, end, id) = (l.u32s("start")?, l.u32s("end")?, l.u32s("id")?);
    Some((0..l.len()).map(|i| (id[i], [start[i], end[i]])).collect())
}

/// The pieces of `doc`.
fn rust_pieces(doc: &Document) -> Vec<Piece> {
    doc.get::<Subwords>().map_or_else(Vec::new, |sub| {
        sub.ids.iter().zip(sub.spans.iter()).map(|(&id, (_, s))| (id as u32, [s.start, s.end])).collect()
    })
}

impl WordPiece {
    /// The spans `WordpieceEncoder` gives pieces with the ids `ids` over
    /// the mapped text, word by word: a word over the length limit, or
    /// one the port made unknown, is one piece over the word; otherwise
    /// each piece covers as many code points as its vocabulary entry has.
    /// None when the ids do not fit the words.
    fn jvm_spans(&self, units: &[Unit], ids: &[u32]) -> Option<Vec<[u32; 2]>> {
        let mut spans = Vec::with_capacity(ids.len());
        let mut k = 0;
        let mut from = 0;
        while from < units.len() {
            if units[from].c == ' ' {
                from += 1;
                continue;
            }
            let to = units[from..].iter().position(|u| u.c == ' ').map_or(units.len(), |i| from + i);
            let word = &units[from..to];
            from = to;
            let &next = ids.get(k)?;
            if word.len() > MAX_WORD_CODE_POINTS || next == self.unknown {
                if word.len() > MAX_WORD_CODE_POINTS && next != self.unknown {
                    return None;
                }
                spans.push([word[0].start, word[word.len() - 1].end]);
                k += 1;
                continue;
            }
            let mut at = 0;
            while at < word.len() {
                let len = *self.lengths.get(*ids.get(k)? as usize)?;
                if len == 0 || at + len > word.len() {
                    return None;
                }
                spans.push([word[at].start, word[at + len - 1].end]);
                at += len;
                k += 1;
            }
        }
        (k == ids.len()).then_some(spans)
    }

    /// Which documented difference of `analysis/embed/tests/goldens.rs`
    /// explains `jvm` against `rust`, or why none does. Both sides are checked:
    /// the JVM's spans must be what the model gives for its ids, and the
    /// port's pieces must be what upstream `tokenizers` gives for the same
    /// text, as [`expected_port_spans`] derives them from the JVM's.
    fn documented(&self, text: &str, jvm: &[Piece], rust: &[Piece]) -> Result<String, String> {
        let mapped = map(text);
        // With final sigma written as the JVM lowercases it, the port's ids
        // are the JVM's: the port's pieces on that text stand in for its
        // pieces on `text` from here on, once the two agree on every word
        // without a capital sigma. The words with one may split differently,
        // as the vocabulary has pieces for one sigma and not the other.
        let final_sigma_pieces;
        let mut rust = rust;
        if mapped.as_jvm != text {
            final_sigma_pieces = analyze(&self.analyzer, &mapped.as_jvm)
                .map(|d| rust_pieces(&d))
                .map_err(|e| format!("the port fails on the text with final sigma written: {e}"))?;
            let words = sigma_words(text, &mapped.units);
            let outside = |pieces: &[Piece]| -> Vec<Piece> {
                pieces.iter().copied().filter(|&(_, [a, b])| !words.iter().any(|&[s, e]| a >= s && b <= e)).collect()
            };
            let (on_text, with_final) = (outside(rust), outside(&final_sigma_pieces));
            if on_text != with_final {
                let i = (0..on_text.len().max(with_final.len()))
                    .find(|&i| on_text.get(i) != with_final.get(i))
                    .unwrap_or(0);
                return Err(format!(
                    "outside the words with a capital sigma, the port's piece {i} is {:?} on the text and {:?} with final sigma written",
                    on_text.get(i),
                    with_final.get(i)
                ));
            }
            // Inside such a word the port's pieces still cover the word.
            for &word in &words {
                let inside: Vec<Piece> =
                    rust.iter().copied().filter(|&(_, [a, b])| a >= word[0] && b <= word[1]).collect();
                let expected = vec![PortSpan::Within(word); inside.len()];
                if inside.is_empty() || !port_spans_fit(text, &expected, &inside) {
                    return Err(format!(
                        "the port's pieces {inside:?} do not cover the word with a capital sigma at {}..{}",
                        word[0], word[1]
                    ));
                }
            }
            rust = &final_sigma_pieces;
        }
        // The special token strings: one piece of the token's id here, text
        // pieces inside the string on the JVM; the rest is compared as usual.
        let occurrences = special_occurrences(text, &self.special);
        let mut rust_outside: Vec<Piece> = Vec::with_capacity(rust.len());
        for &piece in rust {
            if let Some(&(a, b, id)) = occurrences.iter().find(|&&(a, b, _)| piece.1[0] >= a && piece.1[1] <= b) {
                if piece != (id, [a, b]) {
                    return Err(format!(
                        "the port's piece {piece:?} inside the special token at {a}..{b} is not the token"
                    ));
                }
            } else {
                rust_outside.push(piece);
            }
        }
        if let Some(&(a, b, id)) = occurrences.iter().find(|&&(a, b, id)| !rust.contains(&(id, [a, b]))) {
            return Err(format!("the port has no piece {id} over the special token at {a}..{b}"));
        }
        let mut jvm_outside: Vec<Piece> = Vec::with_capacity(jvm.len());
        for &piece in jvm {
            let [a, b] = without_trailing_marks(text, piece.1);
            let inside = occurrences.iter().any(|&(s, e, _)| a >= s && b <= e);
            if !inside {
                if occurrences.iter().any(|&(s, e, _)| a < e && b > s) {
                    return Err(format!("the JVM's piece at {a}..{b} crosses a special token's string"));
                }
                jvm_outside.push(piece);
            }
        }
        let cjk = text.chars().any(jvm_only_cjk);
        if cjk && rust_outside.iter().map(|p| p.0).ne(jvm_outside.iter().map(|p| p.0)) {
            // The ideographs the JVM isolates and the port does not: each is
            // a word of its own, an unknown piece, where the port has one
            // unknown piece over the whole word. Split those unknown pieces
            // and continue through the normal JVM and port span checks so CJK
            // differences compose with special tokens and stripped marks.
            let mut split = Vec::with_capacity(rust_outside.len());
            for &(id, [a, b]) in &rust_outside {
                let piece = &text[a as usize..b as usize];
                if id == self.unknown && piece.chars().any(jvm_only_cjk) && piece.chars().count() > 1 {
                    let mut at = a;
                    for c in piece.chars() {
                        let next = at + c.len_utf8() as u32;
                        if !only_stripped(text, [at, next]) {
                            split.push((id, [at, next]));
                        }
                        at = next;
                    }
                } else {
                    split.push((id, [a, b]));
                }
            }
            rust_outside = split;
        }
        let ids: Vec<u32> = rust_outside.iter().map(|p| p.0).collect();
        let jvm_ids: Vec<u32> = jvm_outside.iter().map(|p| p.0).collect();
        if ids != jvm_ids {
            let i = (0..ids.len().max(jvm_ids.len())).find(|&i| ids.get(i) != jvm_ids.get(i)).unwrap_or(0);
            return Err(format!(
                "the ids differ at piece {i}: {:?} here, {:?} on the JVM ({} and {} pieces)",
                rust_outside.get(i),
                jvm_outside.get(i),
                ids.len(),
                jvm_ids.len()
            ));
        }
        // The JVM's side: its spans are the model's for its ids.
        let all_ids: Vec<u32> = jvm.iter().map(|p| p.0).collect();
        let spans = self
            .jvm_spans(&mapped.units, &all_ids)
            .ok_or_else(|| "the JVM's ids do not fit the words of the modelled WordpieceEncoder text".to_string())?;
        if let Some(i) = (0..spans.len().max(jvm.len())).find(|&i| spans.get(i) != jvm.get(i).map(|j| &j.1)) {
            return Err(format!(
                "the JVM's span of piece {i} is {:?} where the model of WordpieceEncoder gives {:?}",
                jvm.get(i).map(|j| j.1),
                spans.get(i)
            ));
        }
        // The port's side: its spans are the JVM's with the documented
        // changes, piece by piece.
        let expected = expected_port_spans(text, &mapped, &jvm_outside);
        if let Some(i) = port_span_misfit(text, &expected, &rust_outside) {
            let inside = match expected.get(i) {
                Some(PortSpan::Within([s, e])) => {
                    let pieces: Vec<[u32; 2]> =
                        rust_outside.iter().map(|r| r.1).filter(|&[a, b]| a >= *s && b <= *e).collect();
                    format!("; the port's pieces inside the run are {pieces:?}")
                }
                _ => String::new(),
            };
            return Err(format!(
                "the port's span of piece {i} is {:?} where the JVM's gives {:?} ({} and {} pieces){inside}",
                rust_outside.get(i).map(|r| r.1),
                expected.get(i),
                rust_outside.len(),
                expected.len()
            ));
        }
        let mut why = Vec::new();
        if !occurrences.is_empty() {
            why.push("a special token's string is that token here and ordinary text on the JVM, as upstream matches added tokens before normalizing (special_token_strings_are_special_tokens_here_only)");
        }
        if mapped.used.sigma {
            why.push("a word final capital sigma lowercases to final sigma on the JVM and to sigma here (a_word_final_capital_sigma_is_final_sigma_on_the_jvm_only)");
        }
        if mapped.used.stripped {
            why.push("WordpieceEncoder runs a piece's span over the nonspacing marks accent stripping removed after it, and over the controls it dropped before them (a_stripped_mark_is_in_the_jvm_span_and_not_in_ours)");
        }
        if !mapped.used.reordered.is_empty() {
            why.push("marks out of canonical order: WordpieceEncoder gives every piece of the run the whole run's span (marks_out_of_order_give_every_jvm_piece_the_whole_run)");
        }
        if cjk {
            why.push("the ideographs U+2B820 to U+2B91F are CJK on the JVM and not upstream (the_ideographs_from_u2b820_are_words_on_the_jvm_only)");
        }
        // The model gives the JVM's spans without any documented rule: a
        // difference nobody has written down yet.
        if why.is_empty() {
            return Err("the spans differ and no documented difference is in play".to_string());
        }
        Ok(format!("{} (analysis/embed/tests/goldens.rs)", why.join("; ")))
    }
}

impl Run for WordPiece {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    /// Exactly, or as one of the differences `analysis/embed/tests/goldens.rs`
    /// documents and tests: a piece's span run over stripped marks or over a
    /// run with marks out of order, final sigma, the CJK range, and special
    /// token strings. The port follows upstream `tokenizers` in each.
    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        let mismatch = match matches_golden::<Subwords>(doc, golden, Tolerance::exact()) {
            Ok(()) => return Outcome::Match,
            Err(m) => m,
        };
        let Some(jvm) = jvm_pieces(golden) else { return Outcome::Divergence(mismatch.to_string()) };
        match self.documented(doc.source(), &jvm, &rust_pieces(doc)) {
            Ok(why) => Outcome::Documented(why),
            Err(reason) => Outcome::Divergence(format!("{mismatch}; {reason}")),
        }
    }
}
