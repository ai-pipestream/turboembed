//! spaCy's `Matcher` tests (spaCy, MIT: `spacy/tests/matcher/
//! test_matcher_logic.py` and `test_matcher_api.py`), ported with the same
//! expected matches. Each test names the one it ports. spaCy's dict
//! patterns are written with `Tok`: `{"ORTH": x}` is `Tok::text(x)`,
//! `{"LOWER": x}` is `Tok::lower(x)`, `{"TAG": x}` and `{"POS": x}` are
//! `Tok::attr(Attr::Pos, x)`, a custom extension attribute is
//! `Tok::text_where`, and `"OP"` is `!`, `.opt()`, `.star()`,
//! `.plus()` or `.repeat()`. Documents built with spaCy's tokenizer are
//! built here from the tokens it makes.

#[path = "spacy_matcher/alignments.rs"]
mod alignments;
#[path = "spacy_matcher/api.rs"]
mod api;
mod common;
#[path = "spacy_matcher/logic.rs"]
mod logic;

use common::Doc;
use turbo_analysis::IxRange;
use turbo_matcher::{Attr, Cmp, Flag, Greedy, Iob, MatchError, Matcher, PatternError, Tok};

fn matcher(rules: &[(&str, Vec<Vec<Tok>>)]) -> Matcher {
    let mut m = Matcher::new();
    for (label, patterns) in rules {
        for p in patterns {
            m.add(label, p).unwrap();
        }
    }
    m
}

fn spans(found: &[(String, usize, usize)]) -> Vec<(usize, usize)> {
    found.iter().map(|m| (m.1, m.2)).collect()
}

/// A pattern from spaCy's test notation: `a`, `a+`, `a*`, `a?`, `a{2,3}`.
fn notation(pattern: &str) -> Vec<Tok> {
    pattern
        .split(' ')
        .map(|part| {
            let tok = Tok::text(&part[..1]);
            match &part[1..] {
                "" => tok,
                "+" => tok.plus(),
                "*" => tok.star(),
                "?" => tok.opt(),
                range => {
                    let inner = range.trim_start_matches('{').trim_end_matches('}');
                    let (min, max) = match inner.split_once(',') {
                        None => {
                            let n = inner.parse().unwrap();
                            (n, Some(n))
                        }
                        Some((a, b)) => (a.parse().unwrap_or(0), b.parse().ok()),
                    };
                    tok.repeat(min, max)
                }
            }
        })
        .collect()
}
