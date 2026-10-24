//! Existing exact/set/quantifier path over real CoNLL-U token text.
//! `token_bench CORPUS.conllu [ITERATIONS]` prints median ns and hit count.
use std::{hint::black_box, time::Instant};
use turbo_analysis::{Span, Spans, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_matcher::{Attr, Flag, Input, Matcher, Tok};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).unwrap();
    let iterations: usize = args.get(2).map_or(10, |s| s.parse().unwrap());
    assert!(iterations > 0);
    let mut text = String::new();
    let mut spans = Spans::new(ViewId::SOURCE);
    for line in raw.lines().filter(|l| !l.starts_with('#')) {
        let mut fields = line.split('\t');
        if fields.next().is_none_or(|s| s.parse::<u32>().is_err()) {
            continue;
        }
        let word = fields.next().unwrap();
        let start = text.len() as u32;
        text.push_str(word);
        spans.push(Span::new(start, text.len() as u32));
        text.push(' ');
    }
    let tokens = Tokens { spans, sentence: None };
    let input = Input::new(&text, &tokens).unwrap();
    let mut matcher = Matcher::new();
    for word in ["the", "and", "to", "of", "with", "in", "for", "is", "a", "this"] {
        matcher.add(word, &[Tok::lower(word)]).unwrap();
    }
    matcher
        .add("article-noun", &[Tok::attr_in(Attr::Lower, ["the", "a", "an"]), Tok::flag(Flag::Alpha, true)])
        .unwrap();
    matcher.add("negated", &[Tok::lower("not"), Tok::flag(Flag::Punct, false).repeat(1, Some(3))]).unwrap();
    let hits = matcher.find(&input).unwrap().len();
    assert!(hits > 0);
    let mut times = Vec::new();
    for _ in 0..iterations {
        let now = Instant::now();
        let found = matcher.find(black_box(&input)).unwrap();
        times.push(now.elapsed().as_nanos());
        assert_eq!(found.len(), hits);
        black_box(found);
    }
    times.sort_unstable();
    println!("tokens={} hits={} median_ns={}", input.len(), hits, times[times.len() / 2]);
}
