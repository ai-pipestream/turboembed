//! Linear scaling probe for `GrammarChecker` with dependency trees on long multi-clause sentences.

use std::time::Instant;
use turbo_analysis::{Document, Ix, Span, Spans, TagSet, View, ViewId};
use turbo_analysis_layers::{Dependencies, GrammarChecker, PosTags, Sentences, Tokens};

#[test]
fn dependency_lookup_scaling() {
    let mut medians = Vec::new();
    for clauses in [250, 500, 1000, 2000] {
        let text = std::iter::repeat_n("They work and ", clauses).collect::<String>();
        let document = Document::new(text.clone());
        let view = document.get::<View>().unwrap();
        let mut sentence_spans = Spans::new(ViewId::SOURCE);
        sentence_spans.push(Span::new(0, text.len() as u32));
        let sentences = Sentences { spans: sentence_spans };
        let mut spans = Spans::new(ViewId::SOURCE);
        let tagset = TagSet::new("penn", &["PRP", "VBP", "CC"]);
        let relations = TagSet::new("ud", &["nsubj", "root", "cc", "conj"]);
        let mut tags = Vec::new();
        let mut head = Vec::new();
        let mut relation = Vec::new();
        for k in 0..clauses {
            for (offset, len, tag) in [(0, 4, "PRP"), (5, 4, "VBP"), (10, 3, "CC")] {
                spans.push(Span::new((k * 14 + offset) as u32, (k * 14 + offset + len) as u32));
                tags.push(tagset.tag(tag).unwrap());
            }
            head.extend([
                Some(Ix::new((3 * k + 1) as u32)),
                if k == 0 { None } else { Some(Ix::new(1)) },
                Some(Ix::new((3 * k + 1) as u32)),
            ]);
            relation.extend([
                relations.tag("nsubj").unwrap(),
                relations.tag(if k == 0 { "root" } else { "conj" }).unwrap(),
                relations.tag("cc").unwrap(),
            ]);
        }
        let tokens = Tokens { spans, sentence: None };
        let pos = PosTags { tagset, tags, prob: None };
        let deps = Dependencies { head, relation, relations };
        let mut timings = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let result = GrammarChecker::new().check_view(view, &sentences, &tokens, &pos, None, Some(&deps)).unwrap();
            std::hint::black_box(result);
            timings.push(start.elapsed().as_nanos());
        }
        timings.sort_unstable();
        println!("tokens={} median_ns={}", clauses * 3, timings[2]);
        medians.push((clauses * 3, timings[2]));
    }
    let (_, t_750) = medians[0];
    let (_, t_6000) = medians[3];
    assert!(
        t_6000 < t_750.saturating_mul(25),
        "8x token increase should scale near-linearly, got 750 tokens = {t_750}ns vs 6000 tokens = {t_6000}ns"
    );
}
