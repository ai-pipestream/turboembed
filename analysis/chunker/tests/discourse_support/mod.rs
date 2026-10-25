//! Hand-parsed documents for the discourse tests: each sentence's tokens
//! split on spaces, with an optional CoNLL-style `head:relation` per
//! token (1-based heads within the sentence, `0` for the root).

#![allow(dead_code)]

use turbo_analysis::{Document, Ix, Key, Provenance, Span, Spans, TagSet, ViewId};
use turbo_analysis_layers::{Dependencies, DiscourseEdgeLabel, DiscourseRel, Nuclearity, Sentences, Tokens};
use turbo_chunker::DiscourseAnnotator;

/// A document of `sentences` joined by spaces; with dependencies when
/// `parsed`.
pub fn doc(sentences: &[(&str, &str)], parsed: bool) -> Document {
    let text = sentences.iter().map(|s| s.0).collect::<Vec<_>>().join(" ");
    let (mut spans, mut sent_spans) = (Spans::new(ViewId(0)), Spans::new(ViewId(0)));
    let (mut sentence, mut head, mut rel) = (Vec::new(), Vec::new(), Vec::new());
    let mut at = 0usize;
    for (s, (words, conll)) in sentences.iter().enumerate() {
        let base = spans.len() as u32;
        let start = at;
        for w in words.split(' ') {
            spans.push(Span::new(at as u32, (at + w.len()) as u32));
            sentence.push(Ix::new(s as u32));
            at += w.len() + 1;
        }
        sent_spans.push(Span::new(start as u32, (at - 1) as u32));
        for r in conll.split(' ') {
            let (h, l) = r.split_once(':').unwrap();
            let h: u32 = h.parse().unwrap();
            head.push((h > 0).then(|| Ix::new(base + h - 1)));
            rel.push(l.to_string());
        }
        assert_eq!(head.len(), spans.len(), "{words}");
    }
    let mut d = Document::new(text.as_str());
    d.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test:sentences")).unwrap();
    d.add(&Key::<Tokens>::DEFAULT, Tokens { spans, sentence: Some(sentence) }, Provenance::by("test:tokens")).unwrap();
    if parsed {
        let mut labels: Vec<&str> = rel.iter().map(String::as_str).collect();
        labels.sort_unstable();
        labels.dedup();
        let relations = TagSet::new("ud", &labels);
        let relation = rel.iter().map(|r| relations.tag(r).unwrap()).collect();
        d.add(&Key::<Dependencies>::DEFAULT, Dependencies { head, relation, relations }, Provenance::by("test:deps"))
            .unwrap();
    }
    d
}

/// The units as text and the edges as `(from, to, relation, nuclearity)`.
pub type Edge = (usize, usize, DiscourseRel, Nuclearity);

pub fn run(d: &Document) -> (Vec<String>, Vec<Edge>) {
    let (edus, edges) = DiscourseAnnotator::new().discourse(d).unwrap();
    let text = d.text(ViewId(0));
    let units = edus.spans.iter().map(|(_, s)| text[s.start as usize..s.end as usize].to_string()).collect();
    let edges = edges
        .iter()
        .map(|(_, e)| {
            let l = DiscourseEdgeLabel::from_tag(e.kind).unwrap();
            (e.from.get(), e.to.get(), l.relation, l.nuclearity)
        })
        .collect();
    (units, edges)
}
