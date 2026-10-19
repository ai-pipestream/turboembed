//! Glossary terms across the document graph boundary: the terms
//! and the glossary they point into round-trip through bytes, and
//! payloads naming the wrong tokens instance, entries outside the
//! glossary, or blank glossary entries are refused.

use std::sync::Arc;

use turbo_analysis::export::{Codecs, DocumentGraph, ExportError, Writer, export, import};
use turbo_analysis::{Document, IxRange, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_glossary_layers::export::register_codecs;
use turbo_glossary_layers::{EntryIx, Glossary, GlossaryTerms};

fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    register_codecs(&mut codecs);
    codecs
}

fn tokens(pairs: &[(u32, u32)]) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        spans.push(Span::new(a, b));
    }
    Tokens { spans, sentence: None }
}

fn glossary() -> Arc<Glossary> {
    let mut g = Glossary::new();
    g.push("c1", "hot dog").unwrap();
    g.push("c2", "stand").unwrap();
    Arc::new(g)
}

fn document() -> Document {
    let mut doc = Document::new("hot dog stand");
    doc.add(&Key::DEFAULT, tokens(&[(0, 3), (4, 7), (8, 13)]), Provenance::by("test")).unwrap();
    let mut terms = GlossaryTerms::new(ViewId::SOURCE, glossary());
    terms.push(Span::new(0, 7), IxRange::new(0, 2), EntryIx(0));
    terms.push(Span::new(8, 13), IxRange::new(2, 3), EntryIx(1));
    doc.add(&Key::DEFAULT, terms, Provenance::by("test")).unwrap();
    doc
}

/// The span column encoding, written out longhand so corrupted
/// payloads can be built without an export context.
fn write_spans(w: &mut Writer, spans: &[Span]) {
    w.str("").unwrap();
    w.usize(spans.len()).unwrap();
    for s in spans {
        w.usize(1).unwrap();
        w.span(*s);
    }
}

fn body(instance: &str, ranges: &[(u32, u32)], entries: &[(&str, &str)], entry: &[u32]) -> Vec<u8> {
    let mut w = Writer::new();
    write_spans(&mut w, &[Span::new(0, 1)]);
    w.str("opennlp:tokens").unwrap();
    w.str(instance).unwrap();
    w.usize(ranges.len()).unwrap();
    for &(a, b) in ranges {
        w.u32(a);
        w.u32(b);
    }
    w.usize(entries.len()).unwrap();
    for &(id, term) in entries {
        w.str(id).unwrap();
        w.str(term).unwrap();
    }
    w.usize(entry.len()).unwrap();
    for &e in entry {
        w.u32(e);
    }
    w.finish()
}

fn splice(graph: &mut DocumentGraph, payload: Vec<u8>) {
    let at = graph.layers.iter().position(|l| l.layer == "turbo:glossary-terms").unwrap();
    graph.layers[at].body = Some(payload);
    graph.layers[at].len = 1;
}

#[test]
fn glossary_terms_round_trip() {
    let graph = export(&document(), &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    let terms = back.get::<GlossaryTerms>().unwrap();
    assert_eq!(terms.tokens, [IxRange::new(0, 2), IxRange::new(2, 3)]);
    assert_eq!(terms.entry_of(0).term, "hot dog");
    assert_eq!(terms.entry_of(1).id, "c2");
    assert_eq!(terms.glossary.as_ref(), glossary().as_ref());
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn corrupted_glossary_terms_are_refused() {
    let base = export(&document(), &codecs()).unwrap();
    for (what, payload) in [
        ("an entry outside the glossary", body("", &[(0, 1)], &[("c1", "hot dog")], &[1])),
        ("a token range past the tokens", body("", &[(0, 9)], &[("c1", "hot dog")], &[0])),
        ("a backwards token range", body("", &[(1, 0)], &[("c1", "hot dog")], &[0])),
        ("a blank glossary id", body("", &[(0, 1)], &[(" ", "hot dog")], &[0])),
        ("a blank glossary term", body("", &[(0, 1)], &[("c1", "\u{a0}")], &[0])),
        ("a missing tokens instance", body("missing", &[(0, 1)], &[("c1", "hot dog")], &[0])),
        ("truncated", Vec::new()),
    ] {
        let mut graph = base.clone();
        splice(&mut graph, payload);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }
}

#[test]
fn keyed_default_glossary_target_must_not_rebind_through_provenance() {
    let mut doc = Document::new("ab");
    doc.add(&Key::DEFAULT, tokens(&[(0, 1)]), Provenance::by("test")).unwrap();
    doc.add(&Key::named("alt"), tokens(&[(1, 2)]), Provenance::by("test")).unwrap();
    let mut terms = GlossaryTerms::new(ViewId::SOURCE, glossary());
    terms.push(Span::new(1, 2), IxRange::new(0, 1), EntryIx(0));
    let mut provenance = Provenance::by("test");
    provenance.reads = vec![Key::<Tokens>::named("alt").id()];
    doc.add(&Key::DEFAULT, terms, provenance).unwrap();

    // The valid document, indexing the alt tokens, round-trips.
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());

    // The same body with only the encoded target instance emptied
    // would validate against the default tokens and then rebind to
    // alt through provenance, so it must be refused.
    let mut graph = export(&doc, &codecs()).unwrap();
    let payload = body("", &[(0, 1)], &[("c1", "hot dog"), ("c2", "stand")], &[0]);
    splice(&mut graph, payload);
    match import(&graph, &codecs()) {
        Err(ExportError::InvalidLayer { detail, .. }) => assert!(detail.contains("alt"), "{detail}"),
        Ok(_) => panic!("an empty glossary target rebound through provenance imported"),
        Err(other) => panic!("an empty glossary target failed as {other}"),
    }
}
