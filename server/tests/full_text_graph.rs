//! Full text to document graph through the in-process Rust API (#1948):
//! the progressive step registry compiled into one analyzer over UTF-8,
//! multi-sentence text, every stage that needs no model bundle at once.
//! The graph's spans slice the source, map to the UTF-16 and code point
//! offsets the text itself counts (also past characters outside the Basic
//! Multilingual Plane) and back, project into the normalized and the
//! case-folded view and back, and cross export and import byte for byte; a run is deterministic, empty input is a graph,
//! a long document keeps every invariant, and a stage the request cannot
//! run is refused by name rather than dropped.

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Key, Layer, Spans, View, ViewId};
use turbo_analysis_layers::{
    Abbreviations, Boundaries, Entities, Money, Numbers, Pii, Sentences, Stems, Structure, Temporals, Terms, Tokens,
};
use turbo_keywords::Keywords;
use turbo_kserve::progressive::compile_analyzer;
use turbo_kserve::results::document::codecs;
use turbo_stopword_layers::Stopwords;

/// Every model-free stage: normalization, structure, sentences, tokens,
/// abbreviations, terms and stems, stopwords, keywords, the place-name
/// gazetteer, PII, numbers, money, dates and both boundary layers.
const STEPS: &[&str] = &[
    "turbo:normalize",
    "turbo-structure:blocks",
    "turbo-sentdetect:rules",
    "turbo-tokenize:uax29-untyped",
    "turbo-abbrev:schwartz-hearst",
    "turbo-stem:terms",
    "turbo-stem:stemmer",
    "turbo-stopword:bundled",
    "turbo-keywords:yake",
    "turbo-geo:names",
    "turbo-pii:pii",
    "turbo-numeric:numbers",
    "turbo-numeric:money",
    "turbo-numeric:temporal",
    "turbo:structure-boundaries",
    "turbo:lexical-boundaries",
];

/// A heading, three sentences, two paragraphs, Latin letters with
/// diacritics, a ligature NFKC unfolds (`ﬁ`, three bytes, to `fi`), a
/// currency sign, an address and a phone number. Outside the Basic
/// Multilingual Plane: an emoji before the first place and an ideograph
/// (U+20BB7) in the last sentence, each two UTF-16 units and one code
/// point. Spellings whose length changes: `e` with a combining acute
/// (NFKC composes it), sharp s (folds to `ss`) and dotted capital I
/// (folds to `i` and a combining dot).
const BODY: &str = "Zoë Müller \u{1F680} moved from São Paulo to Paris on 3 March 2024. She paid €1,200 for the ﬁrst \
                    flat, mail zoe.muller@example.com or call +1 555 010 0777.\n\nThe World Health Organization \
                    (WHO) met in Berlin with \u{20BB7}野 at the cafe\u{301} on Straße 5 by İpek.";

fn text() -> String {
    format!("# Relatório\n\n{BODY}")
}

fn analyzed(text: &str) -> Document {
    let mut doc = Document::new(text);
    let analyzer = compile_analyzer(&doc, STEPS).unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().for_each(Result::unwrap);
    doc
}

fn graph_bytes(doc: &Document) -> Vec<u8> {
    export(doc, codecs()).unwrap().to_bytes().unwrap()
}

fn slices<L>(doc: &Document, spans: &Spans<L>) -> Vec<String> {
    let source = doc.source();
    spans
        .iter()
        .map(|(_, s)| source.get(s.start as usize..s.end as usize).expect("a span on char boundaries").to_owned())
        .collect()
}

fn layer<L: Layer>(doc: &Document) -> &L {
    doc.get::<L>().unwrap_or_else(|| panic!("{} is in the graph", L::NAME))
}

/// Every span of `spans`, in the source view, to UTF-16 and code points
/// and back to the same bytes. Each converted offset is also the count of
/// units the source text itself has before it, so a conversion that is
/// wrong and reversible does not pass.
fn offsets_round_trip<L>(doc: &Document, spans: &Spans<L>, what: &str) {
    assert_eq!(spans.view(), ViewId::SOURCE, "{what} is projected to the source");
    // Units and code points before each byte offset, counted once.
    let source = doc.source();
    let (mut units_at, mut points_at) = (vec![0u32; source.len() + 1], vec![0u32; source.len() + 1]);
    let (mut u, mut p) = (0, 0);
    for (at, ch) in source.char_indices() {
        (units_at[at], points_at[at]) = (u, p);
        (u, p) = (u + ch.len_utf16() as u32, p + 1);
    }
    (units_at[source.len()], points_at[source.len()]) = (u, p);
    let units = |to: u32| units_at[to as usize];
    let points = |to: u32| points_at[to as usize];
    for (_, s) in spans.iter() {
        let u = doc.view_to_source_utf16(ViewId::SOURCE, s).unwrap();
        assert_eq!((u.start.0, u.end.0), (units(s.start), units(s.end)), "{what} {s:?} in UTF-16 units");
        assert_eq!(doc.source_utf16_to_view(ViewId::SOURCE, u).unwrap(), s, "{what} {s:?} through UTF-16");
        let c = doc.view_to_source_code_point(ViewId::SOURCE, s).unwrap();
        assert_eq!((c.start.0, c.end.0), (points(s.start), points(s.end)), "{what} {s:?} in code points");
        assert_eq!(doc.source_code_point_to_view(ViewId::SOURCE, c).unwrap(), s, "{what} {s:?} through code points");
    }
}

#[test]
fn every_model_free_stage_writes_a_graph_that_slices_the_source() {
    let text = text();
    let doc = analyzed(&text);
    assert_eq!(doc.source(), text);
    let tokens = layer::<Tokens>(&doc);
    let words = slices(&doc, &tokens.spans);
    for w in ["Relatório", "Zoë", "Müller", "São", "ﬁrst", "1,200", "WHO"] {
        assert!(words.iter().any(|t| t == w), "token {w:?} in {words:?}");
    }
    let sentences = slices(&doc, &layer::<Sentences>(&doc).spans);
    assert_eq!(sentences.len(), 3, "{sentences:?}");
    assert!(sentences[1].starts_with("She paid") && sentences[2].starts_with("The World"), "{sentences:?}");
    assert_eq!(slices(&doc, &layer::<Entities>(&doc).spans), ["São Paulo", "Paris", "Berlin"], "gazetteer places");
    let pii = slices(&doc, &layer::<Pii>(&doc).spans);
    assert_eq!(pii, ["zoe.muller@example.com", "+1 555 010 0777"]);
    assert!(slices(&doc, &layer::<Money>(&doc).spans).iter().any(|m| m.contains("1,200")));
    assert!(!layer::<Temporals>(&doc).spans.is_empty() && !layer::<Numbers>(&doc).spans.is_empty());
    let abbreviations = slices(&doc, &layer::<Abbreviations>(&doc).spans);
    assert!(abbreviations.iter().any(|a| a == "WHO"), "{abbreviations:?}");
    // The token-aligned layers have one row per token.
    for (what, n) in [
        ("terms", layer::<Terms>(&doc).len()),
        ("stems", layer::<Stems>(&doc).len()),
        ("stopwords", layer::<Stopwords>(&doc).len()),
    ] {
        assert_eq!(n, tokens.len(), "{what}");
    }
    assert!(!layer::<Keywords>(&doc).is_empty());
    assert!(layer::<Structure>(&doc).len() >= 2, "a heading block and paragraphs");
    assert!(doc.get::<Boundaries>().is_some() && doc.get_key(&Key::<Boundaries>::named("lexical")).is_some());

    for (what, spans) in [
        ("tokens", slices_view(&tokens.spans)),
        ("entities", slices_view(&layer::<Entities>(&doc).spans)),
        ("pii", slices_view(&layer::<Pii>(&doc).spans)),
        ("sentences", slices_view(&layer::<Sentences>(&doc).spans)),
    ] {
        offsets_round_trip(&doc, &spans, what);
    }
}

/// `spans` as untyped spans, for one helper over every layer.
fn slices_view<L>(spans: &Spans<L>) -> Spans<()> {
    let mut out = Spans::new(spans.view());
    for (_, s) in spans.iter() {
        out.push(s);
    }
    out
}

#[test]
fn source_spans_project_into_the_normalized_view_and_back() {
    let doc = analyzed(&text());
    let normalized = doc.require(&Key::<View>::named("normalized")).unwrap();
    assert!(normalized.text().contains("first flat") && !normalized.text().contains('ﬁ'), "NFKC unfolds the ligature");
    let tokens = &layer::<Tokens>(&doc).spans;
    let there = doc.project_spans(tokens, normalized.id()).unwrap();
    let back = doc.project_spans(&there, ViewId::SOURCE).unwrap();
    let at = |s: &Spans<Tokens>, text: &str| -> Vec<String> {
        s.iter().map(|(_, s)| text[s.start as usize..s.end as usize].to_owned()).collect()
    };
    let in_view = at(&there, normalized.text());
    let in_source = at(tokens, doc.source());
    let i = in_source.iter().position(|t| t == "ﬁrst").unwrap();
    assert_eq!((in_source[i].len(), in_view[i].as_str()), (6, "first"), "one token, two spellings");
    assert_eq!(at(&back, doc.source()), in_source, "every token maps back to its source bytes");
}

#[test]
fn a_span_after_an_astral_character_is_one_utf16_unit_further_per_surrogate_pair() {
    let doc = analyzed(&text());
    let source = doc.source();
    let found = |spans: &Spans<()>, word: &str| {
        let hit = spans.iter().map(|(_, s)| s).find(|s| &source[s.start as usize..s.end as usize] == word);
        hit.unwrap_or_else(|| panic!("a span over {word:?}"))
    };
    let tokens = slices_view(&layer::<Tokens>(&doc).spans);
    // (token, surrogate pairs before it, surrogate pairs inside it)
    for (word, before, inside) in
        [("Müller", 0, 0), ("\u{1F680}", 0, 1), ("moved", 1, 0), ("\u{20BB7}", 1, 1), ("İpek", 2, 0)]
    {
        let s = found(&tokens, word);
        let u = doc.view_to_source_utf16(ViewId::SOURCE, s).unwrap();
        let c = doc.view_to_source_code_point(ViewId::SOURCE, s).unwrap();
        assert_eq!(u.start.0 - c.start.0, before, "{word:?} starts {before} surrogate pairs in");
        assert_eq!(u.end.0 - c.end.0, before + inside, "{word:?} ends after {inside} more");
    }
    // The same holds for the layers that are not tokens: a place and a PII
    // value after the emoji, and the sentence that holds the ideograph.
    for (spans, word, before) in [
        (slices_view(&layer::<Entities>(&doc).spans), "Paris", 1),
        (slices_view(&layer::<Pii>(&doc).spans), "zoe.muller@example.com", 1),
    ] {
        let s = found(&spans, word);
        let u = doc.view_to_source_utf16(ViewId::SOURCE, s).unwrap();
        let c = doc.view_to_source_code_point(ViewId::SOURCE, s).unwrap();
        assert_eq!((u.start.0 - c.start.0, u.end.0 - c.end.0), (before, before), "{word:?}");
    }
    let last = layer::<Sentences>(&doc).spans.iter().map(|(_, s)| s).last().unwrap();
    let u = doc.view_to_source_utf16(ViewId::SOURCE, last).unwrap();
    let c = doc.view_to_source_code_point(ViewId::SOURCE, last).unwrap();
    assert_eq!((u.start.0 - c.start.0, u.end.0 - c.end.0), (1, 2), "the ideograph is inside the last sentence");
}

#[test]
fn spellings_that_change_length_project_into_the_folded_view_and_back() {
    let text = text();
    let mut doc = Document::new(text.as_str());
    let steps = ["turbo:normalize", "turbo:normalize:case-fold", "turbo-tokenize:uax29-untyped"];
    let analyzer = compile_analyzer(&doc, &steps).unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().for_each(Result::unwrap);
    let folded = doc.require(&Key::<View>::named("normalized")).unwrap();
    let tokens = &layer::<Tokens>(&doc).spans;
    let there = doc.project_spans(tokens, folded.id()).unwrap();
    let back = doc.project_spans(&there, ViewId::SOURCE).unwrap();
    let pairs: Vec<(&str, &str)> = tokens
        .iter()
        .zip(there.iter())
        .map(|((_, s), (_, v))| {
            (&text[s.start as usize..s.end as usize], &folded.text()[v.start as usize..v.end as usize])
        })
        .collect();
    // (source spelling, folded spelling): composed by NFKC, then folded.
    for want in [
        ("cafe\u{301}", "café"),
        ("Straße", "strasse"),
        ("İpek", "i\u{307}pek"),
        ("ﬁrst", "first"),
        ("\u{20BB7}", "\u{20BB7}"),
        ("Zoë", "zoë"),
    ] {
        assert!(pairs.contains(&want), "{want:?} in {pairs:?}");
    }
    let bytes = |s: &Spans<Tokens>| s.iter().map(|(_, s)| (s.start, s.end)).collect::<Vec<_>>();
    assert_eq!(bytes(&back), bytes(tokens), "every token maps back to its source bytes");
    offsets_round_trip(&doc, &slices_view(tokens), "tokens of the folded chain");
}

#[test]
fn the_graph_crosses_export_and_import_byte_for_byte_and_runs_are_deterministic() {
    let text = text();
    let doc = analyzed(&text);
    let bytes = graph_bytes(&doc);
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), codecs()).unwrap();
    assert_eq!(back.source(), text);
    assert_eq!(graph_bytes(&back), bytes, "re-export is byte-identical");
    assert_eq!(slices(&back, &layer::<Entities>(&back).spans), slices(&doc, &layer::<Entities>(&doc).spans));
    assert_eq!(slices(&back, &layer::<Pii>(&back).spans), slices(&doc, &layer::<Pii>(&doc).spans));
    assert_eq!(graph_bytes(&analyzed(&text)), bytes, "a second run writes the same graph");
}

#[test]
fn empty_input_is_an_empty_graph_and_a_long_document_keeps_every_invariant() {
    let doc = analyzed("");
    assert!(layer::<Tokens>(&doc).is_empty() && layer::<Sentences>(&doc).is_empty());
    let bytes = graph_bytes(&doc);
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), codecs()).unwrap();
    assert_eq!(graph_bytes(&back), bytes);

    // 200 copies of the body: 600 sentences, every place and PII value
    // found in every copy, at its own offsets.
    let copies = 200;
    let long = vec![BODY; copies].join("\n\n");
    let doc = analyzed(&long);
    assert_eq!(layer::<Sentences>(&doc).len(), 3 * copies);
    let entities = slices(&doc, &layer::<Entities>(&doc).spans);
    assert_eq!(entities.len(), 3 * copies);
    assert!(entities.chunks(3).all(|c| c == ["São Paulo", "Paris", "Berlin"]));
    let pii = &layer::<Pii>(&doc).spans;
    assert_eq!(pii.len(), 2 * copies);
    let starts: Vec<u32> = pii.iter().map(|(_, s)| s.start).collect();
    assert!(starts.windows(2).all(|w| w[0] < w[1]), "each copy's values at their own offsets");
    offsets_round_trip(&doc, &slices_view(&layer::<Tokens>(&doc).spans), "tokens");
    assert_eq!(graph_bytes(&analyzed(&long)), graph_bytes(&doc), "deterministic at length");
}

#[test]
fn a_stage_that_cannot_run_is_refused_by_name() {
    let doc = Document::new("a b");
    let refused = |steps: &[&str]| match compile_analyzer(&doc, steps) {
        Ok(_) => panic!("{steps:?} compiled"),
        Err(f) => f.message,
    };
    // Smart chunking reads the embedding model's subwords; without a
    // bundle it is refused, naming the layer it lacks.
    let e = refused(&["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo:chunker"]);
    assert!(e.contains("turbo:chunker") && e.contains("turbo:subwords"), "{e}");
    let e = refused(&["turbo-tokenize:uax29-untyped", "turbo:nonexistent"]);
    assert!(e.contains("turbo:nonexistent"), "{e}");
    // Open-label entities need their labels, and relations their model.
    let e = refused(&["turbo-tokenize:uax29-untyped", "turbo:gliner"]);
    assert!(e.contains("turbo:gliner") && e.contains("open_span"), "{e}");
    let e = refused(&["turbo-tokenize:uax29-untyped", "turbo:zero-shot-relations"]);
    assert!(e.contains("turbo:zero-shot-relations") && e.contains("relation labels"), "{e}");
}
