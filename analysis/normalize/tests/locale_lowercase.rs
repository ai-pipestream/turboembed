use turbo_analysis::{Document, Key, Provenance, Span, View, ViewId};
use turbo_normalize::{CaseFold, Normalizer, normalize};

fn decode(hex: &str) -> String {
    let (pairs, remainder) = hex.as_bytes().as_chunks::<2>();
    assert!(remainder.is_empty());
    let bytes: Vec<u8> =
        pairs.iter().map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap()).collect();
    String::from_utf8(bytes).unwrap()
}

#[test]
fn jdk21_turkish_and_azerbaijani_oracles_match() {
    let mut count = 0;
    for line in include_str!("data/locale-lowercase.tsv").lines() {
        let cols: Vec<_> = line.split('\t').map(decode).collect();
        assert_eq!(cols.len(), 3);
        for (i, normalizer) in [CaseFold::turkish_lowercase(), CaseFold::azerbaijani_lowercase()].iter().enumerate() {
            let (text, alignment) = normalize(normalizer, &cols[0]);
            assert_eq!(text, cols[i + 1], "row {count}, input {:?}", cols[0]);
            let mut doc = Document::new(cols[0].as_str());
            let child = View::derived(doc.get::<View>().unwrap(), text, alignment);
            doc.add(&Key::named("lower"), child, Provenance::by("locale-lowercase")).unwrap();
            let view = doc.get_key(&Key::<View>::named("lower")).unwrap();
            // Every retained output character maps to valid source UTF-8.
            for (at, c) in view.text().char_indices() {
                let source = doc.to_source(view.id(), Span::new(at as u32, (at + c.len_utf8()) as u32));
                assert!(doc.source().get(source.range()).is_some());
            }
        }
        count += 1;
    }
    assert_eq!(count, 425);
}

#[test]
fn source_and_derived_sentence_spans_round_trip() {
    let mut doc = Document::new("I\u{307}STANBUL. IĞDIR.");
    let (text, a) = normalize(&CaseFold::turkish_lowercase(), doc.source());
    assert_eq!(text, "istanbul. ığdır.");
    doc.add(&Key::named("lower"), View::derived(doc.get::<View>().unwrap(), text, a), Provenance::by("lower")).unwrap();
    let view = doc.get_key(&Key::<View>::named("lower")).unwrap();
    let mut source = turbo_analysis::Spans::<turbo_analysis_layers::Sentences>::new(ViewId::SOURCE);
    source.push(Span::new(0, 11));
    source.push(Span::new(12, doc.source().len() as u32));
    let lower = doc.project_spans(&source, view.id()).unwrap();
    let back = doc.project_spans(&lower, ViewId::SOURCE).unwrap();
    assert_eq!(back.columns(), source.columns());
}

#[test]
fn locale_modes_have_distinct_configurations() {
    let hashes: std::collections::BTreeSet<_> = [
        CaseFold::lowercase(),
        CaseFold::turkish_lowercase(),
        CaseFold::azerbaijani_lowercase(),
        CaseFold::turkic_simple(),
        CaseFold::turkic_full(),
    ]
    .iter()
    .map(Normalizer::config_hash)
    .collect();
    assert_eq!(hashes.len(), 5);
}
