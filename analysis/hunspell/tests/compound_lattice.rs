use std::sync::Arc;

use turbo_analysis::{Analyzer, Document, ViewId};
use turbo_analysis_layers::TokenGraph;
use turbo_hunspell::{Dictionary, HunspellLatticeAnnotator, LoadMode, Stemmer};
use turbo_tokenize::WordTokenizer;

fn german_compound_dict() -> Arc<Dictionary> {
    let aff = b"SET UTF-8\nCOMPOUNDMIN 3\nCOMPOUNDFLAG C\nSIMPLIFIEDTRIPLE\n";
    let dic = b"5\nDonau/C\ndampf/C\nDampf/C\nschiff/C\nfahrt/C\n";
    Arc::new(Dictionary::from_bytes(aff, dic, LoadMode::Strict).expect("valid dictionary"))
}

#[test]
fn hunspell_compound_lattice_emits_parts_and_whole_with_exact_source_spans() {
    let dict = german_compound_dict();
    let stemmer = Stemmer::new(&dict);
    let segments = stemmer.decompose_word("Donaudampfschiff").expect("decomposed compound");
    assert_eq!(segments.len(), 3);
    assert_eq!(&"Donaudampfschiff"[segments[0].byte_range.clone()], "Donau");
    assert_eq!(&"Donaudampfschiff"[segments[1].byte_range.clone()], "dampf");
    assert_eq!(&"Donaudampfschiff"[segments[2].byte_range.clone()], "schiff");

    // Simplified triple "Schiffahrt" -> "Schiff" + "fahrt"
    let triple_segs = stemmer.decompose_word("Schiffahrt").expect("simplified triple compound");
    assert_eq!(triple_segs.len(), 2);
    assert_eq!(&"Schiffahrt"[triple_segs[0].byte_range.clone()], "Schiff");
    assert_eq!(&"Schiffahrt"[triple_segs[1].byte_range.clone()], "fahrt");

    let text = "der Donaudampfschiff Hafen";
    let analyzer =
        Analyzer::builder().add(WordTokenizer::new()).add(HunspellLatticeAnnotator::new(dict)).build().unwrap();

    let mut doc = Document::new(text);
    analyzer.run(std::slice::from_mut(&mut doc)).remove(0).unwrap();
    let graph = doc.get::<TokenGraph>().unwrap();

    assert_eq!(graph.matches_phrase(&["der", "donaudampfschiff", "hafen"]).len(), 1);
    assert_eq!(graph.matches_phrase(&["der", "donau", "dampf", "schiff", "hafen"]).len(), 1);
    assert!(graph.matches_phrase(&["der", "hafen"]).is_empty());

    for (_, arc) in graph.iter() {
        let src = doc.to_source(ViewId::SOURCE, arc.span);
        match arc.term {
            "donau" => assert_eq!(&text[src.range()], "Donau"),
            "dampf" => assert_eq!(&text[src.range()], "dampf"),
            "schiff" => assert_eq!(&text[src.range()], "schiff"),
            "donaudampfschiff" => assert_eq!(&text[src.range()], "Donaudampfschiff"),
            _ => {}
        }
    }
}
