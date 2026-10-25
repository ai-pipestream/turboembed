use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Ix, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{
    ExpansionRow, LexicalExpansions, SAME_SYNSET, SenseRow, Tokens, WORDNET_RELATIONS, WordNetSenses,
};

const TEXT: &str = "Big dogs bark.";

fn document() -> Document {
    let mut doc = Document::new(TEXT);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 3), (4, 8), (9, 13), (13, 14)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let members = ["dog".to_owned(), "domestic dog".to_owned()];
    let mut senses = WordNetSenses::new(ViewId::SOURCE, "test:1".into());
    for (rank, (synset, pos)) in [("s-dog-n", b'n'), ("s-dog-v", b'v')].into_iter().enumerate() {
        let row = SenseRow {
            token: Ix::new(1),
            synset_id: synset,
            sense_id: if rank == 0 { "dog%1:05:00::" } else { "" },
            pos,
            lemma: "dog",
            synset_lemmas: if rank == 0 { &members } else { &[] },
            gloss: "a domestic canine; café",
            sense_rank: rank as u32,
        };
        senses.push(Span::new(4, 8), row);
    }
    let mut expansions = LexicalExpansions::new(ViewId::SOURCE, "test:1".into());
    let rows = [("domestic dog", SAME_SYNSET, 0, 1.0, "s-dog-n"), ("canine", 2, 1, 0.5, "s-canine")];
    for (term, relation, depth, weight, synset_id) in rows {
        let row = ExpansionRow { token: Ix::new(1), term, relation, depth, sense_rank: 0, weight, synset_id };
        expansions.push(Span::new(4, 8), row);
    }
    expansions.push(
        Span::new(9, 13),
        ExpansionRow {
            token: Ix::new(2),
            term: "noise",
            relation: WORDNET_RELATIONS,
            depth: 3,
            sense_rank: 1,
            weight: 0.125,
            synset_id: "s-noise",
        },
    );
    let p = Provenance { reads: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test") };
    doc.add(&Key::DEFAULT, senses, p.clone()).unwrap();
    doc.add(&Key::DEFAULT, expansions, p).unwrap();
    doc
}

#[test]
fn senses_and_expansions_round_trip_through_the_graph() {
    let doc = document();
    let bytes = export(&doc, &codecs()).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();

    let (want, got) = (doc.get::<WordNetSenses>().unwrap(), back.get::<WordNetSenses>().unwrap());
    assert_eq!(got.spans.iter().collect::<Vec<_>>(), want.spans.iter().collect::<Vec<_>>());
    assert_eq!(got.token, want.token);
    assert_eq!(got.synset_id.iter().collect::<Vec<_>>(), ["s-dog-n", "s-dog-v"]);
    assert_eq!(got.sense_id.iter().collect::<Vec<_>>(), ["dog%1:05:00::", ""]);
    assert_eq!(got.pos, b"nv");
    assert_eq!(got.members(0).collect::<Vec<_>>(), ["dog", "domestic dog"]);
    assert_eq!(got.synset_lemma_row, [0, 2, 2]);
    assert_eq!(got.gloss.get(1), "a domestic canine; café");
    assert_eq!(got.sense_rank, [0, 1]);
    assert_eq!(&*got.lexicon_id, "test:1");

    let (want, got) = (doc.get::<LexicalExpansions>().unwrap(), back.get::<LexicalExpansions>().unwrap());
    assert_eq!(got.spans.iter().collect::<Vec<_>>(), want.spans.iter().collect::<Vec<_>>());
    assert_eq!(got.token, want.token);
    assert_eq!(got.term.iter().collect::<Vec<_>>(), ["domestic dog", "canine", "noise"]);
    assert_eq!(got.relation, [SAME_SYNSET, 2, WORDNET_RELATIONS]);
    assert_eq!(got.depth, [0, 1, 3]);
    assert_eq!(got.sense_rank, [0, 0, 1]);
    assert_eq!(got.weight.iter().map(|w| w.to_bits()).collect::<Vec<_>>(), [1.0f64, 0.5, 0.125].map(f64::to_bits));
    assert_eq!(got.synset_id.iter().collect::<Vec<_>>(), ["s-dog-n", "s-canine", "s-noise"]);
    assert_eq!(&*got.lexicon_id, "test:1");

    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), bytes);
}

#[test]
fn the_document_refuses_a_sense_past_its_tokens() {
    let mut doc = document();
    let mut senses = WordNetSenses::new(ViewId::SOURCE, "test:1".into());
    let row = SenseRow {
        token: Ix::new(4),
        synset_id: "s",
        sense_id: "",
        pos: b'n',
        lemma: "x",
        synset_lemmas: &[],
        gloss: "",
        sense_rank: 0,
    };
    senses.push(Span::new(13, 14), row);
    let p = Provenance { reads: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test") };
    assert!(doc.add(&Key::named("past"), senses, p).is_err());
}
