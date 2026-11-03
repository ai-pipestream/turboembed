//! Noun and verb chunks read off UD parses written here by hand, and the
//! annotator over a document.

use turbo_analysis::{Annotator, Document, Ix, Key, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Dependencies, PhraseChunks, PosTags, Tag, TagSet, Tokens};
use turbo_phrase_chunker::{ChunkKind, DependencyChunker, derive};

/// `(word, tag, head counted from 1 with 0 the root, relation)` rows.
fn chunks(rows: &[(&str, &str, usize, &str)]) -> Vec<(String, &'static str)> {
    let words: Vec<&str> = rows.iter().map(|r| r.0).collect();
    let tags: Vec<&str> = rows.iter().map(|r| r.1).collect();
    let heads: Vec<Option<usize>> = rows.iter().map(|r| r.2.checked_sub(1)).collect();
    let rels: Vec<&str> = rows.iter().map(|r| r.3).collect();
    derive(&tags, &heads, &rels, &words)
        .into_iter()
        .map(|c| (words[c.start..c.end].join(" "), c.kind.label()))
        .collect()
}

#[test]
fn base_noun_phrases_and_verb_groups() {
    // The very large dog has not eaten John 's two red apples .
    let rows = [
        ("The", "DET", 3, "det"),
        ("very", "ADV", 3, "advmod"),
        ("large", "ADJ", 4, "amod"),
        ("dog", "NOUN", 7, "nsubj"),
        ("has", "AUX", 7, "aux"),
        ("not", "PART", 7, "advmod"),
        ("eaten", "VERB", 0, "root"),
        ("John", "PROPN", 12, "nmod:poss"),
        ("'s", "PART", 8, "case"),
        ("two", "NUM", 12, "nummod"),
        ("red", "ADJ", 12, "amod"),
        ("apples", "NOUN", 7, "obj"),
        (".", "PUNCT", 7, "punct"),
    ];
    assert_eq!(
        chunks(&rows),
        [("The very large dog".into(), "NP"), ("has not eaten".into(), "VP"), ("John 's two red apples".into(), "NP")]
    );
}

#[test]
fn an_adverb_of_an_adjective_and_a_determiner_are_inside() {
    let rows = [
        ("The", "DT", 4, "det"),
        ("very", "RB", 3, "advmod"),
        ("large", "JJ", 4, "amod"),
        ("dog", "NN", 5, "nsubj"),
        ("barked", "VBD", 0, "root"),
    ];
    assert_eq!(chunks(&rows), [("The very large dog".into(), "NP"), ("barked".into(), "VP")]);
}

#[test]
fn prepositions_conjunctions_and_clauses_stay_outside() {
    // We met in the old house and the garden .
    let rows = [
        ("We", "PRON", 2, "nsubj"),
        ("met", "VERB", 0, "root"),
        ("in", "ADP", 5, "case"),
        ("the", "DET", 5, "det"),
        ("old", "ADJ", 6, "amod"),
        ("house", "NOUN", 2, "obl"),
        ("and", "CCONJ", 9, "cc"),
        ("the", "DET", 9, "det"),
        ("garden", "NOUN", 6, "conj"),
        (".", "PUNCT", 2, "punct"),
    ];
    let rows: Vec<_> = rows.iter().map(|r| if r.0 == "the" && r.2 == 5 { (r.0, r.1, 6, r.3) } else { *r }).collect();
    assert_eq!(
        chunks(&rows),
        [("We".into(), "NP"), ("met".into(), "VP"), ("the old house".into(), "NP"), ("the garden".into(), "NP")]
    );
}

#[test]
fn a_participle_inside_a_noun_chunk_is_not_a_verb_group() {
    // the running water flows
    let rows = [
        ("the", "DT", 3, "det"),
        ("running", "VBG", 3, "amod"),
        ("water", "NN", 4, "nsubj"),
        ("flows", "VBZ", 0, "root"),
    ];
    assert_eq!(chunks(&rows), [("the running water".into(), "NP"), ("flows".into(), "VP")]);
}

#[test]
fn a_dependent_across_a_gap_is_left_out() {
    // the , dog : the det of dog, but a comma between that is not in it.
    let rows = [("the", "DT", 3, "det"), (",", ",", 3, "punct"), ("dog", "NN", 0, "root")];
    assert_eq!(chunks(&rows), [("dog".into(), "NP")]);
    assert_eq!(ChunkKind::Verb.label(), "VP");
    assert!(chunks(&[]).is_empty());
    // A cycle ends rather than loops: each token is inside the other's
    // chunk, so neither heads one; a cycle through a head stops there.
    assert!(chunks(&[("the", "DT", 2, "det"), ("dog", "NN", 1, "compound")]).is_empty());
    let looped = [("the", "DT", 3, "det"), ("big", "JJ", 1, "amod"), ("dog", "NN", 2, "nsubj")];
    assert_eq!(chunks(&looped), [("the big dog".into(), "NP")]);
}

/// A document of `rows` (word, Penn tag, head from 1 with 0 the root,
/// relation), the words joined by single spaces, and their parse. The
/// parse is kept out of the document, which refuses a bad one on `add`, so
/// the annotator's own checks can be reached.
fn document(rows: &[(&str, &str, usize, &str)]) -> (Document, Dependencies) {
    let text = rows.iter().map(|r| r.0).collect::<Vec<_>>().join(" ");
    let mut doc = Document::new(text.as_str());
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0u32;
    for r in rows {
        spans.push(Span::new(at, at + r.0.len() as u32));
        at += r.0.len() as u32 + 1;
    }
    let (tag_col, rel_col): (Vec<&str>, Vec<&str>) = rows.iter().map(|r| (r.1, r.3)).unzip();
    fn distinct<'a>(col: &[&'a str]) -> Vec<&'a str> {
        let mut v = Vec::new();
        for &x in col {
            if !v.contains(&x) {
                v.push(x);
            }
        }
        v
    }
    let (tags, rels) = (distinct(&tag_col), distinct(&rel_col));
    let tagset = TagSet::new("penn", &tags);
    let relations = TagSet::new("UD", &rels);
    let pos = PosTags { tags: tag_col.iter().map(|t| tagset.tag(t).unwrap()).collect(), tagset, prob: None };
    let deps = Dependencies {
        head: rows.iter().map(|r| r.2.checked_sub(1).map(|h| Ix::new(h as u32))).collect(),
        relation: rel_col.iter().map(|r| relations.tag(r).unwrap()).collect(),
        relations,
    };
    doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    doc.add(&Key::<PosTags>::DEFAULT, pos, Provenance::by("test")).unwrap();
    (doc, deps)
}

fn annotate(doc: &Document, deps: &Dependencies) -> Result<PhraseChunks, turbo_analysis::Error> {
    let inputs = (doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap(), doc.get::<PosTags>().unwrap(), deps);
    DependencyChunker::default().annotate(inputs).map(|(chunks,)| chunks)
}

const BARKED: [(&str, &str, usize, &str); 5] = [
    ("The", "DT", 3, "det"),
    ("big", "JJ", 3, "amod"),
    ("dog", "NN", 5, "nsubj"),
    ("has", "VBZ", 5, "aux"),
    ("barked", "VBN", 0, "root"),
];

#[test]
fn the_annotator_writes_chunks_over_a_document() {
    let (mut doc, deps) = document(&BARKED);
    let chunks = annotate(&doc, &deps).unwrap();
    // The same parse read back from the document gives the same chunks.
    doc.add(&Key::<Dependencies>::DEFAULT, deps, Provenance::by("test")).unwrap();
    let again = annotate(&doc, doc.get::<Dependencies>().unwrap()).unwrap();
    assert_eq!(again.tokens, chunks.tokens);
    let text = doc.get::<View>().unwrap().text();
    let (starts, ends) = chunks.spans.columns();
    let got: Vec<(&str, &str, (usize, usize))> = (0..chunks.spans.len())
        .map(|i| {
            let r = chunks.tokens[i];
            (
                &text[starts[i] as usize..ends[i] as usize],
                chunks.kinds.label(chunks.kind[i]),
                (r.start.get(), r.end.get()),
            )
        })
        .collect();
    assert_eq!(got, [("The big dog", "NP", (0, 3)), ("has barked", "VP", (3, 5))]);
}

#[test]
fn the_annotator_refuses_a_head_past_the_tokens_and_a_short_parse() {
    let mut rows = BARKED;
    rows[1].2 = 9;
    let (doc, deps) = document(&rows);
    let err = annotate(&doc, &deps).err().expect("refused").to_string();
    assert!(err.contains("dependency head 8 at 1 is out of bounds (5)"), "{err}");

    // A relation outside the tag set, a token its own head, and a cycle.
    let (doc, mut deps) = document(&BARKED);
    deps.relation[0] = Tag(u16::MAX);
    let err = annotate(&doc, &deps).err().expect("refused").to_string();
    assert!(err.contains("a relation outside"), "{err}");
    let (doc, mut deps) = document(&BARKED);
    deps.head[0] = Some(Ix::new(0));
    let err = annotate(&doc, &deps).err().expect("refused").to_string();
    assert!(err.contains("is its own head"), "{err}");
    let (doc, mut deps) = document(&BARKED);
    deps.head[2] = Some(Ix::new(1));
    deps.head[1] = Some(Ix::new(2));
    assert!(annotate(&doc, &deps).is_err(), "big and dog head each other");

    let (doc, mut deps) = document(&BARKED);
    deps.relation.pop();
    let err = annotate(&doc, &deps).err().expect("refused").to_string();
    assert!(err.contains("5 heads and 4 relations for 5 tokens"), "{err}");
}
