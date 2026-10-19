//! Near-duplicate grouping and version links over structure blocks and
//! sentences: exact copies, edited versions, nested blocks, fragments,
//! a case folded view, and a batch of documents.

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Structure};
use turbo_group::{
    DuplicateKind, Member, NearDuplicates, Sketcher, Sketches, Thresholds, cross_duplicates, duplicate_kinds,
};
use turbo_normalize::{CaseFold, NormalizeStep};
use turbo_structure::{Syntax, detect};

/// `n` distinct words, `w0 w1 ...`, with word `swap` replaced by `edit`.
fn words(n: usize, from: usize, swap: Option<(usize, &str)>) -> String {
    let mut out: Vec<String> = (from..from + n).map(|i| format!("w{i}")).collect();
    if let Some((at, word)) = swap {
        out[at] = word.to_owned();
    }
    out.join(" ")
}

fn analyze(text: &str) -> Document {
    let mut docs = [Document::new(text.to_owned())];
    docs[0].add(&Key::DEFAULT, detect(text, Syntax::Markdown, ViewId::SOURCE), Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(Sketcher::<Structure>::default())
        .add(NearDuplicates::<Structure>::default())
        .build()
        .unwrap();
    assert_eq!(pipeline.run(&mut docs), [Ok(())]);
    let [doc] = docs;
    doc
}

fn paragraphs(doc: &Document) -> Vec<Ix<Structure>> {
    let st = doc.get::<Structure>().unwrap();
    st.spans.iter().filter(|(i, _)| st.kind[i.get()] == Block::Paragraph).map(|(i, _)| i).collect()
}

#[test]
fn exact_copies_near_duplicates_and_versions_link_from_earlier_to_later() {
    let long = words(200, 0, None);
    let text = [
        long.clone(),
        // The same words with other punctuation and spacing: an exact copy.
        long.replace(' ', ",  "),
        // One word of 200 changed: a near duplicate.
        words(200, 0, Some((100, "changed"))),
        words(60, 1000, None),
        // Two words of 60 changed: a version.
        words(60, 1000, Some((20, "a"))).replace("w1040", "b"),
        words(30, 5000, None),
    ]
    .join("\n\n");
    let doc = analyze(&text);
    let p = paragraphs(&doc);
    assert_eq!(p.len(), 6);
    let edges = doc.get::<Edges<Structure, Structure>>().unwrap();
    let kinds = duplicate_kinds();
    assert_eq!(edges.kinds.name, kinds.name);
    let rows: Vec<(Ix<Structure>, Ix<Structure>, &str)> =
        edges.iter().map(|(_, e)| (e.from, e.to, edges.kinds.label(e.kind))).collect();
    assert_eq!(rows, [(p[0], p[1], "exact"), (p[0], p[2], "near-duplicate"), (p[3], p[4], "version-edit")]);
    let scores: Vec<f64> = edges.iter().map(|(_, e)| e.score).collect();
    assert_eq!(scores[0], 1.0);
    assert!(scores[1] >= 0.9 && scores[1] < 1.0, "{scores:?}");
    assert!((0.5..0.9).contains(&scores[2]), "{scores:?}");

    // Both directions are binary searches: the first copy's later copies,
    // and the earlier copy of the version.
    let out: Vec<_> = edges.outgoing(p[0]).map(|(_, e)| e.to).collect();
    assert_eq!(out, [p[1], p[2]]);
    let into: Vec<_> = edges.incoming(p[4]).map(|(_, e)| (e.from, edges.kinds.label(e.kind))).collect();
    assert_eq!(into, [(p[3], "version-edit")]);
    assert_eq!(edges.incoming(p[5]).len() + edges.outgoing(p[5]).len(), 0);

    // The exact copy and the near duplicate are one group; the version is
    // a link only.
    let groups = doc.get::<Groups<Structure>>().unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups.get(Ix::new(0)), [p[0], p[1], p[2]]);
}

#[test]
fn a_block_and_the_block_nested_in_it_are_not_duplicates() {
    let text = "- one list item of several words\n\n- another item\n\none list item of several words";
    let doc = analyze(text);
    let st = doc.get::<Structure>().unwrap();
    let edges = doc.get::<Edges<Structure, Structure>>().unwrap();
    let pairs: Vec<(Block, Block)> = edges.iter().map(|(_, e)| (st.kind[e.from.get()], st.kind[e.to.get()])).collect();
    // The first item and its paragraph hold one text; only the copy at
    // the end, outside the list, is a duplicate, linked from the item.
    assert_eq!(pairs, [(Block::ListItem, Block::Paragraph)]);
    assert!(edges.iter().all(|(_, e)| st.spans.get(e.from).end <= st.spans.get(e.to).start));
}

#[test]
fn fragments_are_read_apart_and_no_word_joins_two() {
    let text = "alpha beta | gamma delta. alpha beta gamma delta. alp ha.";
    let mut doc = Document::new(text);
    let at = |s: &str| text.find(s).unwrap() as u32;
    let mut st =
        Structure { spans: Spans::new(ViewId::SOURCE), kind: Vec::new(), parent: Vec::new(), depth: Vec::new() };
    st.spans.push_fragments(&[Span::new(0, 10), Span::new(13, 24)]);
    st.spans.push(Span::new(at("alpha beta gamma"), at("alpha beta gamma") + 22));
    st.spans.push_fragments(&[Span::new(at("alp ha"), at("alp ha") + 3), Span::new(at("ha."), at("ha.") + 2)]);
    for _ in 0..3 {
        st.kind.push(Block::Paragraph);
        st.parent.push(None);
        st.depth.push(0);
    }
    doc.add(&Key::DEFAULT, st, Provenance::by("t")).unwrap();
    let mut docs = [doc];
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(Sketcher::<Structure>::default())
        .build()
        .unwrap();
    assert_eq!(pipeline.run(&mut docs), [Ok(())]);
    let s = docs[0].get::<Sketches<Structure>>().unwrap();
    assert_eq!(s.words, [4, 4, 2]);
    assert_eq!(s.exact[0], s.exact[1], "the fragments read as the contiguous sentence does");
    assert_eq!(s.jaccard(Ix::new(0), Ix::new(1)), 1.0);
    assert_eq!(s.simhash_similarity(Ix::new(0), Ix::new(1)), 1.0);
}

#[test]
fn sketches_read_from_a_folded_view_map_the_blocks_through_its_alignment() {
    let text = "Hello World again and again\n\nhello world AGAIN and Again";
    let mut docs = [Document::new(text)];
    docs[0].add(&Key::DEFAULT, detect(text, Syntax::Markdown, ViewId::SOURCE), Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .step(Bound::new(NormalizeStep::new(CaseFold::full())).write(0, "folded"))
        .step(Bound::new(Sketcher::<Structure>::default()).read(0, "folded"))
        .add(NearDuplicates::<Structure>::default())
        .build()
        .unwrap();
    assert_eq!(pipeline.run(&mut docs), [Ok(())]);
    let edges = docs[0].get::<Edges<Structure, Structure>>().unwrap();
    assert_eq!(edges.iter().map(|(_, e)| edges.kinds.label(e.kind)).collect::<Vec<_>>(), ["exact"]);
}

#[test]
fn a_batch_links_across_documents_from_earlier_to_later() {
    let a = analyze(&[words(50, 0, None), words(40, 300, None)].join("\n\n"));
    let b = analyze(&[words(40, 900, None), words(50, 0, None)].join("\n\n"));
    let c = analyze(&words(50, 0, Some((10, "x"))).replace("w30", "y").replace("w45", "z"));
    let docs = [&a, &b, &c];
    let batch: Vec<(&Structure, &Sketches<Structure>)> =
        docs.iter().map(|d| (d.get::<Structure>().unwrap(), d.get::<Sketches<Structure>>().unwrap())).collect();
    let found = cross_duplicates(&batch, &Thresholds::default()).unwrap();
    let m = |doc: u32, d: &Document, k: usize| Member { doc, item: paragraphs(d)[k] };
    let links: Vec<_> = found.links.iter().map(|l| (l.from, l.to, l.kind)).collect();
    assert_eq!(
        links,
        [(m(0, &a, 0), m(1, &b, 1), DuplicateKind::Exact), (m(0, &a, 0), m(2, &c, 0), DuplicateKind::VersionEdit),]
    );
    assert_eq!(found.groups, [vec![m(0, &a, 0), m(1, &b, 1)]]);
    assert_eq!(found.outgoing(m(0, &a, 0)).len(), 2);
    assert!(found.outgoing(m(1, &b, 1)).is_empty());

    // Sketches of another shingle width do not compare.
    let mut other = batch[1].1.clone();
    other.shingle = 2;
    let mixed = [batch[0], (batch[1].0, &other)];
    assert!(cross_duplicates(&mixed, &Thresholds::default()).unwrap_err().contains("shingles of 2 words"));
    assert!(cross_duplicates::<Structure>(&[], &Thresholds::default()).unwrap().links.is_empty());
}

#[test]
fn the_same_text_sketches_the_same_every_time() {
    // Pinned values: a change to any hash constant shows here, since
    // sketches are compared across processes and machines.
    let doc = analyze("The same text sketches the same.");
    let s = doc.get::<Sketches<Structure>>().unwrap();
    let again = analyze("The same text sketches the same.");
    let t = again.get::<Sketches<Structure>>().unwrap();
    assert_eq!(
        (s.exact.clone(), s.simhash.clone(), s.minhash.clone()),
        (t.exact.clone(), t.simhash.clone(), t.minhash.clone())
    );
    assert_eq!(s.words, [6]);
    assert_eq!(format!("{:016x}", s.simhash[0]), PINNED_SIMHASH);
}

const PINNED_SIMHASH: &str = "a01b432609080004";

#[test]
fn bad_thresholds_and_a_shingle_of_no_words_are_refused() {
    let text = "one two three";
    let mut docs = [Document::new(text)];
    docs[0].add(&Key::DEFAULT, detect(text, Syntax::Markdown, ViewId::SOURCE), Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(Sketcher::<Structure>::new(0))
        .build()
        .unwrap();
    assert!(pipeline.run(&mut docs)[0].as_ref().unwrap_err().to_string().contains("no words"));
    let t = Thresholds { near: 0.5, version: 0.9, min_words: 1 };
    assert!(cross_duplicates::<Structure>(&[], &t).is_err());
}
