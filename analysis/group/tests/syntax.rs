//! Syntactic sketches from CoNLL-U fixtures: entity and number
//! substitution leaves the shape, SimHash and MinHash unchanged, an added
//! modifier still shares a band and links, unrelated clauses do not,
//! other tag sets give other sketches, and the layer round-trips through
//! the document graph.

use std::sync::Arc;

use turbo_analysis::export::{Codecs, DocumentGraph, export, import};
use turbo_analysis::*;
use turbo_analysis_layers::{
    Block, Entities, Sentences, Structure, Tokens, WordDependencies, WordLemmas, WordPosTags, Words,
};
use turbo_formats::conllu::{ConlluWordReader, SyntacticSentence};
use turbo_group::hash::{BANDS, band};
use turbo_group::{Sketches, SyntaxSketcher, Thresholds, cross_duplicates};

const UPOS: [&str; 17] = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ",
    "SYM", "VERB", "X",
];
const DEPREL: [&str; 14] = [
    "root", "nsubj", "obj", "iobj", "obl", "nmod", "amod", "det", "case", "aux", "punct", "compound", "flat", "nummod",
];

/// Reads every sentence of `src` into its own document, with one POS tag
/// set and one relation tag set shared by all of them, as a tagger and a
/// parser give: `to_document` builds each sentence's sets from the labels
/// it uses, and sketches are only comparable under one schema.
fn documents(src: &str) -> Vec<Document> {
    let sentences: Vec<SyntacticSentence<'_>> = ConlluWordReader::new(src).unwrap().map(Result::unwrap).collect();
    let upos = TagSet::new("upos", &UPOS);
    let deprel = TagSet::new("deprel", &DEPREL);
    sentences
        .iter()
        .map(|s| {
            let mut doc = s.to_document().unwrap();
            retag(&mut doc, &upos, &deprel);
            doc
        })
        .collect()
}

fn retag(doc: &mut Document, upos: &Arc<TagSet>, deprel: &Arc<TagSet>) {
    let pos = doc.get::<WordPosTags>().unwrap();
    let tags: Vec<Tag> = pos.tags.iter().map(|&t| upos.tag(pos.tagset.label(t)).unwrap()).collect();
    let deps = doc.get::<WordDependencies>().unwrap();
    let head = deps.head.clone();
    let relation: Vec<Tag> = deps.relation.iter().map(|&r| deprel.tag(deps.relations.label(r)).unwrap()).collect();
    assert!(doc.remove(Key::<WordPosTags>::DEFAULT.id()));
    assert!(doc.remove(Key::<WordDependencies>::DEFAULT.id()));
    doc.add(&Key::DEFAULT, WordPosTags { tagset: upos.clone(), tags, prob: None }, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, WordDependencies { head, relation, relations: deprel.clone() }, Provenance::by("t"))
        .unwrap();
}

/// Adds entities over the given token ranges, each `(kind, first token,
/// one past the last token)`.
fn entities(doc: &mut Document, kinds: &Arc<TagSet>, ranges: &[(&str, u32, u32)]) {
    let tokens = doc.get::<Tokens>().unwrap();
    let mut spans = Spans::<Entities>::new(ViewId::SOURCE);
    let mut ranges_out = Vec::new();
    let mut kind = Vec::new();
    for &(k, a, b) in ranges {
        let start = tokens.spans.get(Ix::new(a)).start;
        let end = tokens.spans.get(Ix::new(b - 1)).end;
        spans.push(Span::new(start, end));
        ranges_out.push(IxRange::new(a, b));
        kind.push(kinds.tag(k).unwrap());
    }
    let score = vec![1.0; kind.len()];
    let layer = Entities { spans, tokens: ranges_out, kind, kinds: kinds.clone(), score };
    doc.add(&Key::DEFAULT, layer, Provenance::by("t")).unwrap();
}

fn sketch<L: Layer>(docs: &mut [Document], depth: u8, with_entities: bool) {
    let mut b = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Words>::DEFAULT)
        .given(Key::<WordPosTags>::DEFAULT)
        .given(Key::<WordLemmas>::DEFAULT)
        .given(Key::<WordDependencies>::DEFAULT)
        .given(Key::<L>::DEFAULT);
    if with_entities {
        b = b.given(Key::<Entities>::DEFAULT);
    }
    let pipeline = b.step(Bound::new(SyntaxSketcher::<L, Words>::new(depth)).write(0, "syntax")).build().unwrap();
    for r in pipeline.run(docs) {
        r.unwrap();
    }
}

fn sketches<L: Layer>(doc: &Document) -> &Sketches<L> {
    doc.get_key(&Key::<Sketches<L>>::named("syntax")).unwrap()
}

fn row<L: Layer>(s: &Sketches<L>, i: usize) -> (u64, u64, &[u32]) {
    (s.shape[i], s.simhash[i], s.row(Ix::new(i as u32)))
}

const PAYMENTS: &str = "\
# text = Acme Corp paid $ 5 million on Jan 1 .
1\tAcme\tAcme\tPROPN\t_\t_\t2\tcompound\t_\t_
2\tCorp\tCorp\tPROPN\t_\t_\t3\tnsubj\t_\t_
3\tpaid\tpay\tVERB\t_\t_\t0\troot\t_\t_
4\t$\t$\tSYM\t_\t_\t3\tobj\t_\t_
5\t5\t5\tNUM\t_\t_\t4\tnummod\t_\t_
6\tmillion\tmillion\tNUM\t_\t_\t5\tflat\t_\t_
7\ton\ton\tADP\t_\t_\t8\tcase\t_\t_
8\tJan\tJan\tPROPN\t_\t_\t3\tobl\t_\t_
9\t1\t1\tNUM\t_\t_\t8\tnummod\t_\t_
10\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_

# text = Globex Holdings International paid $ 12 million on Mar 4 .
1\tGlobex\tGlobex\tPROPN\t_\t_\t3\tcompound\t_\t_
2\tHoldings\tHoldings\tPROPN\t_\t_\t3\tcompound\t_\t_
3\tInternational\tInternational\tPROPN\t_\t_\t4\tnsubj\t_\t_
4\tpaid\tpay\tVERB\t_\t_\t0\troot\t_\t_
5\t$\t$\tSYM\t_\t_\t4\tobj\t_\t_
6\t12\t12\tNUM\t_\t_\t5\tnummod\t_\t_
7\tmillion\tmillion\tNUM\t_\t_\t6\tflat\t_\t_
8\ton\ton\tADP\t_\t_\t9\tcase\t_\t_
9\tMar\tMar\tPROPN\t_\t_\t4\tobl\t_\t_
10\t4\t4\tNUM\t_\t_\t9\tnummod\t_\t_
11\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_
";

fn payments() -> Vec<Document> {
    let mut docs = documents(PAYMENTS);
    let kinds = TagSet::new("ner", &["ORG", "MONEY", "DATE"]);
    entities(&mut docs[0], &kinds, &[("ORG", 0, 2), ("MONEY", 3, 6), ("DATE", 7, 9)]);
    entities(&mut docs[1], &kinds, &[("ORG", 0, 3), ("MONEY", 4, 7), ("DATE", 8, 10)]);
    docs
}

#[test]
fn substituted_entities_of_other_lengths_and_other_numbers_sketch_identically() {
    let mut docs = payments();
    sketch::<Sentences>(&mut docs, 2, true);
    let (a, b) = (sketches::<Sentences>(&docs[0]), sketches::<Sentences>(&docs[1]));
    assert_eq!((a.shingle, a.words[0], b.words[0]), (2, 10, 11));
    assert_eq!(row(a, 0), row(b, 0), "shape, SimHash and MinHash agree");
    assert_ne!(a.exact[0], b.exact[0], "the words differ");
    assert_ne!(a.simhash[0], 0);
    assert!(a.row(Ix::new(0)).iter().any(|&v| v != u32::MAX), "the clause has treelets");
}

#[test]
fn without_entities_the_names_tell_the_clauses_apart() {
    let mut docs = documents(PAYMENTS);
    sketch::<Sentences>(&mut docs, 2, false);
    let (a, b) = (sketches::<Sentences>(&docs[0]), sketches::<Sentences>(&docs[1]));
    assert_ne!(a.shape[0], b.shape[0]);
    assert_ne!(a.row(Ix::new(0)), b.row(Ix::new(0)));
}

const CLAUSES: &str = "\
# text = The supplier shall deliver the goods within thirty days .
1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\t_
2\tsupplier\tsupplier\tNOUN\t_\t_\t4\tnsubj\t_\t_
3\tshall\tshall\tAUX\t_\t_\t4\taux\t_\t_
4\tdeliver\tdeliver\tVERB\t_\t_\t0\troot\t_\t_
5\tthe\tthe\tDET\t_\t_\t6\tdet\t_\t_
6\tgoods\tgoods\tNOUN\t_\t_\t4\tobj\t_\t_
7\twithin\twithin\tADP\t_\t_\t9\tcase\t_\t_
8\tthirty\tthirty\tNUM\t_\t_\t9\tnummod\t_\t_
9\tdays\tday\tNOUN\t_\t_\t4\tobl\t_\t_
10\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_

# text = The supplier shall deliver the goods to the buyer within thirty days .
1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\t_
2\tsupplier\tsupplier\tNOUN\t_\t_\t4\tnsubj\t_\t_
3\tshall\tshall\tAUX\t_\t_\t4\taux\t_\t_
4\tdeliver\tdeliver\tVERB\t_\t_\t0\troot\t_\t_
5\tthe\tthe\tDET\t_\t_\t6\tdet\t_\t_
6\tgoods\tgoods\tNOUN\t_\t_\t4\tobj\t_\t_
7\tto\tto\tADP\t_\t_\t9\tcase\t_\t_
8\tthe\tthe\tDET\t_\t_\t9\tdet\t_\t_
9\tbuyer\tbuyer\tNOUN\t_\t_\t4\tobl\t_\t_
10\twithin\twithin\tADP\t_\t_\t12\tcase\t_\t_
11\tthirty\tthirty\tNUM\t_\t_\t12\tnummod\t_\t_
12\tdays\tday\tNOUN\t_\t_\t4\tobl\t_\t_
13\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_

# text = Rain fell on the old roof all night .
1\tRain\train\tNOUN\t_\t_\t2\tnsubj\t_\t_
2\tfell\tfall\tVERB\t_\t_\t0\troot\t_\t_
3\ton\ton\tADP\t_\t_\t6\tcase\t_\t_
4\tthe\tthe\tDET\t_\t_\t6\tdet\t_\t_
5\told\told\tADJ\t_\t_\t6\tamod\t_\t_
6\troof\troof\tNOUN\t_\t_\t2\tobl\t_\t_
7\tall\tall\tDET\t_\t_\t8\tdet\t_\t_
8\tnight\tnight\tNOUN\t_\t_\t2\tobl\t_\t_
9\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_
";

#[test]
fn an_added_modifier_shares_a_band_and_links_while_an_unrelated_clause_does_not() {
    let mut docs = documents(CLAUSES);
    sketch::<Sentences>(&mut docs, 2, false);
    let s: Vec<&Sketches<Sentences>> = docs.iter().map(sketches::<Sentences>).collect();
    let rows: Vec<&[u32]> = s.iter().map(|s| s.row(Ix::new(0))).collect();
    let shared = |x: &[u32], y: &[u32]| (0..BANDS).filter(|&b| band(x, b) == band(y, b)).count();
    assert!(shared(rows[0], rows[1]) >= 1, "the clause with a modifier shares a band");
    assert_eq!(shared(rows[0], rows[2]), 0, "an unrelated clause shares none");
    assert_eq!(shared(rows[1], rows[2]), 0);
    let jaccard = turbo_group::hash::jaccard(rows[0], rows[1]);
    assert!((0.5..0.9).contains(&jaccard), "{jaccard}");

    let batch: Vec<(&Sentences, &Sketches<Sentences>)> =
        docs.iter().zip(&s).map(|(d, s)| (d.get::<Sentences>().unwrap(), *s)).collect();
    let dupes = cross_duplicates(&batch, &Thresholds::default()).unwrap();
    let links: Vec<(u32, u32)> = dupes.links.iter().map(|l| (l.from.doc, l.to.doc)).collect();
    assert_eq!(links, [(0, 1)], "{dupes:?}");
}

#[test]
fn another_tag_set_gives_other_sketches() {
    let mut docs = documents(CLAUSES);
    let mut renamed = docs[0].clone();
    let upos = TagSet::new("penn", &UPOS);
    let deprel = TagSet::new("deprel", &DEPREL);
    retag(&mut renamed, &upos, &deprel);
    docs.push(renamed);
    sketch::<Sentences>(&mut docs, 2, false);
    let (a, b) = (sketches::<Sentences>(&docs[0]), sketches::<Sentences>(&docs[3]));
    assert_eq!(a.words[0], b.words[0]);
    assert_ne!(a.exact[0], b.exact[0]);
    assert_ne!(a.shape[0], b.shape[0]);
    assert_ne!(a.simhash[0], b.simhash[0]);
    assert_ne!(a.row(Ix::new(0)), b.row(Ix::new(0)));
}

#[test]
fn depth_one_leaves_out_the_chains() {
    let mut deep = documents(CLAUSES);
    let mut flat = documents(CLAUSES);
    sketch::<Sentences>(&mut deep, 2, false);
    sketch::<Sentences>(&mut flat, 1, false);
    let (a, b) = (sketches::<Sentences>(&deep[0]), sketches::<Sentences>(&flat[0]));
    assert_eq!((a.shingle, b.shingle), (2, 1));
    assert_eq!((a.exact[0], a.shape[0]), (b.exact[0], b.shape[0]), "the nodes are the same");
    assert_ne!(a.row(Ix::new(0)), b.row(Ix::new(0)), "the shingles are not");
}

#[test]
fn blocks_cut_inside_a_sentence_keep_the_outside_head_as_its_tag_and_an_empty_block_has_no_treelet() {
    let mut docs = documents(CLAUSES);
    let text_len = docs[0].text(ViewId::SOURCE).len() as u32;
    let tokens = docs[0].get::<Tokens>().unwrap();
    let cut = tokens.spans.get(Ix::new(4)).start;
    let mut spans = Spans::<Structure>::new(ViewId::SOURCE);
    spans.push(Span::new(0, cut));
    spans.push(Span::new(cut, text_len));
    spans.push(Span::new(text_len, text_len));
    let blocks = Structure { spans, kind: vec![Block::Paragraph; 3], parent: vec![None; 3], depth: vec![0; 3] };
    docs[0].add(&Key::DEFAULT, blocks, Provenance::by("t")).unwrap();
    let mut docs = vec![docs.swap_remove(0)];
    sketch::<Structure>(&mut docs, 2, false);
    let s = sketches::<Structure>(&docs[0]);
    assert_eq!(s.words, [4, 6, 0]);
    assert_eq!(s.simhash[2], 0);
    assert!(s.row(Ix::new(2)).iter().all(|&v| v == u32::MAX));
    assert_ne!(s.simhash[0], 0, "the first block has arcs to a head outside it");
    assert_ne!(s.simhash[1], 0);
    assert_ne!(row(s, 0), row(s, 1));
}

const ROOTS: &str = "\
# text = Hello
1\tHello\thello\tINTJ\t_\t_\t0\troot\t_\t_

# text = Goodbye
1\tGoodbye\tgoodbye\tINTJ\t_\t_\t0\troot\t_\t_
";

/// A one-word sentence has no treelet: its MinHash row is the empty
/// signature, and two such sentences are not near duplicates of each
/// other, while two clauses with treelets still link.
#[test]
fn sentences_without_treelets_do_not_link_while_clauses_with_them_do() {
    let mut docs = documents(ROOTS);
    docs.extend(documents(CLAUSES).drain(..2));
    sketch::<Sentences>(&mut docs, 2, false);
    let s: Vec<&Sketches<Sentences>> = docs.iter().map(sketches::<Sentences>).collect();
    assert_eq!((s[0].words[0], s[1].words[0]), (1, 1), "the word count is kept");
    assert!(turbo_group::hash::is_empty_signature(s[0].row(Ix::new(0))));
    assert!(turbo_group::hash::is_empty_signature(s[1].row(Ix::new(0))));
    assert_eq!(turbo_group::hash::jaccard(s[0].row(Ix::new(0)), s[1].row(Ix::new(0))), 0.0);
    assert_ne!(s[0].exact[0], s[1].exact[0]);
    let batch: Vec<(&Sentences, &Sketches<Sentences>)> =
        docs.iter().zip(&s).map(|(d, s)| (d.get::<Sentences>().unwrap(), *s)).collect();
    let dupes = cross_duplicates(&batch, &Thresholds::default()).unwrap();
    let links: Vec<(u32, u32)> = dupes.links.iter().map(|l| (l.from.doc, l.to.doc)).collect();
    assert_eq!(links, [(2, 3)], "the one-word sentences are in no link: {dupes:?}");
    assert!(dupes.groups.iter().flatten().all(|m| m.doc >= 2), "{dupes:?}");
}

/// The same one-word sentence under two tag set names is two sketches
/// in two namespaces: not an exact copy, and with no treelet not a near
/// duplicate either.
#[test]
fn the_same_root_only_sentence_under_two_schemas_does_not_link() {
    let mut docs = documents(ROOTS);
    docs.truncate(1);
    let mut renamed = docs[0].clone();
    retag(&mut renamed, &TagSet::new("penn", &UPOS), &TagSet::new("rels", &DEPREL));
    docs.push(renamed);
    sketch::<Sentences>(&mut docs, 2, false);
    let s: Vec<&Sketches<Sentences>> = docs.iter().map(sketches::<Sentences>).collect();
    assert_ne!(s[0].exact[0], s[1].exact[0]);
    assert_ne!(s[0].shape[0], s[1].shape[0]);
    let batch: Vec<(&Sentences, &Sketches<Sentences>)> =
        docs.iter().zip(&s).map(|(d, s)| (d.get::<Sentences>().unwrap(), *s)).collect();
    let dupes = cross_duplicates(&batch, &Thresholds::default()).unwrap();
    assert!(dupes.links.is_empty() && dupes.groups.is_empty(), "{dupes:?}");
}

fn codecs() -> Codecs {
    let mut c = turbo_analysis_layers::export::codecs();
    turbo_group::export::register_codecs(&mut c);
    c
}

#[test]
fn the_syntax_sketches_round_trip_through_the_document_graph() {
    let mut docs = payments();
    sketch::<Sentences>(&mut docs, 2, true);
    let graph = export(&docs[0], &codecs()).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), bytes);
    let (a, b) = (sketches::<Sentences>(&docs[0]), sketches::<Sentences>(&back));
    assert_eq!((a.shingle, &a.words, &a.exact, &a.shape), (b.shingle, &b.words, &b.exact, &b.shape));
    assert_eq!((&a.simhash, &a.minhash), (&b.simhash, &b.minhash));
}
