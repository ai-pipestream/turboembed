//! `QuotationAnnotator`: direct quotes attributed before, after and in
//! the middle of the quote, by surface and by dependencies; indirect and
//! mixed speech from `ccomp`; speaker chains and dialog turns; the layer
//! through the analyzer and the document graph bytes.

use turbo_analysis::{
    Analyzer, Document, DocumentGraph, Groups, Ix, IxRange, Key, Provenance, Span, Spans, TagSet, ViewId,
};
use turbo_analysis::{Annotator, export, import};
use turbo_analysis_layers::{Dependencies, Quotation, Quotations, QuoteKind, Sentences, Tokens};
use turbo_coref::{
    Agreement, Animacy, Gender, MentionKind, Mentions, Number, Origin, Person, QuotationAnnotator, QuotationConfig,
    codecs,
};

/// `text` with its whitespace-separated tokens.
fn doc(text: &str) -> Document {
    let mut spans = Spans::new(ViewId(0));
    let mut at = 0;
    for w in text.split(' ') {
        spans.push(Span::new(at as u32, (at + w.len()) as u32));
        at += w.len() + 1;
    }
    let n = spans.len();
    let mut d = Document::new(text);
    let mut sentence = Spans::new(ViewId(0));
    sentence.push(Span::new(0, text.len() as u32));
    d.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sentence }, Provenance::by("test:sentences")).unwrap();
    let tokens = Tokens { spans, sentence: Some(vec![Ix::new(0); n]) };
    d.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test:tokens")).unwrap();
    d
}

/// `(head, relation)` per token; `None` heads the sentence.
fn add_deps(d: &mut Document, rows: &[(Option<u32>, &str)]) {
    let mut labels: Vec<&str> = rows.iter().map(|r| r.1).collect();
    labels.sort_unstable();
    labels.dedup();
    let relations = TagSet::new("ud", &labels);
    let deps = Dependencies {
        head: rows.iter().map(|r| r.0.map(Ix::new)).collect(),
        relation: rows.iter().map(|r| relations.tag(r.1).unwrap()).collect(),
        relations,
    };
    d.add(&Key::<Dependencies>::DEFAULT, deps, Provenance::by("test:deps")).unwrap();
}

/// Mentions over token ranges, each its own chain in order.
fn add_mentions(d: &mut Document, ranges: &[(u32, u32)]) {
    let tokens = d.require(&Key::<Tokens>::DEFAULT).unwrap();
    let spans: Vec<Span> = ranges
        .iter()
        .map(|&(a, b)| Span::new(tokens.spans.get(Ix::new(a)).start, tokens.spans.get(Ix::new(b - 1)).end))
        .collect();
    let n = ranges.len();
    let unk = Agreement::new(Number::Unknown, Gender::Unknown, Animacy::Unknown, Person::Unknown);
    let mut m_spans = Spans::new(ViewId(0));
    for s in spans {
        m_spans.push(s);
    }
    let mentions = Mentions {
        spans: m_spans,
        sentence: vec![Ix::new(0); n],
        tokens: ranges.iter().map(|&(a, b)| IxRange::new(a, b)).collect(),
        head: ranges.iter().map(|&(_, b)| Ix::new(b - 1)).collect(),
        entity: None,
        kind: vec![MentionKind::Proper; n],
        origin: vec![Origin::NounPhrase; n],
        agreement: vec![unk; n],
    };
    d.add(&Key::<Mentions>::DEFAULT, mentions, Provenance::by("test:mentions")).unwrap();
    let groups: Vec<Vec<Ix<Mentions>>> = (0..n as u32).map(|i| vec![Ix::new(i)]).collect();
    d.add(&Key::<Groups<Mentions>>::DEFAULT, Groups::new(groups), Provenance::by("test:coref")).unwrap();
}

fn text_of(d: &Document, s: Option<Span>) -> Option<&str> {
    s.map(|s| &d.text(ViewId(0))[s.start as usize..s.end as usize])
}

fn rows(d: &Document) -> Vec<Quotation> {
    let q = QuotationAnnotator::new().quotations(d).unwrap();
    turbo_analysis::Layer::check(&q).unwrap();
    q.iter().collect()
}

fn show<'a>(d: &'a Document, q: &Quotation) -> (Option<&'a str>, Option<&'a str>, Option<&'a str>, QuoteKind) {
    (text_of(d, Some(q.content)), text_of(d, q.speaker), text_of(d, q.cue_verb), q.kind)
}

#[test]
fn pre_quote_attribution_by_surface() {
    let d = doc("Smith said , \" We grew . \"");
    let q = rows(&d);
    assert_eq!(q.len(), 1);
    assert_eq!(show(&d, &q[0]), (Some("We grew ."), Some("Smith"), Some("said"), QuoteKind::Direct));
    assert_eq!(q[0].confidence, 0.75);
}

#[test]
fn post_quote_attribution_with_a_mention_and_its_chain() {
    let mut d = doc("\" Sales fell , \" said the CFO .");
    add_mentions(&mut d, &[(6, 8)]);
    let q = rows(&d);
    assert_eq!(show(&d, &q[0]), (Some("Sales fell ,"), Some("the CFO"), Some("said"), QuoteKind::Direct));
    assert_eq!(q[0].speaker_chain, Some(0));
}

#[test]
fn a_quote_interrupted_by_its_attribution_gets_it_on_both_halves() {
    let d = doc("\" We grew , \" she said , \" and we hired . \"");
    let q = rows(&d);
    assert_eq!(q.len(), 2);
    assert_eq!(show(&d, &q[0]), (Some("We grew ,"), Some("she"), Some("said"), QuoteKind::Direct));
    assert_eq!(show(&d, &q[1]), (Some("and we hired ."), Some("she"), Some("said"), QuoteKind::Direct));
}

#[test]
fn dependency_attribution_through_parataxis() {
    let mut d = doc("\" We grew , \" she said .");
    add_deps(
        &mut d,
        &[
            (Some(2), "punct"),
            (Some(2), "nsubj"),
            (None, "root"),
            (Some(2), "punct"),
            (Some(2), "punct"),
            (Some(6), "nsubj"),
            (Some(2), "parataxis"),
            (Some(2), "punct"),
        ],
    );
    let q = rows(&d);
    assert_eq!(q.len(), 1);
    assert_eq!(show(&d, &q[0]), (Some("We grew ,"), Some("she"), Some("said"), QuoteKind::Direct));
    assert_eq!(q[0].confidence, 0.9);
}

#[test]
fn indirect_speech_from_a_ccomp() {
    let mut d = doc("The CEO said that revenue rose .");
    add_deps(
        &mut d,
        &[
            (Some(1), "det"),
            (Some(2), "nsubj"),
            (None, "root"),
            (Some(5), "mark"),
            (Some(5), "nsubj"),
            (Some(2), "ccomp"),
            (Some(2), "punct"),
        ],
    );
    let q = rows(&d);
    assert_eq!(q.len(), 1);
    assert_eq!(show(&d, &q[0]), (Some("revenue rose"), Some("The CEO"), Some("said"), QuoteKind::Indirect));
    let off = QuotationAnnotator::new().with_config(QuotationConfig { indirect: false, ..Default::default() });
    assert!(off.quotations(&d).unwrap().iter().next().is_none());
}

#[test]
fn a_complement_quoting_part_of_itself_is_mixed() {
    let mut d = doc("She said the year was \" a turning point \" .");
    add_deps(
        &mut d,
        &[
            (Some(1), "nsubj"),
            (None, "root"),
            (Some(3), "det"),
            (Some(8), "nsubj"),
            (Some(8), "cop"),
            (Some(8), "punct"),
            (Some(8), "det"),
            (Some(8), "amod"),
            (Some(1), "ccomp"),
            (Some(8), "punct"),
            (Some(1), "punct"),
        ],
    );
    let q = rows(&d);
    assert_eq!(q.len(), 1, "{q:?}");
    let (content, speaker, cue, kind) = show(&d, &q[0]);
    assert_eq!(content, Some("the year was \" a turning point \""));
    assert_eq!((speaker, cue, kind), (Some("She"), Some("said"), QuoteKind::Mixed));
}

#[test]
fn an_unattributed_dialog_turn_alternates_speakers() {
    let mut d = doc("Ann said , \" Hi . \" Bob said , \" Hello . \" \" How are you ? \"");
    add_mentions(&mut d, &[(0, 1), (7, 8)]);
    let q = rows(&d);
    let chains: Vec<Option<u32>> = q.iter().map(|q| q.speaker_chain).collect();
    assert_eq!(chains, [Some(0), Some(1), Some(0)]);
    assert_eq!((q[2].speaker, q[2].cue_verb, q[2].confidence), (None, None, 0.5));
    let off = QuotationAnnotator::new().with_config(QuotationConfig { alternate_dialog: false, ..Default::default() });
    let plain: Vec<Option<u32>> = off.quotations(&d).unwrap().iter().map(|q| q.speaker_chain).collect();
    assert_eq!(plain, [Some(0), Some(1), None]);
}

#[test]
fn config_hash_tracks_every_setting() {
    let base = QuotationAnnotator::new().config_hash();
    assert_eq!(base, QuotationAnnotator::new().config_hash());
    for config in [
        QuotationConfig { max_cue_distance: 2, ..Default::default() },
        QuotationConfig { alternate_dialog: false, ..Default::default() },
        QuotationConfig { indirect: false, ..Default::default() },
    ] {
        assert_ne!(QuotationAnnotator::new().with_config(config).config_hash(), base);
    }
}

#[test]
fn analyzer_provenance_and_graph_round_trip() {
    let mut d = doc("\" Sales fell , \" said the CFO .");
    add_mentions(&mut d, &[(6, 8)]);
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Mentions>::DEFAULT)
        .given(Key::<Groups<Mentions>>::DEFAULT)
        .add(QuotationAnnotator::new())
        .build()
        .unwrap();
    let mut batch = vec![d];
    for r in analyzer.run(&mut batch) {
        r.unwrap();
    }
    let d = batch.pop().unwrap();
    let prov = d.provenance(Key::<Quotations>::DEFAULT.id()).unwrap();
    assert_eq!(prov.annotator, QuotationAnnotator::ID);
    assert!(prov.reads.contains(&Key::<Mentions>::DEFAULT.id()));
    assert!(!prov.reads.contains(&Key::<Dependencies>::DEFAULT.id()));
    let q = d.require(&Key::<Quotations>::DEFAULT).unwrap();
    let codecs = codecs();
    let bytes = export(&d, &codecs).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(back.require(&Key::<Quotations>::DEFAULT).unwrap(), q);
}
