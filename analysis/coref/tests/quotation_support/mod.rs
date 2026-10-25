use turbo_analysis::{
    Analyzer, Bound, Document, Groups, Ix, IxRange, Key, Provenance, Span, Spans, TagSet, View, ViewId,
};
use turbo_analysis_layers::{Dependencies, Quotations, Tokens};
use turbo_coref::{Agreement, Animacy, Gender, MentionKind, Mentions, Number, Origin, Person, QuotationAnnotator};
use turbo_sentdetect::NewlineDetector;
use turbo_tokenize::{SimpleTokenizer, TokenizerOptions};

pub const TEXT: &str = "Zoë said , “ Café grew . ”";
pub fn prepared(text: &str) -> Document {
    let a = Analyzer::builder()
        .add(NewlineDetector::default())
        .add(SimpleTokenizer::new(TokenizerOptions::default()))
        .build()
        .unwrap();
    let mut docs = [Document::new(text)];
    assert_eq!(a.run(&mut docs)[0], Ok(()));
    docs.into_iter().next().unwrap()
}
pub fn add_dependencies(d: &mut Document, rows: &[(Option<u32>, &str)]) {
    let mut names: Vec<_> = rows.iter().map(|r| r.1).collect();
    names.sort_unstable();
    names.dedup();
    let relations = TagSet::new("review:ud", &names);
    let dep: Dependencies<Tokens> = Dependencies {
        head: rows.iter().map(|r| r.0.map(Ix::new)).collect(),
        relation: rows.iter().map(|r| relations.tag(r.1).unwrap()).collect(),
        relations,
    };
    d.add(&Key::DEFAULT, dep, Provenance::by("review:parsed-dependencies")).unwrap();
}
pub fn add_mentions(d: &mut Document, ranges: &[(u32, u32)]) {
    let t = d.get::<Tokens>().unwrap();
    let mut spans = Spans::new(t.spans.view());
    for &(a, b) in ranges {
        spans.push(Span::new(t.spans.get(Ix::new(a)).start, t.spans.get(Ix::new(b - 1)).end));
    }
    let n = ranges.len();
    let m = Mentions {
        spans,
        sentence: vec![Ix::new(0); n],
        tokens: ranges.iter().map(|&(a, b)| IxRange::new(a, b)).collect(),
        head: ranges.iter().map(|&(_, b)| Ix::new(b - 1)).collect(),
        entity: None,
        kind: vec![MentionKind::Proper; n],
        origin: vec![Origin::NounPhrase; n],
        agreement: vec![Agreement::new(Number::Unknown, Gender::Unknown, Animacy::Unknown, Person::Unknown); n],
    };
    d.add(&Key::DEFAULT, m, Provenance::by("review:source-mentions")).unwrap();
    d.add(
        &Key::<Groups<Mentions>>::DEFAULT,
        Groups::<Mentions>::new((0..n as u32).map(|i| vec![Ix::new(i)])),
        Provenance::by("review:source-chains"),
    )
    .unwrap();
}
pub fn run(d: &mut Document) {
    let mut b = Analyzer::builder().given(Key::<Tokens>::DEFAULT);
    if d.get::<Dependencies>().is_some() {
        b = b.given(Key::<Dependencies>::DEFAULT);
    }
    if d.get::<Mentions>().is_some() {
        b = b.given(Key::<Mentions>::DEFAULT).given(Key::<Groups<Mentions>>::DEFAULT);
    }
    let a = b.step(Bound::new(QuotationAnnotator::new())).build().unwrap();
    assert_eq!(a.run(std::slice::from_mut(d))[0], Ok(()));
}
pub fn view(d: &Document) -> &View {
    d.get::<View>().unwrap()
}
pub fn quotes(d: &Document) -> &Quotations {
    d.get::<Quotations>().unwrap()
}
pub fn spans(d: &Document) -> Vec<&str> {
    let t = d.get::<Tokens>().unwrap();
    t.spans.iter().map(|(_, s)| &d.text(ViewId::SOURCE)[s.range()]).collect()
}
