//! The `redacted` view and its `turbo:pii-edits` layer: every strategy
//! writes a real derived view, a step stores it with its edits, and spans
//! go source -> redacted -> source as containment, through chains of
//! views built with `ViewBuilder`.

use turbo_analysis::{
    Analyzer, Annotator, Document, Ix, Key, Layer, Provenance, Span, SpanOutcome, View, ViewBuilder, ViewId,
};
use turbo_pii::rewrite::{
    Abbreviator, FormatRedactor, HmacTokenizer, MapReplacer, MaskPolicy, Pseudonymizer, REDACTED, RedactedView,
    RewritePolicy, RewriteStrategy, Sha256Tokenizer, StaticReplacer, Truncate,
};
use turbo_pii::{Pii, PiiDetector, PiiEdit, PiiEdits, PiiEvidence, PiiKind, PiiMention, PiiRule, PiiStrategyKind};

const TEXT: &str = "Für Zoë: jane@example.com, (555) 123-4567, Karte 4111 1111 1111 1111 😀.";

fn strategies() -> Vec<RewriteStrategy> {
    vec![
        MaskPolicy::new('*').unwrap().into(),
        Pseudonymizer::default().into(),
        HmacTokenizer::new(b"0123456789abcdef0123456789abcdef").unwrap().into(),
        FormatRedactor::new("<%t-%v>").unwrap().into(),
        StaticReplacer::new("[X]").into(),
        Sha256Tokenizer::new().into(),
        MapReplacer::new([("jane@example.com", "J")], false).unwrap().into(),
        Truncate::keep_end(2).into(),
        Abbreviator::new().into(),
    ]
}

fn boundaries(text: &str) -> Vec<u32> {
    (0..=text.len()).filter(|&i| text.is_char_boundary(i)).map(|i| i as u32).collect()
}

/// Every span of view `from` (on character boundaries) maps to `to` and
/// back as containment, and the edits layer agrees with the mentions.
#[track_caller]
fn assert_round_trips(doc: &Document, from: ViewId, to: ViewId) {
    let bounds = boundaries(doc.text(from));
    for (i, &s) in bounds.iter().enumerate() {
        for &e in &bounds[i..] {
            let span = Span::new(s, e);
            let there = doc.map_span_outcome(from, to, span).unwrap();
            let back = doc.map_span(to, from, there.span).unwrap();
            // Text the target view deleted comes back as its empty
            // anchor; everything else comes back covering itself.
            if there.outcome == SpanOutcome::Deleted {
                assert!(there.span.is_empty() && !span.is_empty(), "{span:?} -> {there:?}");
            } else {
                assert!(back.contains(span), "{span:?} -> {there:?} -> {back:?}");
            }
        }
    }
}

fn redacted(doc: &Document) -> (&View, &PiiEdits) {
    let view = doc.get_key(&REDACTED).expect("the redacted view");
    let edits = doc.get_key(&Key::<PiiEdits>::named(REDACTED.name)).expect("its edits");
    (view, edits)
}

#[test]
fn the_step_stores_the_view_and_its_edits_for_every_strategy() {
    for strategy in strategies() {
        let kind = strategy.kind();
        let step = RedactedView::new(RewritePolicy::new(strategy).with_revision(41)).bound();
        let analyzer = Analyzer::builder().add(PiiDetector::cursor()).step(step).build().unwrap();
        let mut docs = vec![Document::new(TEXT)];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        let doc = &docs[0];
        let pii = doc.get::<Pii>().unwrap();
        let (view, edits) = redacted(doc);
        assert_eq!(view.parent(), Some(ViewId::SOURCE), "{kind:?}");
        assert_ne!(view.id(), ViewId::PENDING);
        assert_eq!(edits.spans.view(), view.id(), "{kind:?}: the edits moved onto the stored view");
        assert_eq!(edits.policy_revision, 41);
        assert_eq!(edits.len(), pii.len(), "{kind:?}: the cursor pack finds no overlaps here");
        for (i, e) in edits.edits().enumerate() {
            assert_eq!(e.members, &[Ix::<Pii>::new(i as u32)][..]);
            assert_eq!(e.source, pii.spans.get(Ix::new(i as u32)));
            assert_eq!((e.rule, e.strategy), (PiiRule::Default, kind));
            assert_eq!(doc.map_span(ViewId::SOURCE, view.id(), e.source).unwrap(), e.span);
        }
        assert_round_trips(doc, ViewId::SOURCE, view.id());
    }
}

#[test]
fn a_kind_override_is_recorded_as_its_rule() {
    let policy = RewritePolicy::new(Pseudonymizer::default()).with(PiiKind::Card, StaticReplacer::new("[CARD]"));
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(RedactedView::new(policy).bound()).build().unwrap();
    let mut docs = vec![Document::new(TEXT)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let (view, edits) = redacted(&docs[0]);
    assert!(view.text().contains("[CARD]"));
    let card = edits.edits().find(|e| e.rule == PiiRule::Kind(PiiKind::Card)).expect("the card edit");
    assert_eq!(card.strategy, PiiStrategyKind::Static);
    assert_eq!(&view.text()[card.span.range()], "[CARD]");
    assert!(edits.edits().filter(|e| e.rule == PiiRule::Default).all(|e| e.strategy == PiiStrategyKind::Pseudonym));
}

/// A document whose view 1 is `parent` written over the source by
/// `build`, with the mentions over view 1 and the redacted view over it.
fn chain(
    source: &str,
    build: impl FnOnce(&mut ViewBuilder<'_>),
    mentions: &[(u32, u32, PiiKind)],
    strategy: RewriteStrategy,
) -> (Document, ViewId, ViewId) {
    let mut doc = Document::new(source);
    let src = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source);
    build(&mut b);
    let (text, alignment) = b.finish();
    let norm = Key::<View>::named("norm");
    doc.add(&norm, View::derived(&src, text, alignment), Provenance::by("test")).unwrap();
    let mid = doc.get_key(&norm).unwrap().clone();
    let mut pii = Pii::new(mid.id());
    let detector = pii.detector("test", "1").unwrap();
    for &(s, e, kind) in mentions {
        pii.push(PiiMention {
            span: Span::new(s, e),
            kind,
            detector,
            evidence: &[PiiEvidence::Shape],
            score: None,
            normalized: "v",
        });
    }
    let pii_key = Key::<Pii>::named("norm");
    doc.add(&pii_key, pii, Provenance::by("test")).unwrap();
    let step = RedactedView::new(RewritePolicy::new(strategy));
    let (view, edits) = step.annotate((&mid, doc.get_key(&pii_key).unwrap())).unwrap();
    doc.add(&REDACTED, view, Provenance::by("test")).unwrap();
    let out = doc.get_key(&REDACTED).unwrap().id();
    let mut edits = edits;
    edits.bind_pending_view(out);
    let mut p = Provenance::by("test");
    p.reads.push(pii_key.id());
    doc.add(&Key::<PiiEdits>::named(REDACTED.name), edits, p).unwrap();
    (doc, mid.id(), out)
}

#[test]
fn combining_marks_orphaned_by_per_scalar_masking() {
    // "Jose" + U+0301: the mention ends before the combining mark, so
    // masking leaves the mark after the mask.
    let source = "name Jose\u{301} ok";
    let (doc, mid, out) = chain(source, |_| {}, &[(5, 9, PiiKind::Email)], MaskPolicy::new('*').unwrap().into());
    assert_eq!(doc.text(out), "name ****\u{301} ok");
    assert_round_trips(&doc, ViewId::SOURCE, out);
    assert_round_trips(&doc, mid, out);
}

#[test]
fn emoji_and_zwj_sequences() {
    let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
    let source = format!("{family} a@b.co {family}");
    let start = family.len() as u32 + 1;
    let (doc, _, out) = chain(&source, |_| {}, &[(start, start + 6, PiiKind::Email)], Pseudonymizer::default().into());
    assert_eq!(doc.text(out), format!("{family} EMAIL-1 {family}"));
    assert_round_trips(&doc, ViewId::SOURCE, out);
}

#[test]
fn ligature_expansion_across_a_mention_boundary() {
    // The source's "\u{FB01}" (fi) becomes "fi" in view 1, and the
    // mention starts at its "i": the ligature is split by the mention.
    let source = "x \u{FB01}le@a.co y";
    let lig = Span::new(2, 5);
    let (doc, mid, out) = chain(
        source,
        |b| {
            b.keep(lig.start as usize);
            b.replace(lig.end as usize, "fi");
        },
        &[(3, 11, PiiKind::Email)],
        StaticReplacer::new("[E]").into(),
    );
    assert_eq!(doc.text(mid), "x file@a.co y");
    assert_eq!(doc.text(out), "x f[E] y");
    // The ligature's bytes widen to the whole edit chain, never lose bytes.
    let there = doc.map_span(ViewId::SOURCE, out, lig).unwrap();
    assert!(doc.map_span(out, ViewId::SOURCE, there).unwrap().contains(lig));
    assert_round_trips(&doc, ViewId::SOURCE, out);
    assert_round_trips(&doc, mid, out);
}

#[test]
fn overlapping_members_coalesce_and_are_listed() {
    let source = "id 0123456789 end";
    let mentions = [(3, 13, PiiKind::Phone), (5, 9, PiiKind::Email), (8, 11, PiiKind::UsSsn)];
    let (doc, _, out) = chain(source, |_| {}, &mentions, Pseudonymizer::default().into());
    let edits = doc.get_key(&Key::<PiiEdits>::named(REDACTED.name)).unwrap();
    assert_eq!(edits.len(), 1);
    let e = edits.get(Ix::new(0));
    assert_eq!(e.members, &[Ix::new(0), Ix::new(1), Ix::new(2)][..]);
    assert_eq!(e.source, Span::new(3, 13));
    assert_round_trips(&doc, ViewId::SOURCE, out);
}

#[test]
fn unequal_replacement_lengths() {
    let source = "a x@y.io b verylongname@example.org c";
    let mentions = [(2, 8, PiiKind::Email), (11, 34, PiiKind::Email)];
    let (doc, _, out) = chain(source, |_| {}, &mentions, Sha256Tokenizer::new().into());
    let edits = doc.get_key(&Key::<PiiEdits>::named(REDACTED.name)).unwrap();
    let grew = edits.get(Ix::new(0));
    let shrank = edits.get(Ix::new(1));
    assert!(grew.span.len() > grew.source.len() || shrank.span.len() < shrank.source.len());
    assert_round_trips(&doc, ViewId::SOURCE, out);
}

#[test]
fn deletion_and_insertion_anchors() {
    // View 1 deletes "zz " and inserts "+" before the mention; empty
    // spans at both edits stay empty and land before them.
    let source = "zz a@b.co q";
    let (doc, mid, out) = chain(
        source,
        |b| {
            b.replace(3, "");
            b.replace(3, "+");
        },
        &[(1, 7, PiiKind::Email)],
        StaticReplacer::new("[E]").into(),
    );
    assert_eq!(doc.text(mid), "+a@b.co q");
    assert_eq!(doc.text(out), "+[E] q");
    let deleted = doc.map_span_outcome(ViewId::SOURCE, out, Span::new(0, 2)).unwrap();
    assert_eq!(deleted.outcome, SpanOutcome::Deleted);
    assert!(deleted.span.is_empty());
    let anchor = doc.map_span(mid, out, Span::new(0, 0)).unwrap();
    assert_eq!(anchor, Span::new(0, 0));
    let mention = doc.map_span_outcome(mid, out, Span::new(1, 7)).unwrap();
    assert_eq!((mention.span, mention.outcome), (Span::new(1, 4), SpanOutcome::Exact));
    let inside = doc.map_span_outcome(mid, out, Span::new(2, 3)).unwrap();
    assert_eq!(inside.outcome, SpanOutcome::Widened);
    // The rewrite step itself deletes nothing, so across it every span
    // comes back covering itself.
    assert_round_trips(&doc, mid, out);
    // View 1's deleted bytes belong to neither neighbour: a source span
    // starting in them comes back starting after them, so containment
    // from the source holds for spans clear of that deletion (an empty
    // span at its edge is an anchor and lands before it).
    let bounds = boundaries(source);
    for (i, &s) in bounds.iter().enumerate().filter(|&(_, &s)| s >= 3) {
        for &e in bounds[i..].iter().filter(|&&e| s > 3 || e > s) {
            let span = Span::new(s, e);
            let back = doc.map_span(out, ViewId::SOURCE, doc.map_span(ViewId::SOURCE, out, span).unwrap()).unwrap();
            assert!(back.contains(span), "{span:?} -> {back:?}");
        }
    }
    let partly = doc.map_span_outcome(ViewId::SOURCE, out, Span::new(0, 4)).unwrap();
    assert_eq!(doc.map_span(out, ViewId::SOURCE, partly.span).unwrap(), Span::new(3, 9));
    // That loss is never silent: a span starting in the deleted bytes is
    // flagged, never exact; widened when some of it survives, deleted
    // when none does.
    assert_eq!(partly.outcome, SpanOutcome::Widened);
    for (i, &s) in bounds.iter().enumerate().filter(|&(_, &s)| s < 3) {
        for &e in bounds[i..].iter().filter(|&&e| e > s) {
            let p = doc.map_span_outcome(ViewId::SOURCE, out, Span::new(s, e)).unwrap();
            let want = if e <= 3 { SpanOutcome::Deleted } else { SpanOutcome::Widened };
            assert_eq!(p.outcome, want, "{s}..{e} -> {:?}", p.span);
        }
    }
    let frags = doc.project_fragments(ViewId::SOURCE, out, [Span::new(1, 4), Span::new(5, 6)]).unwrap();
    assert!(frags.iter().all(|f| f.members.iter().all(|(i, o)| *i != 0 || *o == SpanOutcome::Widened)));
    assert!(frags.iter().all(|f| f.outcome != SpanOutcome::Exact || f.members.iter().all(|(i, _)| *i != 0)));
}

#[test]
fn edits_that_disagree_with_the_mentions_are_refused() {
    let (mut doc, _, out) = chain("a x@y.io b", |_| {}, &[(2, 8, PiiKind::Email)], StaticReplacer::new("[E]").into());
    let _ = doc.get_key(&Key::<PiiEdits>::named(REDACTED.name)).unwrap();
    // A second edits layer over the same view naming no mention under
    // its span is refused by the document.
    let mut e = PiiEdits::new(out, 0);
    e.push(PiiEdit {
        span: Span::new(2, 5),
        source: Span::new(2, 8),
        members: &[],
        rule: PiiRule::Default,
        strategy: PiiStrategyKind::Static,
        rule_id: "",
    });
    let mut p = Provenance::by("test");
    p.reads.push(Key::<Pii>::named("norm").id());
    assert!(doc.add(&Key::<PiiEdits>::named("other"), e, p).is_err());
}
