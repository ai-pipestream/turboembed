use turbo_analysis::{Document, Key, Provenance, Span, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Quotation, Quotations, QuoteKind};
use turbo_kserve::proto::result::{OffsetMode, SpanFragment, SpanOutcome};

pub fn span(text: &str, needle: &str) -> Span {
    let start = text.find(needle).unwrap();
    Span::new(start as u32, (start + needle.len()) as u32)
}
pub fn document() -> Document {
    let mut d = Document::new("Élodie said: «é 😀». Léa wrote: «ß 中».");
    let source = d.get::<View>().unwrap().clone();
    let first = span(source.text(), "é 😀");
    let mut b = ViewBuilder::new(source.text());
    b.replace(2, "E");
    b.keep(first.start as usize);
    b.replace(first.end as usize, "e grin");
    b.keep(source.text().len());
    let (text, align) = b.finish();
    d.add(&Key::named("folded"), View::derived(&source, text, align), Provenance::by("test:view")).unwrap();
    let mut b = ViewBuilder::new(source.text());
    b.replace(source.text().len(), "");
    let (text, align) = b.finish();
    d.add(&Key::named("deleted"), View::derived(&source, text, align), Provenance::by("test:view")).unwrap();
    let mut quotes = Quotations::new(ViewId::SOURCE);
    quotes.push(Quotation {
        content: first,
        speaker: Some(span(source.text(), "Élodie")),
        cue_verb: Some(span(source.text(), "said")),
        speaker_chain: Some(u32::MAX),
        kind: QuoteKind::Direct,
        confidence: -0.0,
    });
    quotes.push(Quotation {
        content: span(source.text(), "ß 中"),
        speaker: None,
        cue_verb: Some(span(source.text(), "wrote")),
        speaker_chain: None,
        kind: QuoteKind::Indirect,
        confidence: 1.0,
    });
    d.add(&Key::DEFAULT, quotes, Provenance::by("test:quotes")).unwrap();
    let text = d.text(ViewId(1));
    let mut gold = Quotations::new(ViewId(1));
    gold.push(Quotation {
        content: span(text, "e grin"),
        speaker: Some(span(text, "Elodie")),
        cue_verb: None,
        speaker_chain: Some(0),
        kind: QuoteKind::Mixed,
        confidence: 0.0,
    });
    gold.push(Quotation {
        content: span(text, "ß 中"),
        speaker: None,
        cue_verb: None,
        speaker_chain: Some(7),
        kind: QuoteKind::Direct,
        confidence: 0.8,
    });
    d.add(&Key::named("gold"), gold, Provenance::by("test:quotes-gold")).unwrap();
    d
}
/// `span` of `from` on `target` with what projection did to it (#1907),
/// exact on its own view.
pub fn projected(doc: &Document, from: ViewId, target: ViewId, span: Span, mode: OffsetMode) -> SpanFragment {
    let (mapped, outcome) = if from == target {
        (span, SpanOutcome::Exact)
    } else {
        let p = doc.map_span_outcome(from, target, span).unwrap();
        let outcome = match p.outcome {
            turbo_analysis::SpanOutcome::Exact => SpanOutcome::Exact,
            turbo_analysis::SpanOutcome::Widened => SpanOutcome::Widened,
            turbo_analysis::SpanOutcome::Deleted => SpanOutcome::Deleted,
        };
        (p.span, outcome)
    };
    let text = doc.text(target);
    let units = |at: u32| match mode {
        OffsetMode::Utf8 => at,
        OffsetMode::Utf16 => text[..at as usize].encode_utf16().count() as u32,
        OffsetMode::CodePoint => text[..at as usize].chars().count() as u32,
    };
    SpanFragment { start: units(mapped.start), end: units(mapped.end), outcome: outcome as i32, members: Vec::new() }
}
