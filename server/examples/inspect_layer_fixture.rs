//! Writes the named/default positional graph used by the demo browser tests.
use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_kserve::results::document::codecs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected output graph path")?;
    let mut doc = Document::new("é 😀");
    for (instance, span) in [("", Span::new(0, 2)), ("gold", Span::new(3, 7))] {
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(span);
        doc.add(&Key::named(instance), Tokens { spans, sentence: None }, Provenance::by("browser-test"))?;
    }
    let graph = turbo_analysis::export::export(&doc, codecs())?.to_bytes()?;
    std::fs::write(path, graph)?;
    Ok(())
}
