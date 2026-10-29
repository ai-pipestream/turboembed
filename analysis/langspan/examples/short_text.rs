//! Show script spans for a supplied text without loading a language model.
use turbo_analysis::{Analyzer, Document};
use turbo_analysis_layers::LanguageSpans;
use turbo_langspan::ScriptDetector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = std::env::args().nth(1).ok_or("usage: short_text <text>")?;
    let mut document = Document::new(text);
    let analyzer = Analyzer::builder().add(ScriptDetector).build()?;
    analyzer.run(std::slice::from_mut(&mut document)).remove(0)?;
    for span in document.get::<LanguageSpans>().ok_or("missing script spans")?.iter() {
        println!("{}..{}\t{:?}\t{}", span.span.start, span.span.end, span.script, span.code);
    }
    Ok(())
}
