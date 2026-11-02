//! A tag corpus's test split through the part-of-speech annotator
//! (`turbo_tagger::PosTagger`) on an assembled bundle, scored against
//! gold: the annotator path a pipeline runs, where the receipt's score
//! came from the trainer's own evaluation.
//!
//! ```text
//! cargo run --release -p turbo-trainer-recipes --example pos_annotator_eval -- \
//!     --corpus masc-pos --dir MASC-3.0.0 --bundle <assembled bundle> [--head pos]
//! ```
//!
//! Each sentence is its words joined by single spaces, one token a word
//! and one sentence a document, so the annotator reads the same words the
//! trainer read. It prints the words, the accuracy, and the seconds the
//! annotator took.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use turbo_analysis::{Annotator, Document, Ix, Key, Span, Spans, View, ViewId};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Sentences, Subwords, Tokens};
use turbo_tagger::PosTagger;
use turbo_trainer_recipes::corpus::{Sentence, Task};
use turbo_trainer_recipes::{Error, Result, read_corpus};

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("pos_annotator_eval: {}", e.0);
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<()> {
    let (mut corpus, mut dir, mut bundle, mut head) = (None, None, None, None);
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| Error(format!("{flag} needs a value")))?;
        match flag.as_str() {
            "--corpus" => corpus = Some(value),
            "--dir" => dir = Some(PathBuf::from(value)),
            "--bundle" => bundle = Some(PathBuf::from(value)),
            "--head" => head = Some(value),
            _ => return Err(Error(format!("unknown flag {flag:?}: --corpus, --dir, --bundle or --head"))),
        }
    }
    let need = |name: &str| Error(format!("{name} is required"));
    let corpus: String = corpus.ok_or_else(|| need("--corpus"))?;
    let dir: PathBuf = dir.ok_or_else(|| need("--dir"))?;
    let bundle: PathBuf = bundle.ok_or_else(|| need("--bundle"))?;

    let read = read_corpus(&corpus, &dir, false)?;
    if read.record.task != Task::Tags {
        return Err(Error(format!("{corpus} is not a tag corpus")));
    }
    let tagger = PosTagger::open(&bundle, head.as_deref(), &EncoderOptions::default())
        .map_err(|e| Error(format!("{}: {}", bundle.display(), e.message)))?;

    let started = Instant::now();
    let (mut right, mut words) = (0usize, 0usize);
    for s in &read.splits.test {
        let predicted = tag(&tagger, s)?;
        right += predicted.iter().zip(&s.labels).filter(|(p, g)| p == g).count();
        words += s.words.len();
    }
    let seconds = started.elapsed().as_secs_f64();
    if words == 0 {
        return Err(Error(format!("{corpus}: the test split has no words")));
    }
    println!(
        "{corpus} test: {} sentences, {words} words, accuracy {:.4} ({right} right), annotator {seconds:.1} s",
        read.splits.test.len(),
        right as f64 / words as f64
    );
    Ok(())
}

/// The annotator's tag for each word of `s`.
fn tag(tagger: &PosTagger, s: &Sentence) -> Result<Vec<String>> {
    let text = s.words.join(" ");
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0u32;
    for w in &s.words {
        let end = at + w.len() as u32;
        spans.push(Span::new(at, end));
        at = end + 1;
    }
    let tokens = Tokens { spans, sentence: Some(vec![Ix::new(0); s.words.len()]) };
    let doc = Document::new(text.as_str());
    let view = doc.require(&Key::<View>::DEFAULT).map_err(|e| Error(e.to_string()))?;
    let (tags,) =
        tagger.annotate((view, &tokens, None::<&Subwords>, None::<&Sentences>)).map_err(|e| Error(e.to_string()))?;
    if tags.tags.len() != s.words.len() {
        return Err(Error(format!("{} tags for {} words", tags.tags.len(), s.words.len())));
    }
    Ok(tags.tags.iter().map(|&t| tags.tagset.label(t).to_owned()).collect())
}
