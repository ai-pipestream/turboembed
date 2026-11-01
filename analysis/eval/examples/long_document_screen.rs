//! Bounded all-document cut screen, with no retrieval scoring.
//! Usage: long_document_screen <bundle> <cache> <set>...
use serde_json::json;
use std::path::Path;
use turbo_analysis_embed::{Device, Precision};
use turbo_eval::{
    Result,
    analysis::{Analysis, Sentencer},
    chunkers::{Chunker, SentenceWindows, Signals, TurboChunker},
    embedder::{Embedder, Role, TurboEmbedder},
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("long_document_screen <bundle> <cache> <set>...".into());
    }
    let mut embedder = TurboEmbedder::open(Path::new(&args[0]), Device::Cpu, Precision::Model)?;
    let analysis = Analysis { sentences: Sentencer::rules("en"), structure: Some(turbo_structure::Syntax::Plain) };
    let candidate = TurboChunker {
        budget: turbo_chunker::Budget {
            max: 254,
            target: 200,
            size_weight: 1000,
            overhead: embedder.overhead(Role::Document),
        },
        costs: turbo_chunker::Costs::default(),
        signals: Signals::Structure,
        cohesion: None,
        clauses: true,
    };
    let windows = SentenceWindows::new(5)?;
    for name in &args[2..] {
        let set = turbo_eval::sets::find(name)?.load(Path::new(&args[1]))?;
        let mut chunks = [0; 2];
        let mut short = [0; 2];
        let mut changed = 0;
        let mut tokens = Vec::new();
        for d in &set.docs {
            let doc = analysis.analyze(&d.text, &embedder)?;
            let a = candidate.chunk(&doc, &mut embedder)?;
            let b = windows.chunk(&doc, &mut embedder)?;
            for (i, n) in [a.len(), b.len()].into_iter().enumerate() {
                chunks[i] += n;
                short[i] += usize::from(n <= 2);
            }
            changed += usize::from(a != b);
            tokens.push(embedder.count(&d.text, Role::Document));
        }
        tokens.sort_unstable();
        println!(
            "{}",
            json!({"set": name, "sha256": set.sha256, "documents": set.docs.len(),
            "queries": set.queries.len(), "chunks": chunks, "short_documents": short, "changed_documents": changed,
            "tokens_p50": tokens[tokens.len()/2], "tokens_p95": tokens[tokens.len()*95/100],
            "source_commit": turbo_eval::source::BUILD_COMMIT, "manifest_sha256": embedder.identity().bundle.manifest_sha256,
            "candidate": candidate.config(), "reference": windows.config(), "retrieval_scores": false})
        );
    }
    Ok(())
}
