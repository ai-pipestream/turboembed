//! Inspect original MASC annotation representability without training or text output.
use std::collections::BTreeMap;
use std::path::PathBuf;
use turbo_formats::masc::MascCorpus;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::args().nth(1).ok_or("corpus directory required")?);
    let digest = turbo_train_recipes::masc::check_corpus(&root, turbo_train_recipes::masc::RELEASE, false)?;
    let split = turbo_train_recipes::masc::SPLIT.apply(MascCorpus::read(&root)?, &root)?;
    println!("sha256={} files={}", digest.sha256, digest.files);
    let mut counts = BTreeMap::<String, usize>::new();
    for (part, corpus) in [("train", &split.train), ("test", &split.held_out)] {
        for doc in corpus.documents().iter().filter(|d| d.has_named_entities()) {
            *counts.entry(format!("{part}:documents")).or_default() += 1;
            for e in doc.unplaced_entities() {
                *counts.entry(format!("{part}:sentence-unplaced:{:?}", e.cause)).or_default() += 1;
            }
            let Some(all) = doc.document_tokens() else { continue };
            let spans = doc.token_spans(all)?;
            let mut seen = Vec::<&turbo_formats::masc::MascEntityPiece>::new();
            for piece in all.entity_pieces() {
                let e = &piece.entity;
                *counts.entry(format!("{part}:type:{:?}", e.kind)).or_default() += 1;
                let mut issues = Vec::new();
                if piece.continues {
                    issues.push("incomplete");
                }
                if !e.is_contiguous() {
                    issues.push("discontinuous");
                }
                if e.tokens.iter().any(|&i| spans[i as usize].start == spans[i as usize].end) {
                    issues.push("zero-width-token");
                }
                for prior in &seen {
                    let p = &prior.entity;
                    if !e.tokens.iter().any(|i| p.tokens.contains(i)) {
                        continue;
                    }
                    let category = if p.tokens == e.tokens {
                        if p.kind == e.kind { "duplicate-same-type" } else { "same-span-different-type" }
                    } else if (p.start <= e.start && e.end <= p.end) || (e.start <= p.start && p.end <= e.end) {
                        if p.kind == e.kind { "nested-same-type" } else { "nested-different-type" }
                    } else if p.kind == e.kind {
                        "crossing-same-type"
                    } else {
                        "crossing-different-type"
                    };
                    *counts.entry(format!("{part}:pairs:{category}")).or_default() += 1;
                    println!(
                        "{part}:{} ne-n{} {:?} {:?} / ne-n{} {:?} {:?}: {category}",
                        doc.path().strip_prefix(&root)?.display(),
                        piece.id,
                        e.kind,
                        e.tokens,
                        prior.id,
                        p.kind,
                        p.tokens
                    );
                }
                seen.push(piece);
                for issue in issues {
                    *counts.entry(format!("{part}:document:{issue}")).or_default() += 1;
                    println!(
                        "{part}:{} ne-n{} {issue} tokens={:?}",
                        doc.path().strip_prefix(&root)?.display(),
                        piece.id,
                        e.tokens
                    );
                }
            }
            for warning in doc.warnings() {
                *counts
                    .entry(format!("{part}:warning:{warning:?}").split(" {").next().unwrap().to_owned())
                    .or_default() += 1;
            }
        }
    }
    for (key, value) in counts {
        println!("{key}={value}");
    }
    Ok(())
}
