//! Run the real token classifier on inputs exported by crf_namefind.
//! Usage: masc_predict BUNDLE HELD_OUT_JSON OUTPUT_JSON
//! Reads corpus text locally; writes only token spans and provenance.
use std::error::Error;
use std::path::Path;
use turbo::bundle::{Bundle, sha256_hex};
use turbo_analysis_embed::Precision;
use turbo_analysis_layers::Tag;
use turbo_namefind_dl::{NameFinderDl, NameFinderDlOptions};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: masc_predict BUNDLE HELD_OUT_JSON OUTPUT_JSON".into());
    }
    let bundle = Path::new(&args[0]);
    let output = Path::new(&args[2]);
    if output.exists() {
        return Err("output already exists".into());
    }
    let input = std::fs::read(&args[1])?;
    let rows: Vec<serde_json::Value> = serde_json::from_slice(&input)?;
    if rows.is_empty() {
        return Err("empty held-out input".into());
    }
    let mut options = NameFinderDlOptions::default();
    options.encoder.precision = Precision::Exact;
    options.encoder.cpu_threads = 4;
    let finder = NameFinderDl::open(bundle, &options)?;
    let manifest = Bundle::open(bundle)?;
    let mut predicted = Vec::new();
    for (batch, rows) in rows.chunks(8).enumerate() {
        let mut texts = Vec::new();
        for row in rows {
            let mut text = String::new();
            let mut spans = Vec::new();
            let tokens = row["tokens"].as_array().ok_or("tokens must be an array")?;
            if tokens.is_empty() {
                return Err("empty token sample".into());
            }
            for token in tokens {
                let token = token.as_str().ok_or("token must be text")?;
                if token.is_empty() {
                    return Err("empty token".into());
                }
                if !text.is_empty() {
                    text.push(' ');
                }
                spans.push((text.len(), text.len() + token.len()));
                text.push_str(token);
            }
            texts.push((text, spans));
        }
        let inputs: Vec<_> = texts.iter().map(|(t, _)| (t.as_str(), vec![(0, t.len())])).collect();
        let found = finder.find(&inputs)?;
        if found.len() != texts.len() {
            return Err("prediction count differs from input count".into());
        }
        for ((text, spans), names) in texts.iter().zip(found) {
            let mut entities = Vec::new();
            for name in names {
                if name.start >= name.end || name.end > text.len() {
                    return Err("entity byte span is outside the input".into());
                }
                let kind = match finder.kinds().label(Tag(name.kind as u16)) {
                    "PER" => "person",
                    "ORG" => "organization",
                    "LOC" => "location",
                    "MISC" => "misc",
                    _ => return Err("unsupported model entity label".into()),
                };
                let first = spans.iter().position(|&(_, e)| e > name.start).ok_or("entity starts beyond the tokens")?;
                let last = spans.iter().rposition(|&(s, _)| s < name.end).ok_or("entity ends before the tokens")?;
                if first > last {
                    return Err("entity overlaps no token".into());
                }
                let entity = (first, last + 1, kind);
                if !entities.contains(&entity) {
                    entities.push(entity);
                }
            }
            predicted.push(entities);
        }
        if batch % 50 == 0 {
            eprintln!("{} samples complete", predicted.len());
        }
    }
    let record = serde_json::json!({"version":1,"input_sha256":sha256_hex(&input),
        "bundle_sha256":manifest.manifest_sha256,"model":manifest.manifest.model,
        "backend":"cpu","precision":"Exact","cpu_threads":4,"predicted":predicted});
    std::fs::write(output, serde_json::to_vec(&record)?)?;
    Ok(())
}
