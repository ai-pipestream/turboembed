use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::Path;

use turbo_analysis::Annotator;
use turbo_eval::model_gate::{ModelGateRecord, ModelId};
use turbo_formats::conllu::{ConlluReader, ConlluSentenceReader};
use turbo_formats::masc::MascCorpus;
use turbo_sentdetect::{RuleSentenceDetector, RuleSplitter};
use turbo_train_recipes::{masc, segmentation, segmentation::HeldOut, sentence_gold, ud};
use turbo_zoo::Sha256;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage:
  turbo-sentence-eval ud <local-test.conllu> <expected-sha256> <language> <release/test-set>
  turbo-sentence-eval masc <local-MASC-3.0.0-directory>
Reads local data only. The caller must already have permission to use it.
Writes a gold-label rule-vs-newline gate record to stdout, never corpus text.
Exit status: 0 passed gate; 2 failed gate; 1 invalid input or execution error.";

pub fn run(args: Vec<OsString>) -> Result<bool> {
    if args.as_slice() == [OsString::from("--help")] {
        println!("{USAGE}");
        return Ok(true);
    }
    let record = match args.first().and_then(|s| s.to_str()) {
        Some("ud") if args.len() == 5 => {
            let expected = text(&args[2])?;
            if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                return Err("expected SHA-256 must be 64 lowercase hexadecimal characters".into());
            }
            let bytes = std::fs::read(&args[1])?;
            let found = Sha256::of(&bytes).to_string();
            if found != expected {
                return Err(format!("corpus SHA-256 mismatch: expected {expected}, found {found}").into());
            }
            let source = std::str::from_utf8(&bytes)?;
            let language = text(&args[3])?;
            validate_language(language)?;
            let set = text(&args[4])?;
            if set.trim().is_empty() || set.chars().any(char::is_control) {
                return Err("set identity must name the corpus release and test split".into());
            }
            let held_out = conllu_samples(source)?;
            score(&held_out, language, set, &found)?
        }
        Some("masc") if args.len() == 2 => {
            let root = Path::new(&args[1]);
            let digest = masc::check_corpus(root, masc::RELEASE, false)?;
            let split = masc::SPLIT.apply(MascCorpus::read(root)?, root)?;
            let held_out = segmentation::SENTENCE.held_out(&split.held_out)?;
            score(&held_out, "en", "MASC 3.0.0 held-out every tenth document; Twitter excluded", &digest.sha256)?
        }
        _ => return Err(USAGE.into()),
    };
    std::io::stdout().lock().write_all(record.to_json().as_bytes())?;
    Ok(record.result.pass)
}

fn text(value: &OsString) -> Result<&str> {
    value.to_str().ok_or_else(|| "metadata arguments must be UTF-8".into())
}

fn validate_language(language: &str) -> Result<()> {
    if !turbo_sentdetect::rules::LANGUAGES.iter().any(|entry| entry.code == language) {
        return Err("language must be an exact supported lowercase language tag".into());
    }
    Ok(())
}

fn conllu_samples(source: &str) -> Result<HeldOut<'_>> {
    // The general reader substitutes `null` for a missing # text. A measured
    // corpus must never score that placeholder or silently omit a sentence.
    for (index, sentence) in ConlluReader::new(source)?.enumerate() {
        let sentence = sentence?;
        if sentence.text.as_ref().is_none_or(|text| text.as_str().trim().is_empty()) {
            return Err(format!("CoNLL-U sentence {} requires nonblank # text", index + 1).into());
        }
    }
    Ok(HeldOut::Sentences(
        ConlluSentenceReader::new(source, ud::SENTENCES_PER_SAMPLE)?.collect::<std::result::Result<_, _>>()?,
    ))
}

fn score(held_out: &HeldOut<'_>, language: &str, set: &str, corpus_sha256: &str) -> Result<ModelGateRecord> {
    let HeldOut::Sentences(samples) = held_out else { return Err("sentence samples required".into()) };
    let splitter = RuleSplitter::new(language);
    let predictions = samples
        .iter()
        .map(|sample| {
            let mut spans = Vec::new();
            splitter.try_split(sample.document(), &mut spans)?;
            Ok(spans)
        })
        .collect::<std::result::Result<Vec<_>, turbo_analysis::Error>>()?;
    let detector = RuleSentenceDetector::new(language);
    // The binary hash identifies the actual code/data compiled for this run.
    // The candidate identity also makes the rule configuration explicit.
    let candidate = ModelId {
        id: format!(
            "{}:v{}:{language}:config-{:016x}:group-{}",
            RuleSentenceDetector::ID,
            RuleSentenceDetector::VERSION,
            detector.config_hash(),
            ud::SENTENCES_PER_SAMPLE,
        ),
        sha256: executable_sha256()?,
    };
    Ok(sentence_gold::gate(candidate, held_out, &predictions, set, corpus_sha256)?.record)
}

fn executable_sha256() -> Result<String> {
    let mut file = std::fs::File::open(std::env::current_exe()?)?;
    let mut digest = turbo_sha256::Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            return Ok(digest.finish().to_hex());
        }
        digest.update(&buffer[..n]);
    }
}
