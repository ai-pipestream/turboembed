//! Scores the span detector on fixture files with gold spans.
//!
//! cargo run --release -p turbo-langspan --example evaluate -- \
//!     <langdetect-183.bin> <fixture file>... [--switch space,punctuation,sentence] [--verbose]

use turbo_langdetect::{LanguageDetector, LanguageModel};
use turbo_langspan::decode::SwitchCost;
use turbo_langspan::gold::{Scores, parse};
use turbo_langspan::{Run, Settings, SpanDetector, SpanScratch};
use turbo_maxent::ModelFile;

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("a model path");
    let mut files = Vec::new();
    let mut settings = Settings::default();
    let mut verbose = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--switch" => {
                let v: Vec<f64> = args.next().expect("three costs").split(',').map(|x| x.parse().unwrap()).collect();
                settings.switch = SwitchCost { space: v[0], punctuation: v[1], sentence: v[2] };
            }
            "--verbose" => verbose = true,
            "--clause" => settings.clause_letters = args.next().unwrap().parse().unwrap(),
            _ => files.push(a),
        }
    }
    let detector = LanguageDetector::new(LanguageModel::open(ModelFile::read(model).unwrap()).unwrap());
    let spans = SpanDetector::new(detector, settings);
    let mut scratch = SpanScratch::new();
    let mut runs: Vec<Run> = Vec::new();
    for file in files {
        let docs = parse(&std::fs::read_to_string(&file).unwrap()).unwrap();
        let mut scores = Scores::default();
        for doc in &docs {
            spans.detect(&doc.text, None, None, &mut scratch, &mut runs);
            let predicted: Vec<_> = runs.iter().map(|r| (r.span, spans.code(r))).collect();
            let before = scores.exact_documents;
            scores.add(doc, &predicted);
            if verbose && scores.exact_documents == before {
                println!("{}", doc.id);
                for (s, c) in &doc.spans {
                    println!("  gold {c} {:?}", &doc.text[s.range()]);
                }
                for r in &runs {
                    println!("  got  {} {:.2} {:?}", spans.code(r), r.confidence, &doc.text[r.span.range()]);
                }
            }
        }
        println!("{file}\n{scores}\n");
    }
}
