//! A model file written in Rust, read by the JVM: the goldens and decision
//! point dumps under `analysis/goldens/data/langdetect/langdetect-fixture-rust`
//! are OpenNLP's `LanguageDetectorME` running
//! `analysis/goldens/data/fixtures/langdetect-fixture-rust.bin`, which
//! turbo-train-components trains from `inputs/langdetect/samples.txt`
//! (its `tests/fixtures.rs` checks that the file is the one it writes).
//! The fixture is checked in, so these tests always run.

mod common;

use std::slice;

use turbo_analysis::{Analyzer, Document, Key, Status};
use turbo_goldens::{Golden, Tolerance, assert_matches_golden, data_dir, segmentation_shapes};
use turbo_langdetect::{LanguageDetector, LanguageModel, ProbingConfig, ProbingContext, Ranking, Scratch};
use turbo_maxent::ModelFile;

use common::{goldens_in, read_dump};

const FIXTURE: &str = "langdetect-fixture-rust";
const CHUNK_50: ProbingConfig =
    ProbingConfig { max_length: 250, chunk_size: 50, min_consec_improvements: 2, min_diff: 0.20 };

fn goldens() -> Vec<(String, String, std::path::PathBuf)> {
    goldens_in(&data_dir().join("langdetect").join(FIXTURE))
}

fn detector() -> LanguageDetector {
    let path = data_dir().join("fixtures").join(format!("{FIXTURE}.bin"));
    LanguageDetector::new(LanguageModel::open(ModelFile::read(path).unwrap()).unwrap())
}

fn input(golden: &Golden) -> String {
    golden.input_text(data_dir()).unwrap()
}

#[test]
fn the_jvm_reads_the_rust_model_and_every_probability_is_its_bit_for_bit() {
    let all = goldens();
    // The inputs of `langdetect-183`'s goldens, run with the fixture.
    assert_eq!(all.len(), 22);
    let detector = detector();
    let maxent = detector.model().maxent();
    assert_eq!(maxent.num_outcomes(), 9);
    let mut scratch = Scratch::new();
    let mut ranking = Ranking::new();
    let mut compared = 0;
    for (set, id, path) in all {
        let golden = Golden::read(&path).unwrap();
        let points = read_dump(&path.with_extension("dump"));
        assert_eq!(points.len(), 1, "{set}/{id}");
        let point = &points[0];
        detector.predict_languages(&input(&golden), &mut scratch, &mut ranking);
        assert_eq!(point.outcomes.len(), 9, "{set}/{id}");
        for (o, (label, p)) in point.outcomes.iter().enumerate() {
            assert_eq!(maxent.outcome(o), label, "{set}/{id}");
            let rust = scratch.probabilities()[o];
            assert_eq!(rust.to_bits(), p.to_bits(), "{set}/{id}: {label} is {rust} in Rust and {p} on the JVM");
            compared += 1;
        }
    }
    assert_eq!(compared, 22 * 9);
}

/// The texts in which the fixture knows an n-gram match their goldens
/// exactly. In the others (scripts the fixture was not trained on) the JVM
/// ranks every language equal, in outcome order, and the annotator
/// declines them, as with `langdetect-183.bin` (docs/langdetect.md).
#[test]
fn every_ranking_with_the_rust_model_matches_its_golden_or_is_declined_without_evidence() {
    let detector = detector();
    let mut scratch = Scratch::new();
    let mut ranking = Ranking::new();
    let maxent = detector.model().maxent();
    let outcomes: Vec<String> = (0..maxent.num_outcomes()).map(|o| maxent.outcome(o).to_string()).collect();
    let mut evidence = Vec::new();
    for (set, id, path) in goldens() {
        detector.predict_languages(&input(&Golden::read(&path).unwrap()), &mut scratch, &mut ranking);
        evidence.push((format!("{set}/{id}"), ranking.has_evidence()));
    }
    let analyzer = Analyzer::builder().add(detector).build().unwrap();
    let mut declined = Vec::new();
    for ((_, _, path), (name, known)) in goldens().into_iter().zip(&evidence) {
        let golden = Golden::read(&path).unwrap();
        let mut doc = Document::new(input(&golden));
        let results = analyzer.run(slice::from_mut(&mut doc));
        assert!(results[0].is_ok(), "{name}: {:?}", results[0]);
        if *known {
            assert_matches_golden::<turbo_analysis_layers::Language>(&doc, &golden, Tolerance::exact());
            continue;
        }
        let status = doc.status(Key::<turbo_analysis_layers::Language>::DEFAULT.id());
        assert_eq!(status, Some(&Status::Skipped("no language evidence".into())), "{name}");
        let jvm = golden.layer("opennlp:language").unwrap();
        let (codes, p) = (jvm.strs("code").unwrap(), jvm.f64s("confidence").unwrap());
        assert_eq!(codes.iter().collect::<Vec<_>>(), outcomes, "{name}: outcome order");
        assert!(p.iter().all(|q| q.to_bits() == p[0].to_bits()), "{name}: {p:?}");
        declined.push(name.clone());
    }
    assert_eq!(declined, ["multi/ja", "multi/zh"]);
}

#[test]
fn probing_with_the_rust_model_matches_every_chunk_context_probability_ranking_and_length() {
    let detector = detector();
    let maxent = detector.model().maxent();
    let shapes = segmentation_shapes();
    let mut scratch = Scratch::new();
    let mut ranking = Ranking::new();
    let mut probing_ctx = ProbingContext::default();
    let mut chunk_probs = vec![0.0; maxent.num_outcomes()];
    let mut early_stopped = 0usize;
    let mut capped = 0usize;

    for (run, cfg) in
        [(format!("{FIXTURE}+probing"), ProbingConfig::default()), (format!("{FIXTURE}+probing-chunk50"), CHUNK_50)]
    {
        let all = goldens_in(&data_dir().join("langdetect").join(&run));
        assert_eq!(all.len(), 24, "{run}");
        for (set, id, path) in all {
            let golden = Golden::read_with(&path, &shapes).unwrap();
            let text = input(&golden);
            let char_count = text.chars().count();
            let points = read_dump(&path.with_extension("dump"));
            let expected_len = golden.layer("turbo:probing-length").unwrap().u32s("length").unwrap()[0] as usize;

            // Step chunk by chunk and verify every decision point's cumulative n-gram order and probabilities.
            probing_ctx.reset();
            let chars: Vec<char> = text.chars().collect();
            let mut pos = 0usize;
            for (step, point) in points.iter().enumerate() {
                let take = cfg.chunk_size.min(cfg.max_length.saturating_sub(pos)).min(chars.len().saturating_sub(pos));
                let chunk: String = chars[pos..pos + take].iter().collect();
                pos += take;
                probing_ctx.add_chunk(&chunk);
                assert!(probing_ctx.exact_order(), "{run}/{set}/{id} step {step}");
                let rust_ctx: Vec<&[u8]> = probing_ctx.features().collect();
                let jvm_ctx: Vec<&[u8]> = point.contexts.iter().map(Vec::as_slice).collect();
                assert_eq!(rust_ctx, jvm_ctx, "{run}/{set}/{id} step {step} contexts");
                maxent.eval_features(probing_ctx.features(), &mut chunk_probs);
                for (o, (label, p)) in point.outcomes.iter().enumerate() {
                    assert_eq!(maxent.outcome(o), label);
                    assert_eq!(chunk_probs[o].to_bits(), p.to_bits(), "{run}/{set}/{id} step {step} outcome {label}");
                }
            }
            assert_eq!(pos, expected_len, "{run}/{set}/{id} chunked pos");

            let length = detector.probing_predict_languages_with(&text, cfg, &mut scratch, &mut ranking);
            assert_eq!(length, expected_len, "{run}/{set}/{id} probing length");
            if length < char_count && length < cfg.max_length {
                early_stopped += 1;
            }
            if length == cfg.max_length && char_count > cfg.max_length {
                capped += 1;
            }

            let jvm = golden.layer("opennlp:language").unwrap();
            let codes: Vec<&str> = jvm.strs("code").unwrap().iter().collect();
            let confs = jvm.f64s("confidence").unwrap();
            let rust_codes: Vec<&str> = ranking.outcomes().iter().map(|&o| detector.model().code(o as usize)).collect();
            assert_eq!(rust_codes, codes, "{run}/{set}/{id} ranking order");
            for (r, j) in ranking.probabilities().iter().zip(confs) {
                assert_eq!(r.to_bits(), j.to_bits(), "{run}/{set}/{id} confidence");
            }
        }
    }
    assert!(early_stopped > 0, "expected early-stopped inputs via seenEnough");
    assert!(capped > 0, "expected maxLength-capped inputs");
}
