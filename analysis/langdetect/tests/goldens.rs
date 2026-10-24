//! Parity with OpenNLP on the JVM, from the goldens and decision point
//! dumps under `analysis/goldens/data/langdetect/langdetect-183`.
//!
//! The context (every n-gram, its bytes and its position in the order the
//! model receives them) is compared with each dump without the model. The
//! probabilities and the ranking need `langdetect-183.bin`, which is not
//! checked in; see `common::model_path`.

mod common;

use std::slice;

use turbo_analysis::{Analyzer, Document, Key, Status};
use turbo_goldens::{Golden, Tolerance, assert_matches_golden, data_dir, segmentation_shapes};
use turbo_langdetect::{
    ContextGenerator, LanguageDetector, LanguageModel, ProbingConfig, ProbingContext, Ranking, Scratch,
};
use turbo_maxent::ModelFile;

use common::{goldens, goldens_in, model_path, read_dump};

const CHUNK_50: ProbingConfig =
    ProbingConfig { max_length: 250, chunk_size: 50, min_consec_improvements: 2, min_diff: 0.20 };

fn input(golden: &Golden) -> String {
    golden.input_text(data_dir()).unwrap()
}

#[test]
fn there_is_a_golden_and_a_dump_for_every_input() {
    let all = goldens();
    assert_eq!(all.len(), 22);
    for (_, _, path) in &all {
        assert!(path.with_extension("dump").is_file(), "{}", path.display());
    }
}

#[test]
fn every_context_is_the_jvm_context_in_the_jvm_order() {
    let mut generator = ContextGenerator::default();
    let mut total = 0;
    for (set, id, path) in goldens() {
        let golden = Golden::read(&path).unwrap();
        let points = read_dump(&path.with_extension("dump"));
        assert_eq!(points.len(), 1, "{set}/{id}: one decision per document");
        let point = &points[0];
        assert!(point.values.iter().all(|&v| v == 1.0), "{set}/{id}: every value is 1");

        generator.generate(&input(&golden));
        let rust: Vec<&[u8]> = generator.features().collect();
        let jvm: Vec<&[u8]> = point.contexts.iter().map(Vec::as_slice).collect();
        assert_eq!(rust.len(), jvm.len(), "{set}/{id}: the number of n-grams");
        if let Some(i) = (0..rust.len()).find(|&i| rust[i] != jvm[i]) {
            panic!(
                "{set}/{id}: n-gram {i} is {:?} in Rust and {:?} on the JVM",
                String::from_utf8_lossy(rust[i]),
                String::from_utf8_lossy(jvm[i])
            );
        }
        assert!(generator.exact_order(), "{set}/{id}");
        total += rust.len();
    }
    // Every n-gram of all 22 inputs, compared in place.
    assert!(total > 5000, "{total}");
}

fn detector() -> Option<LanguageDetector> {
    let path = model_path()?;
    Some(LanguageDetector::new(LanguageModel::open(ModelFile::read(path).unwrap()).unwrap()))
}

#[test]
fn every_probability_is_the_jvm_probability_bit_for_bit() {
    let Some(detector) = detector() else { return };
    let mut scratch = Scratch::new();
    let mut ranking = Ranking::new();
    let mut max_error = 0.0f64;
    let mut compared = 0;
    for (set, id, path) in goldens() {
        let golden = Golden::read(&path).unwrap();
        let point = &read_dump(&path.with_extension("dump"))[0];
        detector.predict_languages(&input(&golden), &mut scratch, &mut ranking);
        let maxent = detector.model().maxent();
        assert_eq!(point.outcomes.len(), maxent.num_outcomes());
        for (o, (label, p)) in point.outcomes.iter().enumerate() {
            assert_eq!(maxent.outcome(o), label, "{set}/{id}");
            let rust = scratch.probabilities()[o];
            max_error = max_error.max((rust - p).abs());
            assert_eq!(rust.to_bits(), p.to_bits(), "{set}/{id}: {label} is {rust} in Rust and {p} on the JVM");
            compared += 1;
        }
    }
    assert_eq!(compared, 22 * 103);
    assert_eq!(max_error, 0.0);
}

#[test]
fn every_ranking_matches_its_golden_exactly() {
    let Some(detector) = detector() else { return };
    let analyzer = Analyzer::builder().add(detector).build().unwrap();
    for (_, _, path) in goldens() {
        let golden = Golden::read(&path).unwrap();
        let mut doc = Document::new(input(&golden));
        let results = analyzer.run(slice::from_mut(&mut doc));
        assert!(results[0].is_ok(), "{}: {:?}", path.display(), results[0]);
        assert_matches_golden::<turbo_analysis_layers::Language>(&doc, &golden, Tolerance::exact());
    }
}

#[test]
fn predict_language_is_the_first_of_the_ranking() {
    let Some(detector) = detector() else { return };
    let mut scratch = Scratch::new();
    for (set, id, path) in goldens() {
        let golden = Golden::read(&path).unwrap();
        let best = golden.layer("opennlp:language").unwrap();
        let code = best.strs("code").unwrap().get(0);
        let p = best.f64s("confidence").unwrap()[0];
        let prediction = detector.predict_language(&input(&golden), &mut scratch);
        assert_eq!((prediction.code, prediction.probability.to_bits()), (code, p.to_bits()), "{set}/{id}");
        assert!(prediction.evidence);
    }
}

#[test]
fn the_order_of_the_n_grams_changes_the_last_bits() {
    // Summed in another order (here reversed), the same n-grams give
    // probabilities that differ from the JVM's in the last bits, which is
    // why the context keeps the JVM's hash map order.
    let Some(detector) = detector() else { return };
    let maxent = detector.model().maxent();
    let mut generator = ContextGenerator::default();
    let mut probabilities = vec![0.0; maxent.num_outcomes()];
    let (mut differ, mut max_error) = (0, 0.0f64);
    for (_, _, path) in goldens() {
        let golden = Golden::read(&path).unwrap();
        let point = &read_dump(&path.with_extension("dump"))[0];
        generator.generate(&input(&golden));
        let mut reversed: Vec<&[u8]> = generator.features().collect();
        reversed.reverse();
        maxent.eval_modified_utf8(&reversed, &mut probabilities);
        for (p, (_, jvm)) in probabilities.iter().zip(&point.outcomes) {
            differ += usize::from(p.to_bits() != jvm.to_bits());
            max_error = max_error.max((p - jvm).abs());
        }
    }
    assert!(differ > 0);
    assert!(max_error < 1e-12, "{max_error}");
}

#[test]
fn without_evidence_the_jvm_ranks_every_language_equal_and_the_annotator_declines() {
    let Some(detector) = detector() else { return };
    let mut scratch = Scratch::new();
    let mut ranking = Ranking::new();
    // OpenNLP's answer for an empty text: all 103 languages at the prior,
    // in outcome order.
    detector.predict_languages("", &mut scratch, &mut ranking);
    assert!(!ranking.has_evidence());
    assert_eq!(ranking.outcomes(), (0..103).collect::<Vec<u32>>());
    assert!(ranking.probabilities().iter().all(|p| (p - 1.0 / 103.0).abs() < 1e-15));
    // The annotator declines it, and any text the normalizers empty.
    let analyzer = Analyzer::builder().add(detector).build().unwrap();
    let mut docs: Vec<Document> =
        ["", " ", "2024 12 31", "https://example.com", "\u{1F600}"].into_iter().map(Document::new).collect();
    analyzer.run(&mut docs);
    for doc in &docs {
        let status = doc.status(Key::<turbo_analysis_layers::Language>::DEFAULT.id());
        assert_eq!(status, Some(&Status::Skipped("no language evidence".into())), "{:?}", doc.source());
    }
}

#[test]
fn every_probing_context_is_the_jvm_chunk_context_in_the_jvm_order() {
    let shapes = segmentation_shapes();
    let mut probing_ctx = ProbingContext::default();
    let mut total_points = 0usize;
    let mut total_ngrams = 0usize;
    for (run, cfg) in
        [("langdetect-183+probing", ProbingConfig::default()), ("langdetect-183+probing-chunk50", CHUNK_50)]
    {
        let all = goldens_in(&data_dir().join("langdetect").join(run));
        assert_eq!(all.len(), 24, "{run}");
        for (set, id, path) in all {
            let golden = Golden::read_with(&path, &shapes).unwrap();
            let text = input(&golden);
            let points = read_dump(&path.with_extension("dump"));
            let expected_len = golden.layer("turbo:probing-length").unwrap().u32s("length").unwrap()[0] as usize;
            let chars: Vec<char> = text.chars().collect();
            probing_ctx.reset();
            let mut pos = 0usize;
            for (step, point) in points.iter().enumerate() {
                assert!(point.values.iter().all(|&v| v == 1.0), "{run}/{set}/{id} step {step}");
                let take = cfg.chunk_size.min(cfg.max_length.saturating_sub(pos)).min(chars.len().saturating_sub(pos));
                let chunk: String = chars[pos..pos + take].iter().collect();
                pos += take;
                probing_ctx.add_chunk(&chunk);
                assert!(probing_ctx.exact_order(), "{run}/{set}/{id} step {step}");
                let rust: Vec<&[u8]> = probing_ctx.features().collect();
                let jvm: Vec<&[u8]> = point.contexts.iter().map(Vec::as_slice).collect();
                assert_eq!(rust, jvm, "{run}/{set}/{id} step {step}");
                total_points += 1;
                total_ngrams += rust.len();
            }
            assert_eq!(pos, expected_len, "{run}/{set}/{id}");
        }
    }
    assert!(total_points > 100, "{total_points}");
    assert!(total_ngrams > 25_000, "{total_ngrams}");
}

#[test]
fn every_probing_probability_ranking_and_length_is_the_jvms_bit_for_bit() {
    let Some(detector) = detector() else { return };
    let shapes = segmentation_shapes();
    let maxent = detector.model().maxent();
    let mut scratch = Scratch::new();
    let mut ranking = Ranking::new();
    let mut compared = 0usize;
    for (run, cfg) in
        [("langdetect-183+probing", ProbingConfig::default()), ("langdetect-183+probing-chunk50", CHUNK_50)]
    {
        for (set, id, path) in goldens_in(&data_dir().join("langdetect").join(run)) {
            let golden = Golden::read_with(&path, &shapes).unwrap();
            let points = read_dump(&path.with_extension("dump"));
            let last = points.last().unwrap();
            let expected_len = golden.layer("turbo:probing-length").unwrap().u32s("length").unwrap()[0] as usize;
            let length = detector.probing_predict_languages_with(&input(&golden), cfg, &mut scratch, &mut ranking);
            assert_eq!(length, expected_len, "{run}/{set}/{id} length");
            for (o, (label, p)) in last.outcomes.iter().enumerate() {
                assert_eq!(maxent.outcome(o), label, "{run}/{set}/{id}");
                let rust = scratch.probabilities()[o];
                assert_eq!(
                    rust.to_bits(),
                    p.to_bits(),
                    "{run}/{set}/{id}: {label} is {rust} in Rust and {p} on the JVM"
                );
                compared += 1;
            }
            let jvm = golden.layer("opennlp:language").unwrap();
            let codes: Vec<&str> = jvm.strs("code").unwrap().iter().collect();
            let confs = jvm.f64s("confidence").unwrap();
            let rust_codes: Vec<&str> = ranking.outcomes().iter().map(|&o| detector.model().code(o as usize)).collect();
            assert_eq!(rust_codes, codes, "{run}/{set}/{id} ranking");
            for (r, j) in ranking.probabilities().iter().zip(confs) {
                assert_eq!(r.to_bits(), j.to_bits(), "{run}/{set}/{id} confidence");
            }
        }
    }
    assert_eq!(compared, 48 * 103);
}
