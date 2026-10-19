//! The parsers' event streams, evaluation and training against the JVM: the `parse-treebank`
//! goldens `ParserTrainGolden.java` writes, compared with what both test crates share in
//! `analysis/parser/tests/goldens/treebank_runs.rs`, whose documentation describes the runs and
//! the comparison.
//!
//! Here the eight event runs and the two model runs are compared in full: for a model run, the
//! model file, the parses of the model the JVM trained (which `turbo-parser`'s own tests compare
//! too) and `ParserEvaluator`'s scores. A model trained here on the JVM's corpus must be the model
//! the JVM wrote, and the comparison itself is checked on goldens written from this side's own
//! rows for all 13 runs. `turbo-parser` compares the dictionary and generator runs.

#[path = "../../parser/tests/goldens/treebank_runs.rs"]
mod treebank_runs;

use std::collections::BTreeSet;
use std::str::FromStr;
use std::sync::Arc;

use treebank_runs::*;
use turbo_goldens::Golden;
use turbo_parser::model::{CHUNKER_TAGGER_MODEL_ENTRY_NAME, PARSER_TAGGER_MODEL_ENTRY_NAME, PARSER_TYPE};
use turbo_parser::{
    EnglishHeadRules, HeadRulesArtifact, Parse, ParserEventType, ParserFactory, ParserType, build_dictionary,
};
use turbo_train_components::parser::{ParserEvaluator, ParserTrainingParameters, chunking, train_parser, treeinsert};

fn rebuild_events(
    lines: &[&str],
    rules: &EnglishHeadRules,
    stream: Stream,
    etype: ParserEventType,
    dictionary: bool,
) -> Rebuilt {
    let dict = dictionary.then(|| build_dictionary(&parsed(lines), rules, 1));
    let mut out = Rebuilt::default();
    out.layers(&[EVENTS, SAMPLE_ERRORS]);
    for (i, line) in lines.iter().enumerate() {
        let made = Parse::parse(line).map_err(|e| parse_error(&e)).and_then(|p| match stream {
            Stream::Chunking => {
                chunking::ParserEventStream::generate_events(std::slice::from_ref(&p), rules, etype, dict.as_ref())
                    .map_err(|e| parse_error(&e))
            }
            Stream::TreeInsert => {
                treeinsert::ParserEventStream::generate_events(std::slice::from_ref(&p), rules, etype, None)
                    .map_err(|e| tree_insert_error(&e, &p))
            }
        });
        match made {
            Ok(events) => {
                for e in events {
                    let context: Vec<String> = e.context().iter().map(|c| c.to_string_lossy()).collect();
                    let outcome = e.outcome().to_string_lossy();
                    out.row(EVENTS, vec![n(i), s(outcome), n(context.len()), s(context.join(" "))]);
                }
            }
            Err((class, message)) => out.row(SAMPLE_ERRORS, vec![n(i), s(class), message]),
        }
    }
    out
}

/// The evaluation of a model run: `ParserEvaluator`'s scores for the model in `bytes` over the
/// samples, in each of [`EVALUATIONS`].
fn rebuild_evaluation(out: &mut Rebuilt, lines: &[&str], bytes: &[u8]) {
    let model = Arc::new(turbo_parser::ParserModel::from_bytes(bytes.to_vec()).unwrap_or_else(|e| panic!("{e}")));
    let references = references(lines);
    out.layers(&[EVALUATION]);
    for (beam, advance) in EVALUATIONS {
        let parser = ParserFactory::create_with_params(Arc::clone(&model), beam as usize, advance);
        let mut evaluator = ParserEvaluator::new(&parser);
        let row = match evaluator.evaluate(&references) {
            Ok(()) => {
                let fm = evaluator.fmeasure();
                vec![n(beam), f(advance), f(fm.precision()), f(fm.recall()), f(fm.f_measure())]
            }
            // The golden has scores, so this row differs and says why.
            Err(e) => vec![n(beam), f(advance), s(format!("{}: {e}", parse_error(&e).0)), None, None],
        };
        out.row(EVALUATION, row);
    }
}

/// Everything `run` computes from `input`, the input with this id; `model` is the trained model's
/// bytes for a model run.
fn rebuild(run: &Run, id: &str, input: &str, rules: &EnglishHeadRules, model: Option<&[u8]>) -> Rebuilt {
    let lines = sample_lines(input);
    match run.kind {
        Kind::Dictionary(cutoff) => rebuild_dictionary(&lines, rules, cutoff),
        Kind::Events(stream, etype, dictionary) => rebuild_events(&lines, rules, stream, etype, dictionary),
        Kind::Generators => rebuild_generators(&lines, rules),
        Kind::Model(_) => {
            let bytes = model.expect("a model run's model");
            let mut out = rebuild_parses(&lines, bytes, id == INPUTS[0]);
            rebuild_evaluation(&mut out, &lines, bytes);
            out
        }
    }
}

#[test]
fn the_event_goldens_match() {
    check_runs(|k| matches!(k, Kind::Events(..)), &rebuild, &[]);
}

#[test]
fn the_chunking_model_goldens_match() {
    check_runs(|k| matches!(k, Kind::Model(ParserType::Chunking)), &rebuild, &[]);
}

#[test]
fn the_tree_insertion_model_goldens_match() {
    check_runs(|k| matches!(k, Kind::Model(ParserType::TreeInsert)), &rebuild, &[]);
}

/// The parameters of the model runs: every model MAXENT with 100 iterations and cutoff 1, and the
/// dictionary at cutoff 1. The run gives the tagger no beam size.
fn model_run_parameters() -> ParserTrainingParameters {
    let mut params = ParserTrainingParameters::with_iterations_and_cutoff(100, 1);
    params.tagger = params.chunker.clone();
    params
}

/// A model trained here on the corpus the JVM trained on, with the same parameters, is the model
/// the JVM wrote: the same manifests (the parser type under `parser-type`, each maxent model's
/// training report under its name, the serializer classes, and the tagger's and chunker's own
/// entries) and the same bytes in every other entry. The JVM's manifests have no `Timestamp`,
/// which this side writes, so the manifest files and the nested packages holding them are
/// compared by their entries only.
#[test]
fn a_model_trained_here_is_the_model_the_jvm_wrote() {
    let Some(root) = checked_in() else { return };
    let rules = EnglishHeadRules::from_str(&rules_text()).unwrap();
    let shapes = shapes();
    let nested = [PARSER_TAGGER_MODEL_ENTRY_NAME, CHUNKER_TAGGER_MODEL_ENTRY_NAME];
    let keep = |id: &str, row: &[String]| match id {
        MANIFEST => row[1] != "Timestamp",
        _ => !row[0].ends_with("manifest.properties") && !nested.contains(&row[0].as_str()),
    };
    for run in RUNS.iter().filter(|r| matches!(r.kind, Kind::Model(_))) {
        let path = root.join(run.name).join(SET).join(format!("{}.golden", INPUTS[0]));
        let golden = Golden::read_with(&path, &shapes).unwrap_or_else(|e| panic!("{e}"));
        let input = golden.input_text(turbo_goldens::data_dir()).unwrap_or_else(|e| panic!("{e}"));
        let samples = parsed(&sample_lines(&input));
        let artifact = HeadRulesArtifact::English(rules.clone());
        let params = model_run_parameters();
        let Kind::Model(parser_type) = run.kind else { unreachable!("a model run") };
        let model = train_parser("en", &samples, artifact, &params, parser_type)
            .unwrap_or_else(|e| panic!("{}: {e}", run.name));
        let mut rebuilt = Rebuilt::default();
        model_layers(&mut rebuilt, model.as_bytes());
        for id in [MANIFEST, MODEL_ENTRIES] {
            let theirs: Vec<Vec<String>> =
                cells(golden.layer(id).unwrap()).into_iter().filter(|row| keep(id, row)).collect();
            let ours: Vec<Vec<String>> = rebuilt.0[id]
                .iter()
                .map(|row| row.cells.iter().map(|c| c.clone().unwrap()).collect::<Vec<String>>())
                .filter(|row| keep(id, row))
                .collect();
            assert_eq!(ours, theirs, "{}: {id}", run.name);
        }
        assert_eq!(model.file().manifest().get(PARSER_TYPE), Some(model.parser_type().as_str()), "{}", run.name);
    }
}

/// A project-written corpus for the self-test, with brackets so that the unlabeled chunking build
/// context reads a missing label.
const SELF_TEST_TREEBANK: &str = "\
(TOP (S (NP (DT The) (JJ quick) (NN fox)) (VP (VBD jumped) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))) (. .)))

(TOP (S (NP (DT The) (NN dog)) (VP (VBD slept) (PP (IN under) (NP (DT the) (JJ old) (NN tree)))) (. .)))
(TOP (S (NP (NNP Alice)) (VP (VBD wrote) (NP (DT a) (NN program)) (PP (IN in) (NP (NNP Rust)))) (. .)))
(TOP (S (NP (PRP She)) (VP (VBD saw) (NP (DT the) (NN ship)) (, ,) (CC and) (VP (VBD waved))) (. .)))
(TOP (S (PP (IN On) (NP (NNP Monday))) (, ,) (NP (DT the) (NN team)) (VP (VBD shipped) (NP (DT the) (NN release))) (. .)))
(TOP (S (NP (NNP Bob)) (-LRB- -LRB-) (NP (NNP Rust)) (-RRB- -RRB-) (VP (VBD won)) (. .)))
(TOP (S (NP (NP (DT The) (NN man)) (PP (IN with) (NP (DT a) (NN hat)))) (VP (VBD left) (NP (DT the) (NN room))) (. .)))
(TOP (S (NP (NNP Eve)) (VP (VP (VBD read) (NP (DT a) (NN book))) (CC and) (VP (VBD slept))) (. .)))
";

const SELF_TEST_HELD_OUT: &str = "\
(TOP (S (NP (DT The) (NN cat)) (VP (VBD sat) (PP (IN on) (NP (DT the) (NN mat)))) (. .)))
(TOP (S (NP (NNP Eve)) (VP (VBD read) (NP (DT a) (NN book))) (. .)))
";

/// The comparison itself, on goldens written from this side's own rows for every run: each reads
/// back with the shapes here and compares clean, and changing one cell makes it differ. The model
/// runs read models trained here on a small corpus.
#[test]
fn the_comparison_reads_back_what_it_rebuilds_and_finds_a_changed_cell() {
    let rules_text = rules_text();
    let rules = EnglishHeadRules::from_str(&rules_text).unwrap();
    let shapes = shapes();
    let samples = parsed(&sample_lines(SELF_TEST_TREEBANK));
    let params = ParserTrainingParameters::with_iterations_and_cutoff(10, 1);
    let artifact = || HeadRulesArtifact::English(rules.clone());
    let chunking_model = chunking::train("en", &samples, artifact(), &params).unwrap();
    let treeinsert_model = treeinsert::train("en", &samples, artifact(), &params).unwrap();
    let mut layers_seen = BTreeSet::new();
    let mut changes = 0;
    for run in &RUNS {
        let model = match run.kind {
            Kind::Model(ParserType::Chunking) => Some(chunking_model.as_bytes()),
            Kind::Model(ParserType::TreeInsert) => Some(treeinsert_model.as_bytes()),
            _ => None,
        };
        for (id, input) in INPUTS.into_iter().zip([SELF_TEST_TREEBANK, SELF_TEST_HELD_OUT]) {
            let what = format!("{}/{id}", run.name);
            let rebuilt = rebuild(run, id, input, &rules, model);
            let written = to_golden(header(run, id, input, &rules_text), &rebuilt, &shapes);
            let text = written.to_text().unwrap();
            let golden = Golden::parse_with(&text, &shapes).unwrap_or_else(|e| panic!("{what}: {e}"));
            check_header(&what, &golden, run, id, &rules_text);
            if let Some(bytes) = model.filter(|_| id == INPUTS[0]) {
                assert_eq!(model_bytes(&golden), bytes, "{what}");
            }
            assert_eq!(compare(&golden, &rebuilt), Vec::<String>::new(), "{what}");
            if let Some(changed) = with_one_cell_changed(&golden) {
                assert!(!compare(&changed, &rebuilt).is_empty(), "{what}: a changed cell");
                changes += 1;
            }
            layers_seen.extend(golden.layers.iter().filter(|l| !l.is_empty()).map(|l| l.id.clone()));
        }
    }
    // Every layer has rows in some run but the sample and parse errors, which this corpus does
    // not cause.
    let expected: BTreeSet<String> =
        LAYERS.iter().map(|l| l.0).filter(|id| ![SAMPLE_ERRORS, PARSE_ERRORS].contains(id)).map(String::from).collect();
    assert_eq!(layers_seen, expected);
    // Only the held-out input of `dictionary-cutoff-5` has no layer with rows to change.
    assert_eq!(changes, 2 * RUNS.len() - 1);
}
