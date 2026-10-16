//! The parsers' goldens this crate rebuilds alone (see `treebank_runs.rs`): the dictionary and
//! generator runs, and the model runs without their evaluation layer, so that decoding a model
//! the JVM trained is checked here. The event runs, the evaluation and training a model are
//! compared in `turbo-train-components`' `tests/parser_goldens.rs`, which includes the same
//! `treebank_runs.rs`.

use std::collections::BTreeSet;
use std::sync::Arc;

use turbo_parser::{COMPLETE, EnglishHeadRules, Parse, ParserType};

use super::treebank_runs::*;

/// What this crate rebuilds for `run`; the event runs are rebuilt in `turbo-train-components`.
fn rebuild(run: &Run, id: &str, input: &str, rules: &EnglishHeadRules, model: Option<&[u8]>) -> Rebuilt {
    let lines = sample_lines(input);
    match run.kind {
        Kind::Dictionary(cutoff) => rebuild_dictionary(&lines, rules, cutoff),
        Kind::Generators => rebuild_generators(&lines, rules),
        Kind::Model(_) => rebuild_parses(&lines, model.expect("a model run's model"), id == INPUTS[0]),
        Kind::Events(..) => unreachable!("{}: the event runs are compared in turbo-train-components", run.name),
    }
}

#[test]
fn the_parse_treebank_goldens_are_the_26_expected_files() {
    let Some(root) = checked_in() else { return };
    let expected: BTreeSet<String> =
        RUNS.iter().flat_map(|r| INPUTS.map(|id| format!("{}/{SET}/{id}.golden", r.name))).collect();
    assert_eq!(expected.len(), 26);
    let mut found = BTreeSet::new();
    for run in std::fs::read_dir(&root).unwrap() {
        let run = run.unwrap().path();
        let dir = run.join(SET);
        if !dir.exists() {
            continue;
        }
        let mut stack = vec![dir];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    found.insert(path.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }
    let missing: Vec<&String> = expected.difference(&found).collect();
    let extra: Vec<&String> = found.difference(&expected).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "under {}: missing {missing:?}, not expected {extra:?}",
        root.display()
    );
    for id in INPUTS {
        let input = turbo_goldens::data_dir().join("inputs").join(SET).join(format!("{id}.txt"));
        assert!(input.is_file(), "{} is missing", input.display());
    }
}

#[test]
fn the_dictionary_goldens_match() {
    check_runs(|k| matches!(k, Kind::Dictionary(_)), &rebuild, &[]);
}

#[test]
fn the_generator_goldens_match() {
    check_runs(|k| matches!(k, Kind::Generators), &rebuild, &[]);
}

/// The model the JVM trained, read from the golden, parses each sample as `ParserTool` does on the
/// JVM. The evaluation layer is compared in `turbo-train-components`.
#[test]
fn the_chunking_model_parses_match() {
    check_runs(|k| matches!(k, Kind::Model(ParserType::Chunking)), &rebuild, &[EVALUATION]);
}

#[test]
fn the_tree_insertion_model_parses_match() {
    check_runs(|k| matches!(k, Kind::Model(ParserType::TreeInsert)), &rebuild, &[EVALUATION]);
}

/// A model golden's entries may leave out the nested packages themselves, but nothing else.
#[test]
fn only_a_nested_package_row_may_be_missing() {
    let mut rebuilt = Rebuilt::default();
    rebuilt.row(MODEL_ENTRIES, vec![s("build.model"), n(3), s("ab")]);
    rebuilt.optional_row(MODEL_ENTRIES, vec![s("parsertager.postagger"), n(5), s("cd")]);
    rebuilt.row(MODEL_ENTRIES, vec![s("parsertager.postagger/pos.model"), n(2), s("ef")]);
    let shapes = shapes();
    let golden = |rows: &[(&str, u32, &str)]| {
        let mut listed = Rebuilt::default();
        listed.layers(&[MODEL_ENTRIES]);
        for &(path, length, sha) in rows {
            listed.row(MODEL_ENTRIES, vec![s(path), n(length), s(sha)]);
        }
        let run = RUNS.iter().find(|r| r.name == "model-chunking").unwrap();
        to_golden(header(run, "treebank", "", ""), &listed, &shapes)
    };
    let both = [("build.model", 3, "ab"), ("parsertager.postagger/pos.model", 2, "ef")];
    assert_eq!(compare(&golden(&both), &rebuilt), Vec::<String>::new());
    let all =
        [("build.model", 3, "ab"), ("parsertager.postagger", 5, "cd"), ("parsertager.postagger/pos.model", 2, "ef")];
    assert_eq!(compare(&golden(&all), &rebuilt), Vec::<String>::new());
    assert!(!compare(&golden(&both[..1]), &rebuilt).is_empty(), "a nested entry missing");
    let wrong =
        [("build.model", 3, "ab"), ("parsertager.postagger", 6, "cd"), ("parsertager.postagger/pos.model", 2, "ef")];
    assert!(!compare(&golden(&wrong), &rebuilt).is_empty(), "a nested package listed with the wrong length");
}

#[test]
fn ngrams_are_in_string_compare_order() {
    let key = |tokens: &[&str]| tokens.iter().map(|t| format!("{}:{t}", t.len())).collect::<String>();
    assert_eq!(ngram(&key(&["1:2", "a"])), ["1:2", "a"]);
    let mut ngrams: Vec<Vec<String>> = [&["b"][..], &["a", "b"], &["a"], &["\u{ff61}"], &["\u{1f600}"], &["B"]]
        .iter()
        .map(|t| t.iter().map(|s| s.to_string()).collect())
        .collect();
    ngrams.sort_by(|a, b| java_order(a, b));
    // UTF-16 order puts a surrogate pair before U+FF61, where UTF-8 order puts it after.
    let order: Vec<String> = ngrams.iter().map(|n| n.join(" ")).collect();
    assert_eq!(order, ["B", "a", "a b", "b", "\u{1f600}", "\u{ff61}"]);
}

#[test]
fn a_missing_label_is_read_where_java_reads_it() {
    let chunks = |kinds: &[&str]| -> Vec<Parse> {
        let text: String = kinds.iter().map(|_| "x ").collect();
        let text: Arc<str> = text.into();
        kinds
            .iter()
            .enumerate()
            .map(|(i, k)| {
                Parse::new(
                    Arc::clone(&text),
                    turbo_analysis::Span::new(2 * i as u32, 2 * i as u32 + 1),
                    *k,
                    1.0,
                    i as i32,
                )
            })
            .collect()
    };
    let cs = chunks(&["NP", "-LRB-", "NP", "-RRB-", "VP", "."]);
    assert!(!reads_missing_label(&cs, 0));
    assert!(reads_missing_label(&cs, 3), "the NP before the bracket has no label");
    assert!(reads_missing_label(&cs, 5), "a last `.` reads the label before it");
    let cs = chunks(&["NP", "-LRB-", "-RRB-"]);
    assert!(!reads_missing_label(&cs, 2), "the opening bracket comes first");
    let mut cs = chunks(&["NP", "NP", "-RRB-"]);
    cs[1].set_label(Some("S-NP".into()));
    assert!(!reads_missing_label(&cs, 2), "a start stops the walk");
    cs[1].set_label(Some(COMPLETE.into()));
    assert!(reads_missing_label(&cs, 2), "the walk goes on past a complete chunk");
}
