use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};
use turbo_formats::conllu::words::ConlluWordReader;
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target").join(format!(
            "dependency-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_owned()
    }
    fn put(&self, name: &str, bytes: &[u8]) -> String {
        let path = self.path(name);
        fs::write(&path, bytes).unwrap();
        path
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-depparse")).args(args).output().unwrap()
}
fn success(args: &[&str]) -> String {
    let output = run(args);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).unwrap()
}
fn corpus() -> String {
    format!(
        "{}\n\n{}\n\n",
        include_str!("data/ud-en.conllu").trim_end_matches('\n'),
        include_str!("data/ud-es.conllu").trim_end_matches('\n')
    )
}
fn table(text: &str) -> String {
    let mut out = String::new();
    for (s, sentence) in ConlluWordReader::new(text).unwrap().enumerate() {
        for (i, word) in sentence.unwrap().words.iter().enumerate() {
            let head = word.head.as_str().parse::<i64>().unwrap() - 1;
            out.push_str(&format!("{s}\t{i}\t{head}\t{}\n", word.deprel.as_str()));
        }
    }
    out
}
fn unparsed(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            let mut fields: Vec<_> = line.split('\t').collect();
            if fields.len() == 10 && fields[0].bytes().all(|b| b.is_ascii_digit()) {
                fields[6] = "_";
                fields[7] = "_";
            }
            fields.join("\t") + "\n"
        })
        .collect()
}

#[test]
fn neural_conversion_preserves_apache_default_beam_and_real_predictions() {
    let dir = Scratch::new();
    let model = dir.put("apache.bin", include_bytes!("apache/apache-refined.bin"));
    let converted = dir.path("native.bin");
    let receipt = success(&["convert", "feedforward", &model, &converted]);
    assert!(receipt.contains("source_sha256="));
    assert!(receipt.contains("output_sha256="));
    let inspect = success(&["inspect", "feedforward", &converted]);
    assert!(inspect.contains("default_beam=4"));
    let input = dir.put("input.conllu", unparsed(&corpus()).as_bytes());
    let output = dir.path("output.conllu");
    success(&["parse", "feedforward", &converted, &input, &output]);
    let actual = fs::read_to_string(output).unwrap();
    let expected: String = include_str!("apache/apache-refined.tsv")
        .lines()
        .filter_map(|row| row.strip_prefix("4\t"))
        .map(|row| format!("{row}\n"))
        .collect();
    assert_eq!(table(&actual), expected);
    let original_output = dir.path("original.conllu");
    success(&["parse", "feedforward", &model, &input, &original_output]);
    assert_eq!(actual, fs::read_to_string(original_output).unwrap());
    for (original, predicted) in corpus().lines().zip(actual.lines()) {
        let a: Vec<_> = original.split('\t').collect();
        let b: Vec<_> = predicted.split('\t').collect();
        if a.len() == 10 && a[0].bytes().all(|c| c.is_ascii_digit()) {
            for col in [0, 1, 2, 3, 4, 5, 8, 9] {
                assert_eq!(a[col], b[col]);
            }
        } else {
            assert_eq!(original, predicted);
        }
    }
}

#[test]
fn linear_apache_zip_and_raw_classifier_convert_without_prediction_change() {
    let dir = Scratch::new();
    let packaged = include_bytes!("apache/linear.bin");
    let zip = turbo_maxent::zip::Zip::new(packaged.as_slice()).unwrap();
    let raw = zip.entry("depparse.model").unwrap();
    let input = dir.put("input.conllu", unparsed(&corpus()).as_bytes());
    for (name, bytes) in [("zip", packaged.as_slice()), ("raw", raw.as_ref())] {
        let model = dir.put(name, bytes);
        let converted = dir.path(&format!("{name}.bin"));
        success(&["convert", "linear", &model, &converted]);
        let output = dir.path(&format!("{name}.conllu"));
        success(&["parse", "linear", &converted, &input, &output]);
        assert_eq!(table(&fs::read_to_string(output).unwrap()), include_str!("apache/linear.tsv"));
    }
}

#[test]
fn cli_loads_legacy_and_original_native_neural_models() {
    let dir = Scratch::new();
    for (name, bytes) in [
        ("legacy", include_bytes!("feedforward/data/plain.bin").as_slice()),
        ("v1", include_bytes!("apache/native-v1.bin").as_slice()),
    ] {
        let input = dir.put(name, bytes);
        let output = dir.path(&format!("{name}.bin"));
        assert!(success(&["inspect", "feedforward", &input]).contains("default_beam=1"));
        success(&["convert", "feedforward", &input, &output]);
        let loaded = turbo_depparse::FeedforwardDependencyModel::from_bytes(&fs::read(output).unwrap()).unwrap();
        assert_eq!(loaded.to_java_bytes().unwrap(), include_bytes!("feedforward/data/plain.bin"));
    }
}

#[test]
fn cli_errors_do_not_overwrite_or_leave_output_for_invalid_inputs() {
    let dir = Scratch::new();
    let valid = dir.put("valid", include_bytes!("apache/apache.bin"));
    let existing = dir.put("existing", b"keep these bytes");
    assert!(!run(&["convert", "feedforward", &valid, &existing]).status.success());
    assert_eq!(fs::read(existing).unwrap(), b"keep these bytes");
    let bad = dir.put("bad", b"FFDP");
    let absent = dir.path("absent");
    assert!(!run(&["convert", "feedforward", &bad, &absent]).status.success());
    assert!(!std::path::Path::new(&absent).exists());
    let no_tags = corpus()
        .lines()
        .map(|line| {
            let mut f: Vec<_> = line.split('\t').collect();
            if f.len() == 10 {
                f[3] = "_";
            }
            f.join("\t") + "\n"
        })
        .collect::<String>();
    let input = dir.put("notags", no_tags.as_bytes());
    assert!(!run(&["parse", "feedforward", &valid, &input, &absent]).status.success());
    assert!(!std::path::Path::new(&absent).exists());
    let joined = dir.put("joined", corpus().replace("\n\n", "\n").as_bytes());
    assert!(!run(&["parse", "feedforward", &valid, &joined, &absent]).status.success());
    assert!(!std::path::Path::new(&absent).exists());
}

#[test]
fn cli_trains_real_trees_with_explicit_exclusions_and_reloads_both_models() {
    let dir = Scratch::new();
    let input = dir
        .put("training.conllu", format!("{}{}", corpus(), include_str!("data/ud-en-nonprojective.conllu")).as_bytes());
    let inference = dir.put("input.conllu", unparsed(&corpus()).as_bytes());
    for (kind, settings) in [
        ("linear", "Algorithm=MAXENT\nIterations=2\nCutoff=1\nThreads=1\n"),
        ("feedforward", "Epochs=1\nEmbeddingSize=3\nHiddenSize=5\nBatchSize=4\nDropout=0\nWordCutoff=1\n"),
    ] {
        let params = dir.put(&format!("{kind}.params"), settings.as_bytes());
        let model = dir.path(&format!("{kind}.bin"));
        let args = ["train", kind, "und", &input, &params, &model, "--projective-only"];
        let result = run(&args);
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
        let counts = String::from_utf8(result.stderr).unwrap();
        assert!(
            counts.contains("sentences=5 words=81 excluded_nonprojective_sentences=1 excluded_nonprojective_words=16")
        );
        assert!(String::from_utf8(result.stdout).unwrap().contains("params_sha256="));
        let original = fs::read(&model).unwrap();
        assert!(!run(&args).status.success());
        assert_eq!(fs::read(&model).unwrap(), original);
        let output = dir.path(&format!("{kind}.conllu"));
        success(&["inspect", kind, &model]);
        success(&["parse", kind, &model, &inference, &output]);
        assert_eq!(table(&fs::read_to_string(output).unwrap()).lines().count(), 65);
    }
}

#[test]
fn training_rejects_implicit_filtering_unknown_settings_and_empty_projective_population() {
    let dir = Scratch::new();
    let corpus = dir.put("input", include_bytes!("data/ud-en-nonprojective.conllu"));
    let output = dir.path("model");
    for settings in ["Epochs=1\n", "Epochs=1\nUnknown=2\n", "HiddenSize=2\n", "Epochs=0\n", "Epochs=1\nDropout=NaN\n"] {
        let params = dir.put("params", settings.as_bytes());
        let args = ["train", "feedforward", "en", &corpus, &params, &output, "--projective-only"];
        let result = run(&args);
        assert!(!result.status.success());
        assert!(!PathBuf::from(&output).exists());
        assert!(!run(&args[..6]).status.success());
    }
}
