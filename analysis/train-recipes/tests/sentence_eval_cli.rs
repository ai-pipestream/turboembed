//! Project-authored fixtures exercise the actual offline binary, not accuracy.
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use turbo_eval::model_gate::ModelGateRecord;
use turbo_zoo::Sha256;

struct Corpus(PathBuf);
impl Corpus {
    fn new(text: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "turbo-sentence-eval-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("test.conllu");
        std::fs::write(&path, text).unwrap();
        Self(path)
    }
    fn run(&self, digest: &str, language: &str) -> Output {
        Command::new(env!("CARGO_BIN_EXE_turbo-sentence-eval"))
            .args(["ud", self.0.to_str().unwrap(), digest, language, "project-fixture-v1/test"])
            .output()
            .unwrap()
    }
}
impl Drop for Corpus {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap();
        std::fs::remove_dir(self.0.parent().unwrap()).unwrap();
    }
}
fn fixture(count: usize) -> String {
    (0..count).map(|i| format!(
        "# sent_id = {i}\n# text = Hello world.\n1\tHello\thello\tINTJ\t_\t_\t0\troot\t_\t_\n2\tworld\tworld\tNOUN\t_\t_\t1\tdep\t_\tSpaceAfter=No\n3\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_\n\n"
    )).collect()
}
fn failure(output: Output, expected: &str) {
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty(), "failed input must not publish a partial record");
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains(expected), "{error}");
}

#[test]
fn rules_score_real_conllu_reader_output_with_verified_provenance() {
    let source = fixture(300);
    let input = Corpus::new(&source);
    let digest = Sha256::of(source.as_bytes()).to_string();
    let output = input.run(&digest, "en");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let record: ModelGateRecord = serde_json::from_slice(&output.stdout).unwrap();
    record.check().unwrap();
    assert_eq!(record.units.len(), 30);
    assert_eq!(record.set.corpus_sha256, digest);
    assert_eq!(record.set.name, "project-fixture-v1/test");
    assert_eq!(record.reference.id, "sentence-newline-v1");
    assert!(record.candidate.id.starts_with("turbo-sentdetect:rules:v3:en:config-"));
    assert!(record.candidate.id.ends_with(":group-10"));
    assert_eq!(
        record.candidate.sha256,
        Sha256::of(&std::fs::read(env!("CARGO_BIN_EXE_turbo-sentence-eval")).unwrap()).to_string()
    );
    assert!(record.result.pass);
    assert!(!String::from_utf8(output.stdout).unwrap().contains("Hello world"));
    // Repeated invocations are identical, without times, temporary paths or text.
    assert_eq!(input.run(&digest, "en").stdout, record.to_json().into_bytes());
}

#[test]
fn refuses_digest_mismatch_before_parsing_data() {
    let input = Corpus::new("not CoNLL-U");
    failure(input.run(&"0".repeat(64), "en"), "corpus SHA-256 mismatch");
    failure(input.run("not-a-digest", "en"), "64 lowercase");
}

#[test]
fn missing_sentence_text_is_never_scored_as_null() {
    let source = fixture(300).replacen("# text = Hello world.\n", "", 1);
    let input = Corpus::new(&source);
    failure(input.run(&Sha256::of(source.as_bytes()).to_string(), "en"), "sentence 1 requires nonblank # text");
}

#[test]
fn blank_sentence_text_is_refused() {
    let source = fixture(300).replacen("# text = Hello world.", "# text =   ", 1);
    let input = Corpus::new(&source);
    failure(input.run(&Sha256::of(source.as_bytes()).to_string(), "en"), "requires nonblank # text");
}

#[test]
fn unknown_language_and_small_corpora_are_refused() {
    let source = fixture(10);
    let input = Corpus::new(&source);
    let digest = Sha256::of(source.as_bytes()).to_string();
    failure(input.run(&digest, "invalid-language"), "supported lowercase language tag");
    let output = input.run(&digest, "en");
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr).unwrap().contains("30"));
}

#[test]
fn malformed_utf8_and_unrecorded_masc_are_refused() {
    let input = Corpus::new("");
    std::fs::write(&input.0, [0xff]).unwrap();
    failure(input.run(&Sha256::of(&[0xff]).to_string(), "en"), "utf-8");
    let output = Command::new(env!("CARGO_BIN_EXE_turbo-sentence-eval"))
        .arg("masc")
        .arg(input.0.parent().unwrap())
        .output()
        .unwrap();
    failure(output, "digest");
}

#[test]
fn usage_rejects_extra_arguments_and_explains_exit_statuses() {
    let binary = env!("CARGO_BIN_EXE_turbo-sentence-eval");
    let help = Command::new(binary).arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8(help.stdout).unwrap().contains("2 failed gate"));
    failure(Command::new(binary).args(["masc", ".", "unexpected"]).output().unwrap(), "usage:");
}

#[test]
fn failed_accuracy_gate_still_emits_a_valid_record_and_nonzero_status() {
    // Gold intentionally groups two sentences together; the final one-item
    // group gives newline a correct span while the rules split it in two.
    let source = fixture(291).replace("# text = Hello world.", "# text = Hello. World.");
    let input = Corpus::new(&source);
    let output = input.run(&Sha256::of(source.as_bytes()).to_string(), "en");
    assert_eq!(output.status.code(), Some(2), "{}", String::from_utf8_lossy(&output.stderr));
    let record: ModelGateRecord = serde_json::from_slice(&output.stdout).unwrap();
    record.check().unwrap();
    assert!(!record.result.pass);
    assert_eq!(record.units.len(), 30);
}
