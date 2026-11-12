use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

static NEXT: AtomicU64 = AtomicU64::new(0);

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbo-eval-main-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_usage_names_every_set() {
    for s in sets::ALL {
        assert!(USAGE.contains(s.name), "{} is missing from the usage", s.name);
    }
}

#[test]
fn the_level_flags_are_checked() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let e = TurboEmbedder::open(&dir, Device::Cpu, Precision::Model).unwrap();
    let built = |extra: &[&str]| {
        let mut a = ["--chunker", "turbo-chunker", "--max", "64", "--target", "48"].map(String::from).to_vec();
        a.extend(extra.iter().map(|s| s.to_string()));
        let mut o = Opts::parse(&a).unwrap();
        chunker(&mut o, &e, (&dir, &Device::Cpu, Precision::Model))
    };
    let refused = |extra: &[&str]| built(extra).err().unwrap();
    assert!(refused(&["--level", "paragraph"]).contains("chunk, section or document"));
    assert!(refused(&["--level", "chunk", "--level-vectors", "late"]).contains("local or contextual"));
    assert!(refused(&["--level-vectors", "local"]).contains("needs --level"));
    let config = built(&["--level", "section", "--level-vectors", "contextual"]).unwrap().config();
    assert!(config.iter().any(|s| s.name == "level" && s.value == "section"));
    assert!(config.iter().any(|s| s.name == "level_vectors" && s.value.starts_with("contextual")));
}

#[test]
fn the_sentence_tier_defaults_to_rules_and_sat_needs_a_bundle() {
    let built = |args: &[&str]| {
        let a = args.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut o = Opts::parse(&a).unwrap();
        sentencer(&mut o, &Device::Cpu, Precision::Model)
    };
    for args in [&[][..], &["--sentences", "rule"][..]] {
        assert!(matches!(built(args).unwrap(), Sentencer::Rules(..)));
    }
    assert!(matches!(built(&["--sentences", "newline"]).unwrap(), Sentencer::Newline));
    let err = |args: &[&str]| match built(args) {
        Ok(_) => panic!("sentence options unexpectedly accepted: {args:?}"),
        Err(e) => e,
    };
    assert!(err(&["--sentences", "sat"]).contains("--sat-bundle"));
    assert!(err(&["--sat-bundle", "some-bundle"]).contains("--sentences sat"));
    assert!(err(&["--sentences", "model.bin"]).contains("rules, rules:<tag>, newline or sat"));
}

#[test]
fn an_unknown_set_is_refused_by_name() {
    let e = load_set("no-such-set", Path::new("/nonexistent")).unwrap_err();
    assert!(e.contains("no-such-set"), "{e}");
}

#[test]
fn dump_cuts_options_validate_encryption_flags_and_key_files() {
    let root = scratch("dump-cuts-opts");
    let cuts_dir = root.join("cuts");
    let key_file = root.join("keys.txt");
    fs::write(
        &key_file,
        "01010101010101010101010101010101:4747474747474747474747474747474747474747474747474747474747474747\n",
    )
    .unwrap();
    let parse_dump = |args: &[&str]| {
        let a = args.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut o = Opts::parse(&a).unwrap();
        dump_cuts_opts(&mut o, Precision::Exact)
    };
    assert!(parse_dump(&[]).unwrap().is_none());
    let plain = parse_dump(&["--dump-cuts", cuts_dir.to_str().unwrap()]).unwrap().unwrap();
    assert!(!plain.opaque_keys);
    let enc = parse_dump(&[
        "--dump-cuts",
        cuts_dir.to_str().unwrap(),
        "--cuts-key-file",
        key_file.to_str().unwrap(),
        "--cuts-principal",
        "11111111111111111111111111111111",
    ])
    .unwrap()
    .unwrap();
    assert!(enc.opaque_keys);
    let err = |args: &[&str]| match parse_dump(args) {
        Ok(_) => panic!("unexpectedly accepted {args:?}"),
        Err(e) => e,
    };
    assert!(err(&["--cuts-key-file", key_file.to_str().unwrap()]).contains("--dump-cuts"));
    assert!(
        err(&["--dump-cuts", cuts_dir.to_str().unwrap(), "--cuts-principal", "11111111111111111111111111111111"])
            .contains("--cuts-key-file")
    );
    assert!(
        err(&[
            "--dump-cuts",
            cuts_dir.to_str().unwrap(),
            "--cuts-key-file",
            key_file.to_str().unwrap(),
            "--cuts-principal",
            "short",
        ])
        .contains("32 lower-case hex digits")
    );
    let revoked_file = root.join("revoked.txt");
    fs::write(&revoked_file, "revoked:01010101010101010101010101010101\n").unwrap();
    assert!(
        err(&["--dump-cuts", cuts_dir.to_str().unwrap(), "--cuts-key-file", revoked_file.to_str().unwrap()])
            .contains("artifact key revoked")
    );
    fs::remove_dir_all(root).unwrap();
}
