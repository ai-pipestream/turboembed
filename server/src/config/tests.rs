//! Parsing of the model settings, the flags and their variables.

use super::*;

fn args(s: &str) -> Result<Args, String> {
    Args::parse_with(s.split_whitespace().map(String::from), |_| None)
}

#[test]
fn a_model() {
    let a = args("--listen 127.0.0.1:0 --model bundle=/b/tiny,device=0,sessions=2").unwrap();
    assert_eq!(a.max_message_bytes, 64 * 1024 * 1024);
    let b = args("--listen 127.0.0.1:0 --max-message-bytes 1024 --model bundle=/b/tiny,device=0,sessions=2");
    assert_eq!(b.unwrap().max_message_bytes, 1024);
    assert_eq!(
        a.models,
        [ModelConfig {
            bundle: "/b/tiny".into(),
            name: None,
            device: Device::Index(0),
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            sessions: 2
        }]
    );
    let a = args(
        "--listen=[::1]:8001 --model=bundle=x,device=select,sessions=1,precision=PRECISION_EXACT,max_batch=8,max_seq=128",
    )
    .unwrap();
    let m = &a.models[0];
    assert_eq!((m.device, m.precision, m.max_batch, m.max_seq), (Device::Select, 2, 8, 128));
    assert_eq!(m.name(), Some("x"));
}

#[test]
fn one_bundle_under_two_names() {
    let a = args(
        "--listen 127.0.0.1:0 --model bundle=/b/tiny,device=0,sessions=1 \
         --model name=tiny-fastest,bundle=/b/tiny,device=0,sessions=1,precision=PRECISION_FASTEST",
    )
    .unwrap();
    assert_eq!(names(&a.models).unwrap(), ["tiny", "tiny-fastest"]);
    assert_eq!(a.models[1].name.as_deref(), Some("tiny-fastest"));
    assert!(args("--listen 127.0.0.1:0 --model name=,bundle=/b/tiny,device=0,sessions=1").is_err());
    assert!(args("--listen 127.0.0.1:0 --model name=a,name=b,bundle=/b/tiny,device=0,sessions=1").is_err());
    // A name is a name wherever it comes from.
    assert!(
        args("--listen 127.0.0.1:0 --model bundle=/b/tiny,device=0,sessions=1 --model name=tiny,bundle=/c/x,device=0,sessions=1")
            .is_err()
    );
}

#[test]
fn the_environment_fills_in_absent_flags() {
    let env = |k: &str| match k {
        ENV_LISTEN => Some("127.0.0.1:9000".to_string()),
        ENV_MODELS => Some("bundle=/b/a,device=0,sessions=1; bundle=/b/b,device=1,sessions=2".to_string()),
        ENV_MAX_MESSAGE_BYTES => Some("4096".to_string()),
        _ => None,
    };
    let a = Args::parse_with(std::iter::empty(), env).unwrap();
    assert_eq!(a.listen, "127.0.0.1:9000".parse().unwrap());
    assert_eq!(a.max_message_bytes, 4096);
    assert_eq!(names(&a.models).unwrap(), ["a", "b"]);
    assert_eq!((a.models[1].device, a.models[1].sessions), (Device::Index(1), 2));

    // A flag replaces its variable; one --model replaces every model
    // of the variable.
    let a = Args::parse_with(
        "--listen 127.0.0.1:0 --model bundle=/b/c,device=0,sessions=1 --max-message-bytes 1"
            .split_whitespace()
            .map(String::from),
        env,
    )
    .unwrap();
    assert_eq!(a.listen, "127.0.0.1:0".parse().unwrap());
    assert_eq!(a.max_message_bytes, 1);
    assert_eq!(names(&a.models).unwrap(), ["c"]);

    // A bad variable is refused by name.
    let bad = |k: &str, v: &str| {
        let v = v.to_string();
        let k = k.to_string();
        Args::parse_with(std::iter::empty(), move |q| if q == k { Some(v.clone()) } else { env(q) }).unwrap_err()
    };
    assert!(bad(ENV_LISTEN, "nowhere").starts_with("TURBO_KSERVE_LISTEN=nowhere"));
    assert!(bad(ENV_MODELS, "bundle=/b/a,device=0").starts_with("TURBO_KSERVE_MODELS: "));
    assert!(bad(ENV_MAX_MESSAGE_BYTES, "64MiB").starts_with("TURBO_KSERVE_MAX_MESSAGE_BYTES=64MiB"));
    // An empty variable gives nothing.
    let e = Args::parse_with(std::iter::empty(), |_| Some(String::new())).unwrap_err();
    assert_eq!(e, "--listen is required");
    let a = Args::parse_with(
        "--listen 127.0.0.1:0 --model bundle=/b/c,device=0,sessions=1".split_whitespace().map(String::from),
        |_| Some(String::new()),
    )
    .unwrap();
    assert_eq!(a.max_message_bytes, DEFAULT_MAX_MESSAGE_BYTES);
}

#[test]
fn the_vocabulary_dir_comes_from_its_flag_or_its_variable() {
    let base = "--listen 127.0.0.1:0 --model bundle=/b/a,device=0,sessions=1";
    assert_eq!(args(base).unwrap().vocabulary_dir, None);
    assert_eq!(args(&format!("{base} --vocabulary-dir /v")).unwrap().vocabulary_dir.as_deref(), Some("/v"));
    let env = |k: &str| (k == ENV_VOCABULARY_DIR).then(|| "/from-env".to_string());
    let parsed = |s: &str| Args::parse_with(s.split_whitespace().map(String::from), env).unwrap();
    assert_eq!(parsed(base).vocabulary_dir.as_deref(), Some("/from-env"));
    assert_eq!(parsed(&format!("{base} --vocabulary-dir=/v")).vocabulary_dir.as_deref(), Some("/v"));
    assert!(args(&format!("{base} --vocabulary-dir /v --vocabulary-dir /w")).is_err());
    assert!(args(&format!("{base} --vocabulary-dir=")).is_err());
}

#[test]
fn the_catalog_comes_from_its_flags_or_their_variables() {
    let base = "--listen 127.0.0.1:0 --model bundle=/b/a,device=0,sessions=1";
    let a = args(base).unwrap();
    assert_eq!((a.catalog_dir, a.catalog_index), (None, None));
    let a = args(&format!("{base} --catalog-dir /z --catalog-index=/z/zoo.json")).unwrap();
    assert_eq!((a.catalog_dir.as_deref(), a.catalog_index.as_deref()), (Some("/z"), Some("/z/zoo.json")));
    let env = |k: &str| (k == ENV_CATALOG_DIR).then(|| "/from-env".to_string());
    let parsed = |s: &str| Args::parse_with(s.split_whitespace().map(String::from), env);
    assert_eq!(parsed(base).unwrap().catalog_dir.as_deref(), Some("/from-env"));
    assert_eq!(parsed(&format!("{base} --catalog-dir=/z")).unwrap().catalog_dir.as_deref(), Some("/z"));
    assert!(args(&format!("{base} --catalog-dir /z --catalog-dir /y")).is_err());
    assert!(args(&format!("{base} --catalog-dir=")).is_err());
    assert_eq!(args(&format!("{base} --catalog-index /i")).unwrap_err(), "--catalog-index needs --catalog-dir");
}

#[test]
fn refusals() {
    assert_eq!(names(&[]).unwrap_err(), "at least one --model is required");
    let zero_sessions = ModelConfig {
        bundle: "/b/tiny".into(),
        name: None,
        device: Device::Index(0),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 0,
    };
    assert_eq!(names(&[zero_sessions]).unwrap_err(), "bundle /b/tiny: sessions must be at least 1");
    for bad in [
        "--model bundle=x,device=0,sessions=1",
        "--listen 127.0.0.1:0",
        "--listen 127.0.0.1:0 --model bundle=x,device=0",
        "--listen 127.0.0.1:0 --model bundle=x,sessions=1",
        "--listen 127.0.0.1:0 --model device=0,sessions=1",
        "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=0",
        "--listen 127.0.0.1:0 --model bundle=x,device=cpu,sessions=1",
        "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1,precision=TURBO_PRECISION_EXACT",
        "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1,precision=precision_exact",
        "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1,max_batch=-1",
        "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1,sessions=2",
        "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1,colour=red",
        "--listen 127.0.0.1:0 --model bundle=a/x,device=0,sessions=1 --model bundle=b/x,device=0,sessions=1",
        "--listen 127.0.0.1:0 --model bundle=a/x/,device=0,sessions=1",
        "--listen 127.0.0.1:0 --model bundle=a/.,device=0,sessions=1",
        "--listen 127.0.0.1:0 --model bundle=..,device=0,sessions=1",
        "--listen nowhere --model bundle=x,device=0,sessions=1",
        "--listen 127.0.0.1:0 --listen 127.0.0.1:1 --model bundle=x,device=0,sessions=1",
        "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1 --max-message-bytes -1",
        "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1 --max-message-bytes 64MiB",
        "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1 --max-message-bytes 1 --max-message-bytes 2",
    ] {
        assert!(args(bad).is_err(), "{bad}");
    }
}

#[test]
fn the_name_is_the_last_component_as_spelled() {
    let m = |b: &str| ModelConfig { bundle: b.into(), ..ModelConfig::parse("bundle=x,device=0,sessions=1").unwrap() };
    assert_eq!(m("testdata/Tiny-Bert").name(), Some("Tiny-Bert"));
    assert_eq!(m("tiny").name(), Some("tiny"));
    assert_eq!(m("/a/b/").name(), None);
    assert_eq!(m("").name(), None);
    assert_eq!(m("a/..").name(), None);
    // A name setting is used as given, whatever the path.
    let n = ModelConfig { name: Some("Other".into()), ..m("/a/b/") };
    assert_eq!(n.name(), Some("Other"));
}

#[test]
fn admin_allows_empty_models_and_bundle_dir_discovers_manifests() {
    let a = args("--listen 127.0.0.1:0 --admin").unwrap();
    assert!(a.admin);
    assert!(a.models.is_empty());

    let dir = std::env::temp_dir().join(format!("turbo-kserve-bundles-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("alpha")).unwrap();
    std::fs::create_dir_all(dir.join("beta")).unwrap();
    std::fs::create_dir_all(dir.join("not-a-bundle")).unwrap();
    std::fs::write(dir.join("alpha/manifest.json"), "{}").unwrap();
    std::fs::write(dir.join("beta/manifest.json"), "{}").unwrap();
    let d = dir.to_str().unwrap().to_string();

    let a = args(&format!("--listen 127.0.0.1:0 --bundle-dir {d}")).unwrap();
    assert_eq!(names(&a.models).unwrap(), ["alpha", "beta"]);

    // Explicit --model overrides the discovered entry of the same name.
    let a = Args::parse_with(
        "--listen 127.0.0.1:0 --model name=alpha,bundle=/custom/a,device=0,sessions=4"
            .split_whitespace()
            .map(String::from),
        |k| (k == ENV_BUNDLE_DIR).then(|| d.clone()),
    )
    .unwrap();
    assert_eq!(names(&a.models).unwrap(), ["alpha", "beta"]);
    assert_eq!(a.models[0].sessions, 4);
    assert_eq!(a.models[1].sessions, 1);
    let _ = std::fs::remove_dir_all(&dir);
}
