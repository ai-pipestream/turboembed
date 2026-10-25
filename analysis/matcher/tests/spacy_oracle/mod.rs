//! Optional, local-only output from the real spaCy reference container.
use turbo_matcher::spacy::ImportError;

// Read canonical JSON with the same bounded reader, without a dependency.
#[allow(dead_code)]
#[path = "../../src/spacy/json.rs"]
mod json;
use json::Value;

pub(super) struct Case {
    pub text: String,
    pub spans: Vec<(u32, u32)>,
    pub pos: Vec<String>,
    pub tags: Vec<String>,
    pub lemmas: Vec<String>,
    pub morph: Vec<String>,
    pub hits: Vec<(String, usize, usize)>,
}

fn array(v: &Value) -> &[Value] {
    let Value::Array(a) = v else { panic!("oracle: expected an array") };
    a
}
fn field<'a>(v: &'a Value, name: &str) -> &'a Value {
    let Value::Object(fields) = v else { panic!("oracle: expected an object") };
    &fields.iter().find(|(k, _)| k == name).unwrap_or_else(|| panic!("oracle: missing {name}")).1
}
fn string(v: &Value) -> String {
    let Value::String(s) = v else { panic!("oracle: expected a string") };
    s.clone()
}
fn number(v: &Value) -> usize {
    let Value::Number(n) = v else { panic!("oracle: expected a number") };
    n.parse().expect("oracle: nonnegative integer")
}
fn strings(v: &Value) -> Vec<String> {
    array(v).iter().map(string).collect()
}

pub(super) fn load() -> Option<Vec<Case>> {
    let Some(path) = std::env::var_os("TURBO_SPACY_ORACLE") else {
        assert!(
            !std::env::var("TURBO_TEST_REQUIRE_SPACY_ORACLE").is_ok_and(|v| v == "1"),
            "TURBO_SPACY_ORACLE is unset with TURBO_TEST_REQUIRE_SPACY_ORACLE=1; see tests/data/README.md"
        );
        eprintln!("skipping external EWT oracle: set TURBO_SPACY_ORACLE; see tests/data/README.md");
        return None;
    };
    let raw = std::fs::read_to_string(path).expect("read TURBO_SPACY_ORACLE");
    let root = json::parse(&raw).expect("valid spaCy oracle JSON");
    assert_eq!(string(field(&root, "spacy")), "3.8.7");
    assert_eq!(
        string(field(&root, "corpus_sha256")),
        "3fb78e2b55482c8ee00caa653af436c88c9e5bc9a73ce26e4ae3ddc79d2b7e7b"
    );
    Some(
        array(field(&root, "cases"))
            .iter()
            .map(|v| Case {
                text: string(field(v, "text")),
                spans: array(field(v, "spans"))
                    .iter()
                    .map(|v| {
                        let a = array(v);
                        assert_eq!(a.len(), 2);
                        (u32::try_from(number(&a[0])).unwrap(), u32::try_from(number(&a[1])).unwrap())
                    })
                    .collect(),
                pos: strings(field(v, "pos")),
                tags: strings(field(v, "tags")),
                lemmas: strings(field(v, "lemmas")),
                morph: strings(field(v, "morph")),
                hits: array(field(v, "hits"))
                    .iter()
                    .map(|v| {
                        let a = array(v);
                        assert_eq!(a.len(), 3);
                        (string(&a[0]), number(&a[1]), number(&a[2]))
                    })
                    .collect(),
            })
            .collect(),
    )
}
