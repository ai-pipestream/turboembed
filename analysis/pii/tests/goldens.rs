//! Reviewed PII pack snapshots and layer invariants.
//!
//! `tests/snapshots/<pack>/<id>.tsv` holds the `turbo:pii` rows for each
//! input in `tests/inputs/`. Regenerate with
//! `TURBO_BLESS=1 cargo test -p turbo-pii --test goldens`.
//! JVM goldens are not an oracle. Each case checks `Layer::check`, spans in
//! bounds, non-empty normalized forms, and that kind names parse back.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Document, Layer, ViewId};
use turbo_pii::{Pack, Pii, PiiDetector, PiiKind};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn input_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(root().join("inputs"))
        .expect("tests/inputs")
        .map(|e| {
            let name = e.expect("entry").file_name().into_string().expect("utf-8");
            name.strip_suffix(".txt").expect(".txt").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn text_of(id: &str) -> String {
    std::fs::read_to_string(root().join("inputs").join(format!("{id}.txt"))).unwrap()
}

fn snap_path(pack: &str, id: &str) -> PathBuf {
    root().join("snapshots").join(pack).join(format!("{id}.tsv"))
}

fn render(layer: &Pii) -> String {
    let mut out = String::new();
    for m in layer.mentions() {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\n",
            m.span.start,
            m.span.end,
            m.kind.name(),
            m.confidence.name(),
            m.normalized,
        ));
    }
    out
}

fn check_invariants(text: &str, layer: &Pii) {
    layer.check().unwrap_or_else(|e| panic!("Layer::check: {e}"));
    let len = text.len() as u32;
    for m in layer.mentions() {
        assert!(m.span.start < m.span.end && m.span.end <= len, "span out of bounds: {:?}", m.span);
        assert!(!m.normalized.is_empty());
        assert_eq!(PiiKind::parse(m.kind.name()), Some(m.kind), "kind name round-trips");
        assert!(text.is_char_boundary(m.span.start as usize));
        assert!(text.is_char_boundary(m.span.end as usize));
    }
}

fn check(pack: Pack) -> usize {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let ids = input_ids();
    assert!(ids.len() > 50, "only {} inputs", ids.len());
    let analyzer = Analyzer::builder().add(PiiDetector::pack(pack).keeping_nested()).build().expect("analyzer");
    let empty = Pii::new(ViewId::SOURCE);
    let mut mentions = 0;
    for id in &ids {
        let text = text_of(id);
        let mut docs = vec![Document::new(text.as_str())];
        for result in analyzer.run(&mut docs) {
            result.unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let layer = docs[0].get::<Pii>().unwrap_or(&empty);
        mentions += layer.spans.len();
        check_invariants(&text, layer);
        let body = render(layer);
        let path = snap_path(pack.name(), id);
        if bless {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &body).unwrap();
            continue;
        }
        let want = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing snapshot {} (TURBO_BLESS=1 to write): {e}", path.display()));
        assert_eq!(body, want, "{}/{}", pack.name(), id);
    }
    mentions
}

macro_rules! pack_tests {
    ($($fn_name:ident: $pack:expr, $min:expr;)*) => {$(
        #[test]
        fn $fn_name() {
            let n = check($pack);
            assert!(n >= $min, "only {n} mentions compared for {}", $pack.name());
        }
    )*};
}

pack_tests! {
    payment_snapshots: Pack::Payment, 15;
    contact_snapshots: Pack::Contact, 25;
    network_snapshots: Pack::Network, 10;
    secrets_snapshots: Pack::Secrets, 4;
    crypto_snapshots: Pack::Crypto, 5;
    us_identity_snapshots: Pack::UsIdentity, 3;
    eu_identity_snapshots: Pack::EuIdentity, 3;
    ca_identity_snapshots: Pack::CaIdentity, 1;
    device_snapshots: Pack::Device, 1;
    all_structured_snapshots: Pack::AllStructured, 80;
}
