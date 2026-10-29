//! Reviewed PII rewrite snapshots and rewrite invariants.
//!
//! `tests/rewrite_snapshots/<pack>/<id>.txt` holds the masked text, the
//! mask-all text, the two rewrites (or `OVERLAP`), the labels, and the
//! audit report for each input. Regenerate with
//! `TURBO_BLESS=1 cargo test -p turbo-pii --test rewrite_goldens`.
//! JVM goldens are not an oracle. Each case checks that rewrite labels map
//! back to the source mentions and that refused overlaps agree for both
//! operations.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Document, Span};
use turbo_pii::rewrite::{
    HmacTokenizer, MASKED, MaskedView, PiiAuditReport, PiiRewrite, Pseudonymizer, RewriteError, mask_all,
};
use turbo_pii::{Pack, Pii, PiiDetector};

const KEY: &str = "0123456789abcdef0123456789abcdef";

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
    root().join("rewrite_snapshots").join(pack).join(format!("{id}.txt"))
}

/// Labels of `r` with a check that the offset map agrees.
fn labels(r: &PiiRewrite, pii: &Pii) -> Vec<(u32, u32, u32, u32, String)> {
    let mut replaced: Vec<Span> = pii.mentions().map(|m| m.span).collect();
    replaced.sort_by_key(|s| s.start);
    assert_eq!(replaced.len(), r.labels().mentions().count());
    replaced
        .iter()
        .zip(r.labels().mentions())
        .map(|(s, l)| {
            assert_eq!(r.map_offset(s.start), l.span.start);
            assert_eq!(r.map_span(*s), l.span);
            (s.start, s.end, l.span.start, l.span.end, l.normalized.to_string())
        })
        .collect()
}

fn rewrite_line(op: &str, text: &str, pii: &Pii, tokenizer: &HmacTokenizer) -> (String, usize) {
    let rust = match op {
        "pseudonymize" => Pseudonymizer::default().refusing_overlaps().rewrite(text, pii),
        "hmac" => tokenizer.rewrite(text, pii),
        _ => unreachable!(),
    };
    match rust {
        Ok(r) => {
            let labs = labels(&r, pii);
            let mut line = format!("REWRITE\t{op}\tOK\t{}\n", r.text());
            for (ss, se, s, e, lab) in &labs {
                line.push_str(&format!("LABEL\t{op}\t{ss}\t{se}\t{s}\t{e}\t{lab}\n"));
            }
            (line, labs.len())
        }
        Err(RewriteError::Overlap(_, _)) => (format!("REWRITE\t{op}\tOVERLAP\t\n"), 0),
        Err(e) => panic!("{op}: {e}"),
    }
}

fn render(text: &str, pii: &Pii, masked_text: &str, tokenizer: &HmacTokenizer) -> (String, usize, usize) {
    let mut out = String::new();
    out.push_str(&format!("MASKED\t{masked_text}\n"));
    out.push_str(&format!("MASK_ALL\t{}\n", mask_all(text, pii, '#').unwrap()));
    let mut label_count = 0;
    let mut refused = 0;
    for op in ["pseudonymize", "hmac"] {
        let (line, n) = rewrite_line(op, text, pii, tokenizer);
        if line.contains("\tOVERLAP\t") {
            refused += 1;
        }
        label_count += n;
        out.push_str(&line);
    }
    let report = PiiAuditReport::of(pii, tokenizer, PiiAuditReport::DEFAULT_SAMPLES);
    for k in &report.kinds {
        out.push_str(&format!("AUDIT\t{}\t{}\t{}\t{}\n", k.kind.name(), k.mentions, k.distinct, k.samples.join(", ")));
    }
    out.push_str(&format!("AUDIT_REPORT\t{}\n", report.to_string().replace('\n', "\\n")));
    (out, label_count, refused)
}

struct Counts {
    labels: usize,
    refused: usize,
}

fn check(pack: Pack) -> Counts {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let ids = input_ids();
    assert!(ids.len() > 50, "only {} inputs", ids.len());
    let analyzer = Analyzer::builder()
        .add(PiiDetector::pack(pack).keeping_nested())
        .step(MaskedView::by_kind('*').unwrap().bound())
        .build()
        .expect("analyzer");
    let tokenizer = HmacTokenizer::new(KEY.as_bytes()).unwrap().refusing_overlaps();
    let mut counts = Counts { labels: 0, refused: 0 };
    for id in &ids {
        let text = text_of(id);
        let mut docs = vec![Document::new(text.as_str())];
        for result in analyzer.run(&mut docs) {
            result.unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let empty = Pii::new(turbo_analysis::ViewId::SOURCE);
        let pii = docs[0].get::<Pii>().unwrap_or(&empty);
        let masked = docs[0].get_key(&MASKED).expect("masked view");
        let (body, labels, refused) = render(&text, pii, masked.text(), &tokenizer);
        counts.labels += labels;
        counts.refused += refused;
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
    counts
}

macro_rules! pack_tests {
    ($($fn_name:ident: $pack:expr, $labels:expr, $refused:expr;)*) => {$(
        #[test]
        fn $fn_name() {
            let c = check($pack);
            assert!(c.labels >= $labels, "only {} labels compared", c.labels);
            assert_eq!(c.refused, $refused, "rewrites refused for overlapping mentions");
        }
    )*};
}

pack_tests! {
    payment_rewrite_snapshots: Pack::Payment, 40, 0;
    contact_rewrite_snapshots: Pack::Contact, 68, 0;
    network_rewrite_snapshots: Pack::Network, 22, 2;
    secrets_rewrite_snapshots: Pack::Secrets, 12, 0;
    crypto_rewrite_snapshots: Pack::Crypto, 14, 0;
    us_identity_rewrite_snapshots: Pack::UsIdentity, 8, 0;
    eu_identity_rewrite_snapshots: Pack::EuIdentity, 10, 0;
    ca_identity_rewrite_snapshots: Pack::CaIdentity, 4, 0;
    device_rewrite_snapshots: Pack::Device, 4, 0;
    all_structured_rewrite_snapshots: Pack::AllStructured, 126, 16;
}
