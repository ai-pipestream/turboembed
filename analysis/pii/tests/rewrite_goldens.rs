//! Parity of masking, numbered and keyed labels and the audit report with
//! OpenNLP's `Masker`, `Pseudonymizer`, `HmacTokenizer` and
//! `PiiAuditReport` on the JVM, under the test key.
//!
//! For every text in `analysis/goldens/data/inputs/pii/` and every pack,
//! `pii-rewrite/<pack>/pii/<id>.golden` holds what OpenNLP made of the
//! mentions the pack found (see `analysis/goldens-tool/README.md`): the
//! edits of the masked text, the text with every mention masked, the two
//! rewrites or the exception each threw, their labels, and the audit
//! report. Each test runs the pack and `MaskedView` in an analyzer and
//! compares all of them exactly.
//!
//! The goldens come from `cargo run -p turbo-goldens-tool -- generate`;
//! while the `pii-rewrite` directory is not on disk the tests say so and
//! pass.

use turbo_analysis::{Analyzer, Document, Span};
use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape, data_dir, known_shapes};
use turbo_pii::rewrite::{
    HmacTokenizer, MASKED, MaskedView, PiiAuditReport, PiiRewrite, Pseudonymizer, RewriteError, mask_all,
};
use turbo_pii::{Pack, Pii, PiiDetector};

/// The key the goldens were made with, as their `config key` record
/// writes it.
const KEY: &str = "0123456789abcdef0123456789abcdef";

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const STR: ColumnType = ColumnType::Str;
const U32: ColumnType = ColumnType::U32;

const TEXT: &[ColumnSpec] = &[col("text", STR)];
const REWRITES: &[ColumnSpec] = &[col("operation", STR), col("text", STR), col("error", STR)];
const LABELS: &[ColumnSpec] = &[
    col("operation", STR),
    col("source_start", U32),
    col("source_end", U32),
    col("start", U32),
    col("end", U32),
    col("label", STR),
];
const AUDIT: &[ColumnSpec] = &[col("type", STR), col("mentions", U32), col("distinct", U32), col("samples", STR)];

fn shapes() -> Vec<Shape> {
    let doc = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    let mut shapes = known_shapes();
    shapes.extend([
        doc("turbo:pii-mask-all", TEXT),
        doc("turbo:pii-rewrites", REWRITES),
        doc("turbo:pii-labels", LABELS),
        doc("turbo:pii-audit", AUDIT),
        doc("turbo:pii-audit-report", TEXT),
    ]);
    shapes
}

/// The ids of the texts in `inputs/pii`, sorted.
fn inputs() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(data_dir().join("inputs/pii"))
        .expect("the pii input set")
        .map(|e| {
            let name = e.expect("a directory entry").file_name().into_string().expect("a UTF-8 file name");
            name.strip_suffix(".txt").expect("an input set holds only <id>.txt files").to_string()
        })
        .collect();
    ids.sort();
    ids
}

/// Whether the goldens are on disk; when not, says why the test passes.
fn goldens_present() -> bool {
    let dir = data_dir().join("pii-rewrite");
    let present = dir.is_dir();
    if !present {
        eprintln!(
            "skipped: {} is absent; make the goldens with `cargo run -p turbo-goldens-tool -- generate`",
            dir.display()
        );
    }
    present
}

fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layer(id).unwrap_or_else(|| panic!("{}: no layer {id}", g.header.input.id))
}

fn u32s<'l>(l: &'l GoldenLayer, name: &str) -> &'l [u32] {
    l.u32s(name).unwrap_or_else(|| panic!("{}: no u32 column {name}", l.id))
}

fn strs(l: &GoldenLayer, name: &str) -> Vec<String> {
    l.strs(name).unwrap_or_else(|| panic!("{}: no str column {name}", l.id)).iter().map(str::to_string).collect()
}

fn config<'g>(golden: &'g Golden, key: &str) -> &'g str {
    let entry = golden.header.config.iter().find(|c| c.key == key);
    &entry.unwrap_or_else(|| panic!("no {key} config record")).value
}

/// One label as the goldens hold it: the replaced span of the input, the
/// label's span in the new text, and the label.
type Label = (u32, u32, u32, u32, String);

/// The labels of `r`, a rewrite of `pii`'s mentions, with the spans they
/// replaced, and a check that the offset map agrees with them.
fn labels(r: &PiiRewrite, pii: &Pii) -> Vec<Label> {
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

/// Records a difference between the port and the JVM.
fn differ(failures: &mut Vec<String>, id: &str, what: &str, rust: String, jvm: String) {
    if rust != jvm {
        failures.push(format!("{id}: {what}\n  rust: {rust}\n  jvm:  {jvm}"));
    }
}

/// What one comparison counted.
#[derive(Default)]
struct Counts {
    masked: usize,
    rewrites: usize,
    refused: usize,
    labels: usize,
    audited: usize,
}

/// Compares the rewrites of `pack` over every text with the goldens.
fn check(pack: Pack) -> Counts {
    let ids = inputs();
    assert!(ids.len() > 50, "only {} inputs in the pii set", ids.len());
    let analyzer = Analyzer::builder()
        .add(PiiDetector::pack(pack).keeping_nested())
        .step(MaskedView::by_kind('*').unwrap().bound())
        .build()
        .expect("the analyzer");
    let tokenizer = HmacTokenizer::new(KEY.as_bytes()).unwrap().refusing_overlaps();
    let shapes = shapes();
    let mut counts = Counts::default();
    let mut failures = Vec::new();
    for id in &ids {
        let path = data_dir().join("pii-rewrite").join(pack.name()).join("pii").join(format!("{id}.golden"));
        let g = Golden::read_with(&path, &shapes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(config(&g, "pack"), pack.name(), "{id}: the golden was made with another pack");
        assert_eq!(config(&g, "key"), KEY, "{id}: the golden was made with another key");
        assert_eq!((config(&g, "mask"), config(&g, "mask-all")), ("*", "#"), "{id}");
        let text = g.input_text(data_dir()).unwrap_or_else(|e| panic!("{e}"));
        let mut docs = vec![Document::new(text.as_str())];
        for result in analyzer.run(&mut docs) {
            result.unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let pii = docs[0].get::<Pii>().expect("the pii layer");

        // The masked view, edit for edit.
        let masked = docs[0].get_key(&MASKED).expect("the masked view");
        let edits = layer(&g, "turbo:view");
        let (starts, ends, with) = (u32s(edits, "start"), u32s(edits, "end"), strs(edits, "with"));
        let jvm: Vec<(u32, u32, &str)> = (0..edits.len()).map(|i| (starts[i], ends[i], with[i].as_str())).collect();
        let rust: Vec<(u32, u32, &str)> = masked
            .alignment()
            .edits()
            .iter()
            .map(|e| (e.parent.start, e.parent.end, &masked.text()[e.child.range()]))
            .collect();
        counts.masked += rust.len();
        differ(&mut failures, id, "masked view edits", format!("{rust:?}"), format!("{jvm:?}"));
        let all = &strs(layer(&g, "turbo:pii-mask-all"), "text")[0];
        differ(&mut failures, id, "every mention masked", mask_all(&text, pii, '#').unwrap(), all.clone());

        // The two rewrites, or the overlap both refuse.
        let rewrites = layer(&g, "turbo:pii-rewrites");
        let golden_labels = layer(&g, "turbo:pii-labels");
        let (ops, label_ops) = (strs(rewrites, "operation"), strs(golden_labels, "operation"));
        assert_eq!(ops, ["pseudonymize", "hmac"], "{id}");
        let (texts, errors) = (strs(rewrites, "text"), strs(rewrites, "error"));
        for (k, op) in ops.iter().enumerate() {
            let rust = match op.as_str() {
                "pseudonymize" => Pseudonymizer::default().refusing_overlaps().rewrite(&text, pii),
                _ => tokenizer.rewrite(&text, pii),
            };
            match rust {
                Ok(_) if !errors[k].is_empty() => {
                    failures.push(format!("{id}: {op}: the JVM refused ({}), the port rewrote", errors[k]));
                }
                Ok(r) => {
                    differ(&mut failures, id, op, r.text().to_string(), texts[k].clone());
                    let ss = u32s(golden_labels, "source_start");
                    let se = u32s(golden_labels, "source_end");
                    let (s, e) = (u32s(golden_labels, "start"), u32s(golden_labels, "end"));
                    let l = strs(golden_labels, "label");
                    let jvm: Vec<Label> = (0..golden_labels.len())
                        .filter(|&i| label_ops[i] == *op)
                        .map(|i| (ss[i], se[i], s[i], e[i], l[i].clone()))
                        .collect();
                    let rust = labels(&r, pii);
                    counts.rewrites += 1;
                    counts.labels += rust.len();
                    differ(&mut failures, id, &format!("{op} labels"), format!("{rust:?}"), format!("{jvm:?}"));
                }
                // The JVM's message gives UTF-16 offsets; that it is the
                // overlap refusal is what is compared.
                Err(RewriteError::Overlap(a, b)) => {
                    counts.refused += 1;
                    if !errors[k].starts_with("mentions must not overlap") {
                        failures
                            .push(format!("{id}: {op}: the port refused {a:?} and {b:?}, the JVM: {:?}", errors[k]));
                    }
                }
                Err(e) => failures.push(format!("{id}: {op}: {e}")),
            }
        }

        // The audit report, kind by kind and as text.
        let report = PiiAuditReport::of(pii, &tokenizer, PiiAuditReport::DEFAULT_SAMPLES);
        let audit = layer(&g, "turbo:pii-audit");
        let (types, mentions, distinct, samples) =
            (strs(audit, "type"), u32s(audit, "mentions"), u32s(audit, "distinct"), strs(audit, "samples"));
        let jvm: Vec<(String, u32, u32, String)> =
            (0..audit.len()).map(|i| (types[i].clone(), mentions[i], distinct[i], samples[i].clone())).collect();
        let rust: Vec<(String, u32, u32, String)> = report
            .kinds
            .iter()
            .map(|k| (k.kind.name().to_string(), k.mentions as u32, k.distinct as u32, k.samples.join(", ")))
            .collect();
        counts.audited += report.total;
        differ(&mut failures, id, "audit", format!("{rust:?}"), format!("{jvm:?}"));
        let shown = &strs(layer(&g, "turbo:pii-audit-report"), "text")[0];
        differ(&mut failures, id, "audit report", report.to_string(), shown.clone());
    }
    assert!(
        failures.is_empty(),
        "{} differences from the JVM in the {} pack:\n{}",
        failures.len(),
        pack.name(),
        failures.join("\n")
    );
    counts
}

macro_rules! pack_tests {
    ($($name:ident: $pack:expr, $labels:expr, $refused:expr;)*) => {$(
        #[test]
        fn $name() {
            if goldens_present() {
                let c = check($pack);
                assert!(c.labels >= $labels, "only {} labels compared", c.labels);
                assert_eq!(c.refused, $refused, "rewrites refused for overlapping mentions");
                assert!(c.masked > 0 && c.rewrites > 0 && c.audited > 0);
            }
        }
    )*};
}

// Per pack: the least number of labels compared (two per mention, one for
// each rewrite) and how many rewrites both sides refuse for overlapping
// mentions.
pack_tests! {
    payment_rewrites_match_the_jvm: Pack::Payment, 40, 0;
    contact_rewrites_match_the_jvm: Pack::Contact, 68, 0;
    network_rewrites_match_the_jvm: Pack::Network, 22, 2;
    secrets_rewrites_match_the_jvm: Pack::Secrets, 12, 0;
    crypto_rewrites_match_the_jvm: Pack::Crypto, 14, 0;
    us_identity_rewrites_match_the_jvm: Pack::UsIdentity, 8, 0;
    eu_identity_rewrites_match_the_jvm: Pack::EuIdentity, 10, 0;
    ca_identity_rewrites_match_the_jvm: Pack::CaIdentity, 4, 0;
    device_rewrites_match_the_jvm: Pack::Device, 4, 0;
    all_structured_rewrites_match_the_jvm: Pack::AllStructured, 126, 16;
}
