//! `results::withheld_identity` is the one place that decides what an
//! answer says of a source it withholds (#2408): both projections answer
//! with exactly what it gives, the two contracts differ in the source
//! lengths and in nothing else, and its reader accepts what its writer
//! makes and refuses each half of it.

use prost::Message;
use turbo_analysis::export::graph_content_digest;
use turbo_analysis::{Analyzer, Document, View, ViewId, ViewTextDigest};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{InputBinding, InputIdentity, ResultIdentity};
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::search_projection::document_to_search_projection_bound;
use turbo_kserve::results::typed::document_to_typed_bound;
use turbo_kserve::results::typed::identity::{input_identity, input_identity_with, result_identity};
use turbo_kserve::results::withheld_identity::{self as rule, WithheldLengths};
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::MaskedView;

const EMAIL: &str = "alice.planted@example.com";
const MASKED: ViewId = ViewId(1);
const REWRITTEN: Disclosure = Disclosure { rewritten_text: true, full_text: false, detail: false };
const BOTH: [WithheldLengths; 2] = [WithheldLengths::Kept, WithheldLengths::Zeroed];

/// A document that is only `text`, with its findings and its masked view.
fn masked(text: &str) -> Document {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(text.to_owned())];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

fn graph_bytes(doc: &Document) -> Vec<u8> {
    turbo_analysis::export::export(doc, codecs()).unwrap().to_bytes().unwrap()
}

fn binding() -> InputBinding {
    InputBinding {
        source_revision: "rev-7".into(),
        parse_artifact_id: "parse-3".into(),
        parse_artifact_digest: vec![9; 8],
    }
}

fn holds(bytes: &[u8], needle: &[u8]) -> bool {
    bytes.windows(needle.len()).any(|w| w == needle)
}

/// The acceptance of #2408: one function decides the input and result
/// identity of both projections, each under its own contract.
#[test]
fn each_projection_answers_with_what_the_helper_gives() {
    let pii = masked(EMAIL);
    let plain = Document::new("nothing to find here".to_owned());
    let b = binding();
    for (doc, disclosure, view) in [
        (&pii, REWRITTEN, MASKED),
        (&pii, Disclosure::FULL, MASKED),
        (&pii, Disclosure::FULL, ViewId::SOURCE),
        (&plain, Disclosure::FINDINGS, ViewId::SOURCE),
    ] {
        let graph = graph_bytes(doc);
        let typed = document_to_typed_bound("d", doc, view, OffsetMode::Utf8, Some(&b), "", &disclosure).unwrap();
        let (input, result) = rule::identities(doc, Some(&b), &graph, &disclosure, WithheldLengths::Kept);
        assert_eq!((typed.input, typed.result), (Some(input), Some(result)), "typed, {disclosure:?}");

        let search =
            document_to_search_projection_bound("d", doc, view, OffsetMode::Utf8, Some(&b), &[], &disclosure).unwrap();
        let prov = search.provenance.unwrap();
        let (input, result) = rule::identities(doc, Some(&b), &graph, &disclosure, WithheldLengths::Zeroed);
        assert_eq!((prov.input, prov.result), (Some(input), Some(result)), "search, {disclosure:?}");
    }
    // A findings-only answer on the source view: the typed projection
    // alone gives one, and it is the helper's too.
    let typed =
        document_to_typed_bound("d", &pii, ViewId::SOURCE, OffsetMode::Utf8, None, "", &Disclosure::FINDINGS).unwrap();
    let identities = rule::identities(&pii, None, &graph_bytes(&pii), &Disclosure::FINDINGS, WithheldLengths::Kept);
    assert_eq!((typed.input.unwrap(), typed.result.unwrap()), identities);
}

/// What the parameter changes: the two source lengths of a withheld
/// input and the length of a withheld view. It changes nothing of a
/// shown identity, nothing of a result, and never brings a digest back.
#[test]
fn the_contracts_differ_in_the_withheld_lengths_alone() {
    let doc = masked(EMAIL);
    let graph = graph_bytes(&doc);
    let b = binding();
    let whole = input_identity(&doc, Some(&b));
    assert_eq!((whole.raw_bytes_len, whole.source_text_len), (EMAIL.len() as u64, EMAIL.len() as u32));

    for lengths in BOTH {
        assert_eq!(rule::input(&doc, Some(&b), true, lengths), whole, "{lengths:?}: a shown input is whole");
        assert_eq!(rule::identities(&doc, Some(&b), &graph, &Disclosure::FULL, lengths).0, whole);
        let withheld = rule::input(&doc, Some(&b), false, lengths);
        assert!(withheld.raw_bytes_digest.is_empty() && withheld.source_text_digest.is_empty(), "{lengths:?}");
        assert_eq!(withheld.binding, Some(b.clone()), "{lengths:?}: the binding is the caller's");
        assert!(rule::is_withheld(&withheld) && !rule::is_withheld(&whole));
        for disclosure in [REWRITTEN, Disclosure::FINDINGS] {
            assert_eq!(rule::identities(&doc, Some(&b), &graph, &disclosure, lengths).0, withheld, "{disclosure:?}");
        }
    }
    let kept = rule::input(&doc, Some(&b), false, WithheldLengths::Kept);
    let zeroed = rule::input(&doc, Some(&b), false, WithheldLengths::Zeroed);
    assert_eq!((kept.raw_bytes_len, kept.source_text_len), (whole.raw_bytes_len, whole.source_text_len));
    assert_eq!((zeroed.raw_bytes_len, zeroed.source_text_len), (0, 0));
    assert_eq!(InputIdentity { raw_bytes_len: 0, source_text_len: 0, ..kept.clone() }, zeroed);
    // The receipt of a run with no projection is under the kept contract.
    assert_eq!(input_identity_with(&doc, Some(&b), false), kept);
    assert_eq!(input_identity_with(&doc, Some(&b), true), whole);

    let source = doc.get::<View>().unwrap();
    let digest = ViewTextDigest::of_text(EMAIL).as_bytes().to_vec();
    for lengths in BOTH {
        assert_eq!(rule::view(source, true, lengths), (EMAIL.len() as u32, digest.clone()), "{lengths:?}");
    }
    assert_eq!(rule::view(source, false, WithheldLengths::Kept), (EMAIL.len() as u32, Vec::new()));
    assert_eq!(rule::view(source, false, WithheldLengths::Zeroed), (0, Vec::new()));
}

/// A withheld result is the whole one without the three fields that name
/// the graph, and carries its digest in no form.
#[test]
fn a_withheld_result_keeps_its_versions_and_nothing_of_the_graph() {
    let doc = masked(EMAIL);
    let graph = graph_bytes(&doc);
    let whole = result_identity(&graph);
    assert_eq!(rule::result(&graph, true), whole);
    assert_eq!(whole.content_digest, graph_content_digest(&graph).as_bytes());
    let withheld = rule::result(&graph, false);
    let stripped =
        ResultIdentity { result_id: String::new(), content_digest: Vec::new(), graph_bytes_len: 0, ..whole.clone() };
    assert_eq!(withheld, stripped);
    assert!(withheld.graph_format_version != 0 && withheld.envelope_schema_version != 0, "{withheld:?}");
    let bytes = withheld.encode_to_vec();
    assert!(!holds(&bytes, &whole.content_digest) && !holds(&bytes, whole.result_id.as_bytes()));
    for lengths in BOTH {
        assert_eq!(rule::identities(&doc, None, &graph, &REWRITTEN, lengths).1, withheld, "{lengths:?}");
        assert_eq!(rule::identities(&doc, None, &graph, &Disclosure::FULL, lengths).1, whole, "{lengths:?}");
    }
}

/// The reader takes every identity the writer makes, under the contract
/// it was made for, and says whether it withholds.
#[test]
fn the_reader_accepts_what_the_writer_makes() {
    let doc = masked(EMAIL);
    let graph = graph_bytes(&doc);
    for lengths in BOTH {
        for shown in [true, false] {
            let (input, result) = (rule::input(&doc, None, shown, lengths), rule::result(&graph, shown));
            let withheld = rule::check_input("at.", &input, lengths).unwrap();
            assert_eq!(withheld, !shown, "{lengths:?}");
            rule::check_result("at.", &result, withheld).unwrap();
        }
    }
    // An identity that zeroes its lengths also reads under the kept
    // contract; one that keeps them does not read as the search answer's.
    let kept = rule::input(&doc, None, false, WithheldLengths::Kept);
    let zeroed = rule::input(&doc, None, false, WithheldLengths::Zeroed);
    assert!(rule::check_input("", &zeroed, WithheldLengths::Kept).unwrap());
    let why = rule::check_input("at.", &kept, WithheldLengths::Zeroed).unwrap_err().message;
    assert!(why.starts_with("at.input gives the length of a text"), "{why}");
}

/// All or nothing: each field that names the source or the graph, put
/// back alone on a withheld identity or taken alone from a whole one, is
/// refused, and the refusal says where the identity is.
#[test]
fn the_reader_refuses_each_half_of_an_identity() {
    let doc = masked(EMAIL);
    let graph = graph_bytes(&doc);
    let (whole_in, whole_out) = (rule::input(&doc, None, true, WithheldLengths::Kept), rule::result(&graph, true));
    for lengths in BOTH {
        let (input, result) = (rule::input(&doc, None, false, lengths), rule::result(&graph, false));
        let refused_input = |change: &dyn Fn(&mut InputIdentity), on: &InputIdentity| {
            let mut i = on.clone();
            change(&mut i);
            rule::check_input("at.", &i, lengths).expect_err("half an input identity is whole").message
        };
        let refused_result = |change: &dyn Fn(&mut ResultIdentity), on: &ResultIdentity, withheld: bool| {
            let mut r = on.clone();
            change(&mut r);
            rule::check_result("at.", &r, withheld).expect_err("half a result identity is whole").message
        };
        for why in [
            refused_input(&|i| i.source_text_digest = whole_in.source_text_digest.clone(), &input),
            refused_input(&|i| i.raw_bytes_digest = whole_in.raw_bytes_digest.clone(), &input),
            refused_input(&|i| i.source_text_digest.clear(), &whole_in),
            refused_input(&|i| i.raw_bytes_digest.clear(), &whole_in),
        ] {
            assert!(why.starts_with("at.input") && why.contains("without the other"), "{lengths:?}: {why}");
        }
        for why in [
            refused_result(&|r| r.result_id = whole_out.result_id.clone(), &result, true),
            refused_result(&|r| r.graph_bytes_len = whole_out.graph_bytes_len, &result, true),
            // Without its digest a whole result still names the graph.
            refused_result(&|r| r.content_digest.clear(), &whole_out, false),
        ] {
            assert!(why.starts_with("at.result") && why.contains("no content_digest"), "{lengths:?}: {why}");
        }
        for why in [
            refused_result(&|r| r.content_digest = whole_out.content_digest.clone(), &result, true),
            refused_result(&|_| {}, &whole_out, true),
            refused_result(&|_| {}, &result, false),
        ] {
            assert!(why.contains("input and result digests"), "{lengths:?}: {why}");
        }
    }
    // A length of the withheld text is half an identity only where the
    // contract says the answer gives none.
    let zeroed = rule::input(&doc, None, false, WithheldLengths::Zeroed);
    for change in [(|i: &mut InputIdentity| i.source_text_len = 25) as fn(&mut _), |i| i.raw_bytes_len = 25] {
        let mut i = zeroed.clone();
        change(&mut i);
        let why = rule::check_input("at.", &i, WithheldLengths::Zeroed).unwrap_err().message;
        assert!(why.contains("length of a text"), "{why}");
        assert!(rule::check_input("at.", &i, WithheldLengths::Kept).unwrap());
    }
}
