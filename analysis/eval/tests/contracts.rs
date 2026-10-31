//! Synthetic contracts test grouping, evidence semantics and Unicode offsets.
mod common;
use serde_json::{Value, json};
use turbo_eval::read::{contract_nli, cuad};

fn cuad_data() -> Value {
    json!({"data": (["a", "b"].map(|id| json!({"title": id, "paragraphs": [{
        "context": "é😀 paid", "qas": [{"id": format!("{id}__Payment"), "question": "Find payment",
        "is_impossible": false, "answers": [{"text": "paid", "answer_start": 3}]},
        {"id": format!("{id}__Absent"), "question": "Find absence", "is_impossible": true, "answers": []}]
    }]})))})
}
fn nli_data() -> Value {
    json!({"labels": {"h": {"hypothesis": "Payment is required."}},
    "documents": ([(1, "Entailment"), (2, "Contradiction"), (3, "NotMentioned")].map(|(id, choice)|
        json!({"id": id, "text": "é😀 paid", "spans": [[3, 7]], "annotation_sets": [{"annotations": {
            "h": {"choice": choice, "spans": if id == 3 {vec![]} else {vec![0]}}
        }}]})))})
}
fn read_c(v: &Value) -> Result<turbo_eval::set::LoadedSet, String> {
    cuad::read_json("cuad", "fixture", &v.to_string(), String::new())
}
fn read_n(v: &Value) -> Result<turbo_eval::set::LoadedSet, String> {
    contract_nli::read_json("contract-nli", "fixture", &v.to_string(), String::new())
}
#[test]
fn repeated_questions_are_one_query_with_multiple_source_contracts() {
    let s = read_c(&cuad_data()).unwrap();
    assert_eq!(s.docs.len(), 2);
    assert_eq!(s.queries.len(), 2);
    for id in ["a", "b"] {
        assert_eq!(s.qrels.grade("Payment", id), 1);
        assert_eq!(s.qrels.grade("Absent", id), 0);
        assert_eq!(common::quoted(&s, "Payment", id), ["paid"]);
        assert_eq!(common::spans(&s, "Payment", id), [(7, 11)]);
    }
}
#[test]
fn nli_keeps_both_polarities_and_does_not_duplicate_hypotheses() {
    let s = read_n(&nli_data()).unwrap();
    assert_eq!(s.queries.len(), 1);
    assert_eq!(s.docs.len(), 3);
    for id in ["1", "2"] {
        assert_eq!(s.qrels.grade("h", id), 1);
        assert_eq!(common::quoted(&s, "h", id), ["paid"]);
    }
    assert_eq!(s.qrels.grade("h", "3"), 0);
    assert!(common::spans(&s, "h", "3").is_empty());
}
#[test]
fn cuad_rejects_ambiguous_templates_and_wrong_answer_offsets() {
    let mut v = cuad_data();
    v["data"][1]["paragraphs"][0]["qas"][0]["question"] = json!("Other text");
    assert!(read_c(&v).unwrap_err().contains("different question text"));
    let mut v = cuad_data();
    v["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] = json!(2);
    assert!(read_c(&v).unwrap_err().contains("answer text mismatch"));
}
#[test]
fn nli_rejects_invalid_annotation_keys_indices_and_neutral_evidence() {
    let mut v = nli_data();
    v["documents"][0]["annotation_sets"][0]["annotations"]["h"]["spans"] = json!([1]);
    assert!(read_n(&v).unwrap_err().contains("invalid evidence index"));
    let mut v = nli_data();
    v["documents"][0]["annotation_sets"][0]["annotations"]["unknown"] = json!({"choice": "NotMentioned", "spans": []});
    assert!(read_n(&v).unwrap_err().contains("different hypothesis keys"));
    let mut v = nli_data();
    v["documents"][2]["annotation_sets"][0]["annotations"]["h"]["spans"] = json!([0]);
    assert!(read_n(&v).unwrap_err().contains("NotMentioned has evidence"));
}
#[test]
fn manually_supplied_files_still_enforce_pinned_digest() {
    let dir = common::scratch("contracts-pins");
    std::fs::write(dir.join("test.json"), "{}").unwrap();
    for reader in [cuad::read_in, contract_nli::read_in] {
        let error = reader("fixture", turbo_eval::read::SetDir::new(&dir)).unwrap_err();
        assert!(error.contains("SHA-256"), "{error}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}
