use super::*;

#[test]
fn recipes_cover_all_five_treebanks_for_both_parsers() {
    for r in DEPPARSE_RECIPES {
        let t = r.treebank;
        let expected = format!("turbo-{}-ud-{}-2.16-depparse-{}-1", t.language, t.name, r.kind.name());
        assert_eq!(r.model_id, expected);
        let parsed: turbo_zoo::ArtifactId = r.model_id.parse().unwrap();
        assert_eq!(parsed.as_str(), r.model_id);
        assert_eq!(DepparseRecipe::trains(r.model_id), Some(r));
    }
    assert_eq!(DepparseRecipe::trains("turbo-en-ud-ewt-2.16-depparse-maxent-1"), None);
}
