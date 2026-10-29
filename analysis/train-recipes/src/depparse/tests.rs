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

#[test]
fn train_and_gate_refuse_retired_arc_standard_parser() {
    let conllu = "1\ta\ta\tNOUN\t_\t_\t0\troot\t_\t_\n\n";
    let held_out = read_samples(conllu, "UD_Spanish-AnCora", "es_ancora-ud-test.conllu").unwrap();
    for r in DEPPARSE_RECIPES {
        let train_err = r.train(conllu).unwrap_err();
        assert!(train_err.to_string().contains("retired (#1529)"), "{train_err}");
        let gate_err = r.gate(&[], &[], &held_out).unwrap_err();
        assert!(gate_err.to_string().contains("retired (#1529)"), "{gate_err}");
    }
}
