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
fn train_and_gate_roundtrip_both_parser_kinds() {
    let sent1 = "1\tEl\tel\tDET\t_\t_\t2\tdet\t_\t_\n2\tgato\tgato\tNOUN\t_\t_\t3\tnsubj\t_\t_\n3\tduerme\tdormir\tVERB\t_\t_\t0\troot\t_\t_\n\n";
    let sent2 = "1\tLa\tel\tDET\t_\t_\t2\tdet\t_\t_\n2\tcasa\tcasa\tNOUN\t_\t_\t3\tnsubj\t_\t_\n3\tcae\tcaer\tVERB\t_\t_\t0\troot\t_\t_\n\n";
    let conllu = format!("{}{}", sent1.repeat(16), sent2.repeat(16));
    let held_out = read_samples(&conllu, "UD_Spanish-AnCora", "es_ancora-ud-test.conllu").unwrap();
    for r in &DEPPARSE_RECIPES[..2] {
        let trained = r.train(&conllu).unwrap();
        assert!(!trained.bytes.is_empty());
        assert!(!trained.reference_bytes.is_empty());
        let (gate, uas_gate, card) = r.gate(&trained.bytes, &trained.reference_bytes, &held_out).unwrap();
        assert_eq!(card.samples, 32);
        assert_eq!(gate.candidate.id, r.model_id);
        assert_eq!(uas_gate.candidate.id, r.model_id);
    }
}
