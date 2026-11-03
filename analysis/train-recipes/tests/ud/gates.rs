use super::*;

/// More held-out sentences skipped than `MAX_SKIPPED_SENTENCES` allows
/// fail the check the binary runs before training, and the run.
#[test]
fn skipped_sentences_beyond_the_limit_are_refused_before_training() {
    let dir = temp_dir("skipped");
    let all: Vec<usize> = (0..POOL.len()).collect();
    let empty = "# sent_id = empty\n# text = _\n1.1\t_\t_\tPRON\t_\t_\t_\t_\t_\t_\n\n";
    let test = format!("{}{empty}", conllu(&all, 100, "test"));
    let treebank = write_treebank(&dir, &conllu(&all, 100, "train"), &test);
    let text = treebank.read(&dir).unwrap();
    for component in [Component::Pos, Component::Lemmas] {
        let r = recipe(treebank, component, spanish(component));
        let held_out_skipped = |result: Result<(), Error>| {
            matches!(result, Err(Error::TooManyLeftOut { what: LeftOut::HeldOutSentences, count: 1, of: 101, .. }))
        };
        assert!(held_out_skipped(r.check_data(&text, b"not read")));
        assert!(held_out_skipped(r.held_out(&text).map(|_| ())));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_treebank_whose_files_or_licence_changed_is_refused() {
    let dir = temp_dir("changed");
    let treebank = write_treebank(&dir, &conllu(&[0], 3, "train"), &conllu(&[1], 3, "test"));
    treebank.read(&dir).unwrap();
    std::fs::write(dir.join("es_test-ud-test.conllu"), conllu(&[2], 3, "test")).unwrap();
    assert!(matches!(treebank.read(&dir), Err(Error::TreebankFile { file: "es_test-ud-test.conllu", .. })));
    std::fs::write(dir.join("LICENSE.txt"), LICENCE.replace("Attribution 4.0", "Attribution-ShareAlike 4.0")).unwrap();
    assert!(matches!(treebank.read(&dir), Err(Error::Licence { .. })));
    std::fs::remove_dir_all(dir).unwrap();
}
