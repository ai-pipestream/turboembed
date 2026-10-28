mod common;

use common::{SENTENCES, temp_dir, write_document};
use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::namefind::selection::{Clearing, GRID, RECIPES, select, train_selected};

#[test]
fn fixed_grid_uses_only_training_documents_and_is_repeatable() {
    let root = temp_dir("name-selection");
    for i in 0..10 {
        write_document(&root.join("written/news"), &format!("doc{i:02}"), SENTENCES, &[0, 1, 2, 3, 4, 5, 6, 7]);
    }
    let corpus = MascCorpus::read(&root).unwrap();
    let recipe = RECIPES[0];
    let selection = select(&recipe, &corpus).unwrap();
    assert_eq!(selection, select(&recipe, &corpus).unwrap());
    assert_eq!(selection.training_documents, 10);
    assert_eq!(selection.folds, 10);
    assert_eq!(selection.candidates.len(), 4);
    assert_eq!(selection.candidates.iter().map(|c| c.setting).collect::<Vec<_>>(), GRID);
    assert_eq!(
        GRID.iter().map(|s| (s.iterations, s.cutoff)).collect::<Vec<_>>(),
        [(300, 0), (300, 0), (100, 5), (100, 5)]
    );
    assert_eq!(GRID[0].clearing, Clearing::Document);
    assert_eq!(GRID[1].clearing, Clearing::Sentence);
    for candidate in &selection.candidates {
        assert_eq!(candidate.folds.len(), 10);
        let mut seen = [0; 10];
        for fold in &candidate.folds {
            assert_eq!(fold.training_document_indices.len(), 9);
            assert_eq!(fold.test_document_indices.len(), 1);
            for &index in &fold.test_document_indices {
                assert!(!fold.training_document_indices.contains(&index));
                seen[index] += 1;
            }
        }
        assert_eq!(seen, [1; 10]);
        assert_eq!(candidate.merged.reference, candidate.folds.iter().map(|f| f.scores.reference).sum::<u64>());
        assert_eq!(candidate.merged.correct, candidate.folds.iter().map(|f| f.scores.correct).sum::<u64>());
    }
    let one = train_selected(&recipe, &corpus, selection.clone()).unwrap();
    let two = train_selected(&recipe, &corpus, selection.clone()).unwrap();
    assert_eq!(one.model_bytes(), two.model_bytes());
    let mut bad = selection.clone();
    bad.training_samples_sha256 = "0".repeat(64);
    assert!(train_selected(&recipe, &corpus, bad).is_err());
    let mut bad = selection.clone();
    bad.selected = usize::MAX;
    assert!(train_selected(&recipe, &corpus, bad).is_err());
    let mut bad = selection;
    bad.candidates[0].setting.cutoff = 1;
    assert!(train_selected(&recipe, &corpus, bad).is_err());
    assert!(select(&turbo_train_recipes::namefind::PERSON, &corpus).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn fewer_than_ten_documents_cannot_run_registered_selection() {
    let root = temp_dir("name-selection-small");
    write_document(&root.join("written/news"), "a", SENTENCES, &[0, 1, 2, 3]);
    let corpus = MascCorpus::read(&root).unwrap();
    let error = select(&RECIPES[0], &corpus).unwrap_err();
    assert!(error.to_string().contains("ten nonempty whole documents"));
    std::fs::remove_dir_all(root).unwrap();
}
