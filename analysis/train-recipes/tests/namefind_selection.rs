mod common;

use common::{SENTENCES, temp_dir, write_document};
use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::namefind::selection::{RECIPES, select};

#[test]
fn fewer_than_ten_documents_cannot_run_registered_selection() {
    let root = temp_dir("name-selection-small");
    write_document(&root.join("written/news"), "a", SENTENCES, &[0, 1, 2, 3]);
    let corpus = MascCorpus::read(&root).unwrap();
    let error = select(&RECIPES[0], &corpus).unwrap_err();
    assert!(error.to_string().contains("ten nonempty whole documents"));
    std::fs::remove_dir_all(root).unwrap();
}
