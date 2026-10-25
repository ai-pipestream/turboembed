//! The finder on `dslim/bert-base-NER` with rows shorter than its
//! sentences: an entity whose pieces fall in two rows is the entity the
//! whole sentence in one row gives, however the rows are cut.

use std::path::PathBuf;

use turbo_analysis_embed::EncoderOptions;
use turbo_namefind_dl::{NameFinderDl, NameFinderDlOptions};

const TEXT: &str =
    "Angela Merkel met Emmanuel Macron and Kristalina Georgieva in Paris to discuss the European Union budget.";

#[test]
#[ignore = "needs TURBO_NER_BUNDLE naming a dslim/bert-base-NER bundle (bundle/recipes/dslim-bert-base-ner.json)"]
fn an_entity_across_a_row_boundary_is_whole() {
    let dir = PathBuf::from(std::env::var("TURBO_NER_BUNDLE").expect("TURBO_NER_BUNDLE"));
    let find = |max_seq: u32| {
        let encoder = EncoderOptions { max_seq, ..EncoderOptions::default() };
        let f = NameFinderDl::open(&dir, &NameFinderDlOptions { encoder, ..NameFinderDlOptions::default() }).unwrap();
        let found = f.find(&[(TEXT, vec![(0, TEXT.len())])]).unwrap().remove(0);
        found
            .iter()
            .map(|c| (&TEXT[c.start..c.end], f.kinds().label(turbo_analysis_layers::Tag(c.kind as u16)).to_owned()))
            .collect::<Vec<_>>()
    };
    let whole = find(0);
    eprintln!("{whole:?}");
    assert!(whole.iter().any(|(t, k)| *t == "Kristalina Georgieva" && k == "PER"), "{whole:?}");
    assert!(whole.iter().any(|(t, k)| *t == "European Union" && k == "ORG"), "{whole:?}");
    // Rows of 10 to 23 pieces, read overlapping by half a row, cut every
    // name of more than one piece somewhere.
    for max_seq in 12..=25 {
        assert_eq!(find(max_seq), whole, "max_seq {max_seq}");
    }
}
