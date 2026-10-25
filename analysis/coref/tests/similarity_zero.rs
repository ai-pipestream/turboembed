//! Public word-vector scores retain the positive-zero identity of coref's
//! original accumulator, including ties ordered with `total_cmp`.

use turbo_coref::WordVectors;
use turbo_coref::word2vec::{read_text, similarity};

#[test]
fn orthogonal_signed_zero_vectors_keep_positive_zero_scores_and_tie_order() {
    let vectors = read_text("3 2\nnorth 1 -0\neast -0 1\npositive 0 1\n").unwrap();
    let identity = vectors.config_hash();
    let mut scores: Vec<_> = ["east", "positive"]
        .iter()
        .enumerate()
        .map(|(row, word)| (row, similarity(&vectors, "north", word).unwrap()))
        .collect();
    scores.sort_by(|(a, x), (b, y)| y.total_cmp(x).then(a.cmp(b)));
    assert_eq!(scores.iter().map(|(row, _)| *row).collect::<Vec<_>>(), [0, 1]);
    for (_, score) in scores {
        assert_eq!(score.to_bits(), 0.0f64.to_bits());
    }
    assert_eq!(vectors.config_hash(), identity);
}
