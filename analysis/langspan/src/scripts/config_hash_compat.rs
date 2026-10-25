use super::Candidates;

#[test]
fn retained_candidate_bytes_keep_their_hashes() {
    let codes = ["eng", "fra", "ell", "rus", "srp", "jpn", "xyz"];
    let mut hashes = vec![
        Candidates::new(codes.iter().copied(), |_| true).config_hash(),
        Candidates::new(codes.iter().copied(), |_| false).config_hash(),
        Candidates::new(std::iter::empty(), |_| true).config_hash(),
        Candidates::new(codes.iter().rev().copied(), |_| true).config_hash(),
    ];
    for selected in codes {
        hashes.push(Candidates::new(codes.iter().copied(), |code| code == selected).config_hash());
    }
    assert_eq!(
        hashes,
        [
            11103668656877411099,
            12161962213042174405,
            12161962213042174405,
            1633572796449162277,
            9421812216203707516,
            11654127623171296925,
            5679016186118057836,
            14770056880263335515,
            1791444276733072461,
            17669477107711791338,
            17687766317047062866
        ]
    );
}
