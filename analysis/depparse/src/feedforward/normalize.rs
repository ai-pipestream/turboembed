use turbo_unicode::{java, props};
pub(super) fn normalize(word: &str) -> String {
    if matches!(word, "*UNK*" | "*NULL*" | "*ROOT*") {
        return word.into();
    }
    word.char_indices()
        .map(|(at, c)| {
            if c == 'Σ' {
                let before = word[..at].chars().rev().find(|&c| !props::case_ignorable(c));
                let after = word[at + c.len_utf8()..].chars().find(|&c| !props::case_ignorable(c));
                if before.is_some_and(props::cased) && !after.is_some_and(props::cased) {
                    return 'ς';
                }
            }
            java::to_lower_char(c)
        })
        .collect()
}
