pub(crate) fn normalize(word: &str) -> String {
    if matches!(word, "*UNK*" | "*NULL*" | "*ROOT*") {
        return word.into();
    }
    turbo_unicode::java::lowercase_code_points(word)
}
