//! Line reading shared by the two model files.

/// The lines of `text` numbered from 1, without their `\n` or `\r\n`. A
/// final line end does not start another line.
pub(crate) fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let body = text.strip_suffix('\n').unwrap_or(text);
    let empty = text.is_empty();
    body.split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .enumerate()
        .map(|(i, l)| (i + 1, l))
        .filter(move |_| !empty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_numbered_without_their_ends() {
        assert_eq!(lines("a\nb\r\nc\n").collect::<Vec<_>>(), [(1, "a"), (2, "b"), (3, "c")]);
        assert_eq!(lines("a\n\nb").collect::<Vec<_>>(), [(1, "a"), (2, ""), (3, "b")]);
        assert_eq!(lines("").count(), 0);
        assert_eq!(lines("\n").collect::<Vec<_>>(), [(1, "")]);
    }
}
