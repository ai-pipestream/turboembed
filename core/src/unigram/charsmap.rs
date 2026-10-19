use super::*;
/// SentencePiece's precompiled character map: a double-array trie over
/// UTF-8 bytes whose leaves index a string of replacements, each ended
/// by a NUL. Applied as upstream does: to each grapheme cluster under six
/// bytes as a whole, else to its characters one by one.
pub(super) struct Charsmap {
    trie: Vec<u32>,
    normalized: Vec<u8>,
    /// What the map makes of each ASCII character alone.
    ascii: Vec<Option<Box<str>>>,
}

impl Charsmap {
    pub(super) fn parse(file: &str, b64: &str) -> Result<Charsmap> {
        let bad = |why: &str| invalid(format!("{file}: normalizer.precompiled_charsmap: {why}"));
        let bytes = base64(b64).ok_or_else(|| bad("not base64"))?;
        if bytes.len() < 4 {
            return Err(bad("shorter than its length field"));
        }
        let trie_bytes = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if !trie_bytes.is_multiple_of(4) || trie_bytes > bytes.len() - 4 {
            return Err(bad("trie length is not inside the map"));
        }
        let trie: Vec<u32> =
            bytes[4..4 + trie_bytes].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
        if trie.is_empty() {
            return Err(bad("empty trie"));
        }
        let normalized = bytes[4 + trie_bytes..].to_vec();
        let replacement_text =
            std::str::from_utf8(&normalized).map_err(|_| bad("replacement strings are not UTF-8"))?;
        if !normalized.is_empty() && normalized.last() != Some(&0) {
            return Err(bad("replacement string is not NUL terminated"));
        }
        for &unit in &trie {
            if unit & (1 << 31) != 0 {
                let at = (unit & ((1 << 31) - 1)) as usize;
                if at >= normalized.len() || !replacement_text.is_char_boundary(at) {
                    return Err(bad("leaf points outside a replacement string boundary"));
                }
            }
        }
        let mut map = Charsmap { trie, normalized, ascii: Vec::new() };
        map.ascii = (0..128u8)
            .map(|b| map.transform(&[b]).map(|n| String::from_utf8_lossy(n).into_owned().into_boxed_str()))
            .collect();
        Ok(map)
    }

    /// The replacement for the longest prefix of `key` in the map: the
    /// first leaf met on the way, as upstream takes it.
    fn transform(&self, key: &[u8]) -> Option<&[u8]> {
        let unit = |pos: usize| self.trie.get(pos).copied();
        let offset = |u: u32| ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize;
        let mut pos = 0usize;
        let mut u = unit(pos)?;
        pos ^= offset(u);
        for &c in key {
            if c == 0 {
                break;
            }
            pos ^= c as usize;
            u = unit(pos)?;
            if u & ((1 << 31) | 0xFF) != c as u32 {
                return None;
            }
            pos ^= offset(u);
            if (u >> 8) & 1 == 1 {
                let at = (unit(pos)? & ((1 << 31) - 1)) as usize;
                let rest = self.normalized.get(at..)?;
                let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
                return Some(&rest[..end]);
            }
        }
        None
    }

    pub(super) fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let b = text.as_bytes();
        let mut i = 0;
        while i < b.len() {
            // An ASCII character followed by another (CR LF aside) is a
            // grapheme cluster of its own: the table's.
            if b[i].is_ascii() && b.get(i + 1).is_none_or(|n| n.is_ascii() && !(b[i] == b'\r' && *n == b'\n')) {
                match &self.ascii[b[i] as usize] {
                    Some(n) => out.push_str(n),
                    None => out.push(b[i] as char),
                }
                i += 1;
                continue;
            }
            // Otherwise, up to and including the next ASCII character that
            // another ASCII character follows (a cluster boundary), as
            // clusters.
            let mut end = i + 1;
            while end < b.len()
                && !(b[end - 1].is_ascii() && b[end].is_ascii() && !(b[end - 1] == b'\r' && b[end] == b'\n'))
            {
                end += 1;
            }
            self.clusters(&text[i..end], &mut out);
            i = end;
        }
        out
    }

    /// normalize(), each character beside its source: the span in the
    /// caller's text of the character, or of the grapheme cluster the
    /// map replaces as a whole, it came from; `base` is where `text`
    /// starts there.
    pub(super) fn normalize_aligned(&self, text: &str, base: usize, out: &mut Vec<(char, Span)>) {
        let b = text.as_bytes();
        let span = |from: usize, to: usize| [(base + from) as u32, (base + to) as u32];
        let mut i = 0;
        while i < b.len() {
            if b[i].is_ascii() && b.get(i + 1).is_none_or(|n| n.is_ascii() && !(b[i] == b'\r' && *n == b'\n')) {
                let at = span(i, i + 1);
                match &self.ascii[b[i] as usize] {
                    Some(n) => out.extend(n.chars().map(|c| (c, at))),
                    None => out.push((b[i] as char, at)),
                }
                i += 1;
                continue;
            }
            let mut end = i + 1;
            while end < b.len()
                && !(b[end - 1].is_ascii() && b[end].is_ascii() && !(b[end - 1] == b'\r' && b[end] == b'\n'))
            {
                end += 1;
            }
            let run = &text[i..end];
            for (g0, grapheme) in run.grapheme_indices(true) {
                let g0 = i + g0;
                if grapheme.len() < 6
                    && let Some(n) = self.transform(grapheme.as_bytes())
                {
                    let at = span(g0, g0 + grapheme.len());
                    out.extend(String::from_utf8_lossy(n).chars().map(|c| (c, at)));
                    continue;
                }
                for (c0, c) in grapheme.char_indices() {
                    let at = span(g0 + c0, g0 + c0 + c.len_utf8());
                    let mut buf = [0u8; 4];
                    match self.transform(c.encode_utf8(&mut buf).as_bytes()) {
                        Some(n) => out.extend(String::from_utf8_lossy(n).chars().map(|c| (c, at))),
                        None => out.push((c, at)),
                    }
                }
            }
            i = end;
        }
    }

    /// `text`'s grapheme clusters through the map.
    pub(super) fn clusters(&self, text: &str, out: &mut String) {
        let push = |out: &mut String, bytes: &[u8]| out.push_str(&String::from_utf8_lossy(bytes));
        for grapheme in text.graphemes(true) {
            if grapheme.len() < 6
                && let Some(n) = self.transform(grapheme.as_bytes())
            {
                push(out, n);
                continue;
            }
            for c in grapheme.chars() {
                let mut buf = [0u8; 4];
                match self.transform(c.encode_utf8(&mut buf).as_bytes()) {
                    Some(n) => push(out, n),
                    None => out.push(c),
                }
            }
        }
    }
}
