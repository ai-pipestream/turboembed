//! Added tokens as upstream `tokenizers` matches them in the text
//! (AddedVocabulary): leftmost-longest, the raw ones in the text as it
//! is, the normalized ones in each piece the raw ones leave, normalized.

use super::*;

/// A token matched in raw text, or in normalized text.
pub(crate) struct Special {
    pub(crate) content: String,
    pub(crate) id: i32,
    /// The whitespace before it is part of the match.
    pub(crate) lstrip: bool,
    /// The whitespace after it is part of the match.
    pub(crate) rstrip: bool,
    /// Matched only with no word character on either side.
    pub(crate) single_word: bool,
}

/// What `Added::split` finds in a text, in order.
pub(crate) enum Piece<'a> {
    /// Bytes with no token in them.
    Text(Range<usize>),
    /// A token, its content at this byte.
    Token(&'a Special, usize),
}

const NO: u32 = u32::MAX;

/// The tokens to match, and a byte trie of their contents.
pub(crate) struct Added {
    /// Longest first.
    tokens: Vec<Special>,
    /// The node after each first byte, or NO.
    root: Box<[u32; 256]>,
    nodes: Vec<Node>,
}

struct Node {
    /// (byte, node), few per node.
    edges: Vec<(u8, u32)>,
    /// The token whose content ends here, or NO.
    token: u32,
}

impl Added {
    /// The trie of `tokens`, each content non-empty; two with one content
    /// are refused, as upstream's matcher refuses them.
    pub(super) fn new(mut tokens: Vec<Special>) -> std::result::Result<Added, String> {
        tokens.sort_by_key(|s| std::cmp::Reverse(s.content.len()));
        let mut root = Box::new([NO; 256]);
        let mut nodes: Vec<Node> = Vec::new();
        let node = |nodes: &mut Vec<Node>| {
            nodes.push(Node { edges: Vec::new(), token: NO });
            nodes.len() as u32 - 1
        };
        for (i, s) in tokens.iter().enumerate() {
            let bytes = s.content.as_bytes();
            let Some(&first) = bytes.first() else { return Err("an empty token".into()) };
            let mut at = root[first as usize];
            if at == NO {
                at = node(&mut nodes);
                root[first as usize] = at;
            }
            for &b in &bytes[1..] {
                at = match nodes[at as usize].edges.iter().find(|e| e.0 == b) {
                    Some(e) => e.1,
                    None => {
                        let next = node(&mut nodes);
                        nodes[at as usize].edges.push((b, next));
                        next
                    }
                };
            }
            if nodes[at as usize].token != NO {
                return Err(format!("{:?} twice", s.content));
            }
            nodes[at as usize].token = i as u32;
        }
        Ok(Added { tokens, root, nodes })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The tokens, longest first.
    pub(crate) fn tokens(&self) -> &[Special] {
        &self.tokens
    }

    /// The longest token that starts at `s[p]`.
    fn longest(&self, s: &[u8], p: usize) -> Option<&Special> {
        let mut at = self.root[s[p] as usize];
        let mut best = NO;
        let mut i = p + 1;
        while at != NO {
            let n = &self.nodes[at as usize];
            if n.token != NO {
                best = n.token;
            }
            let Some(&b) = s.get(i) else { break };
            at = n.edges.iter().find(|e| e.0 == b).map_or(NO, |e| e.1);
            i += 1;
        }
        self.tokens.get(best as usize)
    }

    /// `s` cut at its tokens, as upstream's `find_matches` cuts it: the
    /// leftmost-longest match, the search going on after it; one that is
    /// not a whole word when `single_word` is passed over, its bytes left
    /// to the text; `lstrip` takes the whitespace before the match back
    /// to the last cut, `rstrip` the whitespace after it. The search goes
    /// on from the end of the match itself, so a token can start inside
    /// the whitespace one before it took, as upstream's does.
    pub(crate) fn split<'a>(&'a self, s: &str, mut f: impl FnMut(Piece<'a>)) {
        let b = s.as_bytes();
        let (mut pos, mut done) = (0, 0);
        while pos < b.len() {
            // A byte a token starts with is never inside a character.
            let Some(k) = b[pos..].iter().position(|&c| self.root[c as usize] != NO) else { break };
            let p = pos + k;
            let Some(t) = self.longest(b, p) else {
                pos = p + 1;
                continue;
            };
            let mut stop = p + t.content.len();
            pos = stop;
            if t.single_word
                && (s[..p].chars().next_back().is_some_and(is_word) || s[stop..].chars().next().is_some_and(is_word))
            {
                continue;
            }
            let mut start = p;
            if t.lstrip {
                start = s[..p].trim_end().len().max(done);
            }
            if t.rstrip {
                stop = b.len() - s[stop..].trim_start().len();
            }
            if done < start {
                f(Piece::Text(done..start));
            }
            f(Piece::Token(t, p));
            done = stop;
        }
        if done < b.len() {
            f(Piece::Text(done..b.len()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn added(tokens: &[(&str, bool, bool, bool)]) -> Added {
        let tokens = tokens.iter().enumerate().map(|(i, &(c, lstrip, rstrip, single_word))| Special {
            content: c.into(),
            id: i as i32,
            lstrip,
            rstrip,
            single_word,
        });
        Added::new(tokens.collect()).unwrap()
    }

    /// The pieces of `s`: text as `"text"`, a token as `#id`.
    fn cut(a: &Added, s: &str) -> Vec<String> {
        let mut out = Vec::new();
        a.split(s, |p| match p {
            Piece::Text(r) => out.push(format!("{:?}", &s[r])),
            Piece::Token(t, at) => {
                assert_eq!(&s[at..at + t.content.len()], t.content);
                out.push(format!("#{}", t.id))
            }
        });
        out
    }

    #[test]
    fn the_longest_token_at_the_leftmost_start_wins() {
        let a = added(&[("  ", false, false, false), ("   ", false, false, false), ("    ", false, false, false)]);
        assert_eq!(cut(&a, "a b"), ["\"a b\""]);
        assert_eq!(cut(&a, "a  b"), ["\"a\"", "#0", "\"b\""]);
        assert_eq!(cut(&a, "a     b"), ["\"a\"", "#2", "\" b\""]);
        assert_eq!(cut(&a, "      "), ["#2", "#0"]);
    }

    #[test]
    fn unused_tokens_do_not_take_a_shorter_prefix() {
        let a = added(&[("[unused1]", false, false, false), ("[unused10]", false, false, false)]);
        assert_eq!(cut(&a, "[unused10][unused1]x[unused1"), ["#1", "#0", "\"x[unused1\""]);
    }

    #[test]
    fn lstrip_takes_the_whitespace_before_back_to_the_last_cut() {
        let a = added(&[("[MASK]", true, false, false), ("<|padding|>", false, false, false)]);
        assert_eq!(cut(&a, "a  [MASK] b"), ["\"a\"", "#0", "\" b\""]);
        assert_eq!(cut(&a, "<|padding|>  [MASK]"), ["#1", "#0"]);
        assert_eq!(cut(&a, " \u{3000}[MASK]"), ["#0"]);
    }

    #[test]
    fn rstrip_takes_the_whitespace_after_and_the_search_goes_on_inside_it() {
        let a = added(&[("<x>", false, true, false), ("  ", false, false, false)]);
        assert_eq!(cut(&a, "<x>   y"), ["#0", "#1", "\" y\""]);
        assert_eq!(cut(&a, "<x> y"), ["#0", "\"y\""]);
    }

    #[test]
    fn single_word_passes_over_a_match_inside_a_word() {
        let a = added(&[("pad", false, false, true)]);
        assert_eq!(cut(&a, "pad padded _pad pad."), ["#0", "\" padded _pad \"", "#0", "\".\""]);
        assert_eq!(cut(&a, "épad"), ["\"épad\""]);
    }

    #[test]
    fn a_content_twice_is_refused() {
        let twice =
            ["x", "x"].map(|c| Special { content: c.into(), id: 0, lstrip: false, rstrip: false, single_word: false });
        assert!(Added::new(twice.into()).is_err());
    }
}
