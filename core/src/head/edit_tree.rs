//! Edit trees (Chrupała 2008; Müller et al. 2015): how a word form
//! rewrites into its lemma, independent of the word's stem, so one tree
//! serves every word inflected the same way. A lemmatizer head classifies
//! each word into one tree and applies it.
//!
//! Built as spaCy's edit-tree lemmatizer builds them (MIT, read at v3.7.5,
//! `spacy/pipeline/_edit_tree_internals/edit_trees.pyx`): the longest
//! common substring of form and lemma is kept (a match node) and the parts
//! before and after it rewrite by their own trees; with nothing in common
//! the whole part is substituted. Lengths count Unicode scalar values.
//!
//! A tree is written as its label, an S-expression: `(m P S X Y)` keeps all
//! but the first P and last S characters and rewrites those by X and Y
//! (`()` for none); `(s 'from' 'to')` substitutes, with `\'` and `\\`
//! escaping inside quotes. [`EditTree::label`] and [`EditTree::parse`] are
//! inverses.

use crate::status::{Result, invalid};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EditTree {
    /// Keep the characters between the first `prefix` and the last
    /// `suffix`, rewriting those by their trees (none: they must be empty).
    Match { prefix: u32, suffix: u32, before: Option<Box<EditTree>>, after: Option<Box<EditTree>> },
    /// Replace exactly `from` by `to`.
    Subst { from: String, to: String },
}

/// The longest common substring of `a` and `b` as (a start, b start,
/// length): the first one found scanning `a` then `b`, as spaCy finds it.
fn longest_common(a: &[char], b: &[char]) -> (usize, usize, usize) {
    let (mut prev, mut cur) = (vec![0usize; b.len()], vec![0usize; b.len()]);
    let mut best = (0, 0, 0);
    for (i, ca) in a.iter().enumerate() {
        for (j, cb) in b.iter().enumerate() {
            cur[j] = if ca != cb {
                0
            } else if i == 0 || j == 0 {
                1
            } else {
                prev[j - 1] + 1
            };
            if cur[j] > best.2 {
                best = (i + 1 - cur[j], j + 1 - cur[j], cur[j]);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    best
}

fn build(form: &[char], lemma: &[char]) -> EditTree {
    let (a, b, len) = longest_common(form, lemma);
    if len == 0 {
        return EditTree::Subst { from: form.iter().collect(), to: lemma.iter().collect() };
    }
    let before = (a != 0 || b != 0).then(|| Box::new(build(&form[..a], &lemma[..b])));
    let (fa, lb) = (a + len, b + len);
    let after = (fa != form.len() || lb != lemma.len()).then(|| Box::new(build(&form[fa..], &lemma[lb..])));
    EditTree::Match { prefix: a as u32, suffix: (form.len() - fa) as u32, before, after }
}

fn quote(s: &str, out: &mut String) {
    out.push('\'');
    for c in s.chars() {
        if c == '\'' || c == '\\' {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('\'');
}

impl EditTree {
    /// The tree that rewrites `form` into `lemma`. Two empty strings give
    /// the empty match, as spaCy has it.
    pub fn mine(form: &str, lemma: &str) -> EditTree {
        if form.is_empty() && lemma.is_empty() {
            return EditTree::Match { prefix: 0, suffix: 0, before: None, after: None };
        }
        let (f, l): (Vec<char>, Vec<char>) = (form.chars().collect(), lemma.chars().collect());
        build(&f, &l)
    }

    /// The lemma of `form` by this tree, or None when the tree does not
    /// fit the form.
    pub fn apply(&self, form: &str) -> Option<String> {
        let chars: Vec<char> = form.chars().collect();
        let mut out = String::with_capacity(form.len() + 4);
        self.apply_to(&chars, &mut out).then_some(out)
    }

    fn apply_to(&self, part: &[char], out: &mut String) -> bool {
        match self {
            EditTree::Subst { from, to } => {
                let fits = from.chars().eq(part.iter().copied());
                if fits {
                    out.push_str(to);
                }
                fits
            }
            EditTree::Match { prefix, suffix, before, after } => {
                let (p, s) = (*prefix as usize, *suffix as usize);
                if p + s > part.len() {
                    return false;
                }
                let end = part.len() - s;
                let ok = match before {
                    Some(t) => t.apply_to(&part[..p], out),
                    None => p == 0,
                };
                if !ok {
                    return false;
                }
                out.extend(&part[p..end]);
                match after {
                    Some(t) => t.apply_to(&part[end..], out),
                    None => s == 0,
                }
            }
        }
    }

    /// The tree as its label (see the module comment).
    pub fn label(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            EditTree::Subst { from, to } => {
                out.push_str("(s ");
                quote(from, out);
                out.push(' ');
                quote(to, out);
                out.push(')');
            }
            EditTree::Match { prefix, suffix, before, after } => {
                out.push_str(&format!("(m {prefix} {suffix} "));
                match before {
                    Some(t) => t.write(out),
                    None => out.push_str("()"),
                }
                out.push(' ');
                match after {
                    Some(t) => t.write(out),
                    None => out.push_str("()"),
                }
                out.push(')');
            }
        }
    }

    /// The tree a label writes, refused unless the label is exactly what
    /// [`label`](Self::label) writes for it.
    pub fn parse(label: &str) -> Result<EditTree> {
        let mut p = Parser { s: label.as_bytes(), at: 0, text: label, depth: 0 };
        let tree = p.tree()?.ok_or_else(|| p.error("a tree, not ()"))?;
        if p.at != p.s.len() {
            return Err(p.error("text after the tree"));
        }
        if tree.label() != label {
            return Err(p.error("not the canonical form"));
        }
        Ok(tree)
    }
}

/// The deepest tree a label may nest; a label of a head is at most 128
/// bytes, which nests far less.
const MAX_DEPTH: usize = 64;

struct Parser<'a> {
    s: &'a [u8],
    at: usize,
    text: &'a str,
    depth: usize,
}

impl Parser<'_> {
    fn error(&self, what: &str) -> crate::status::Error {
        invalid(format!("edit tree {:?}: {what} at byte {}", self.text, self.at))
    }

    fn eat(&mut self, b: u8) -> Result<()> {
        if self.s.get(self.at) == Some(&b) {
            self.at += 1;
            Ok(())
        } else {
            Err(self.error(&format!("expected {:?}", b as char)))
        }
    }

    fn number(&mut self) -> Result<u32> {
        let start = self.at;
        while self.s.get(self.at).is_some_and(u8::is_ascii_digit) {
            self.at += 1;
        }
        self.text[start..self.at].parse().map_err(|_| self.error("expected a length"))
    }

    fn quoted(&mut self) -> Result<String> {
        self.eat(b'\'')?;
        let mut out = String::new();
        let mut chars = self.text[self.at..].char_indices();
        while let Some((i, c)) = chars.next() {
            match c {
                '\'' => {
                    self.at += i + 1;
                    return Ok(out);
                }
                '\\' => match chars.next() {
                    Some((_, e @ ('\'' | '\\'))) => out.push(e),
                    _ => return Err(self.error("a bad escape")),
                },
                c => out.push(c),
            }
        }
        Err(self.error("an unclosed quote"))
    }

    /// A tree, or None for `()`.
    fn tree(&mut self) -> Result<Option<EditTree>> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.error("nesting too deep"));
        }
        let t = self.node();
        self.depth -= 1;
        t
    }

    fn node(&mut self) -> Result<Option<EditTree>> {
        self.eat(b'(')?;
        match self.s.get(self.at) {
            Some(b')') => {
                self.at += 1;
                Ok(None)
            }
            Some(b's') => {
                self.at += 1;
                self.eat(b' ')?;
                let from = self.quoted()?;
                self.eat(b' ')?;
                let to = self.quoted()?;
                self.eat(b')')?;
                Ok(Some(EditTree::Subst { from, to }))
            }
            Some(b'm') => {
                self.at += 1;
                self.eat(b' ')?;
                let prefix = self.number()?;
                self.eat(b' ')?;
                let suffix = self.number()?;
                self.eat(b' ')?;
                let before = self.tree()?.map(Box::new);
                self.eat(b' ')?;
                let after = self.tree()?.map(Box::new);
                self.eat(b')')?;
                Ok(Some(EditTree::Match { prefix, suffix, before, after }))
            }
            _ => Err(self.error("expected m, s or )")),
        }
    }
}
