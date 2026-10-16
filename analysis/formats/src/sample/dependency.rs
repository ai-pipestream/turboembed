//! Dependency samples: `opennlp.tools.depparse.DependencySample` and the
//! `DependencyTree` it holds.
//!
//! A tree gives every word of a sentence a head and the label of the
//! relation to it. [`DependencyTree::new`] makes the checks of
//! `DependencyTree.of`: at least one word, a head and a relation per word,
//! every head another word of the sentence or the root, exactly one root,
//! no relation that is blank, and no cycle. A tree that passes them is a
//! tree in the graph sense: every word reaches the root by its heads.

use std::fmt;

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use crate::error::FormatError;
use crate::source::Field;

/// The head of a word: the root of the sentence, or another word by its
/// index in the sentence, counting from 0. A CoNLL-U `HEAD` column holds
/// the index plus one, and `0` for the root.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Head {
    Root,
    Word(u32),
}

impl Head {
    /// The head as the CoNLL-U `HEAD` column writes it: `0` for the root
    /// and the word's index plus one for a word. `DependencySample.toString`
    /// writes the same number (`ROOT_HEAD`, -1, plus one).
    pub fn column(self) -> u64 {
        match self {
            Head::Root => 0,
            Head::Word(w) => u64::from(w) + 1,
        }
    }
}

/// One arc of a tree: a word, its head and the relation between them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Arc<'t, 'a> {
    pub dependent: u32,
    pub head: Head,
    pub relation: &'t Field<'a>,
}

/// Why `DependencyTree.of` refuses heads and relations. Each is the
/// `IllegalArgumentException` it throws, with its message as `Display`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeError {
    /// No words.
    Empty,
    /// Not one relation per head.
    Lengths { heads: usize, relations: usize },
    /// A head that names no word of the sentence.
    HeadOutOfRange { word: u32, head: u32 },
    /// A word that is its own head.
    SelfHead { word: u32 },
    /// A relation that is empty or all whitespace in the whitespace mode
    /// (`StringUtil.isBlank`).
    BlankRelation { word: u32 },
    /// Not exactly one root.
    Roots { count: usize },
    /// A chain of heads that comes back to a word on it instead of
    /// reaching the root; `word` is where `checkAcyclic` finds it.
    Cycle { word: u32 },
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::Empty => f.write_str("a dependency tree needs at least one token"),
            TreeError::Lengths { heads, relations } => {
                write!(f, "heads and relations must have the same length: {heads} != {relations}")
            }
            TreeError::HeadOutOfRange { word, head } => write!(f, "head of token {word} is out of range: {head}"),
            TreeError::SelfHead { word } => write!(f, "token {word} must not head itself"),
            TreeError::BlankRelation { word } => write!(f, "relation of token {word} must not be blank"),
            TreeError::Roots { count } => write!(f, "expected exactly one root, found {count}"),
            TreeError::Cycle { word } => write!(f, "dependency tree contains a cycle at token {word}"),
        }
    }
}

impl std::error::Error for TreeError {}

/// The heads and relations of a sentence's words: `DependencyTree`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyTree<'a> {
    heads: Vec<Head>,
    relations: Vec<Field<'a>>,
    root: u32,
}

impl<'a> DependencyTree<'a> {
    /// `DependencyTree.of(heads, relations)`, with blank relations tested
    /// in `mode`. The words are checked in order, each for its head and
    /// then its relation, then the number of roots, then cycles, so the
    /// error is the one Java throws.
    pub fn new(heads: Vec<Head>, relations: Vec<Field<'a>>, mode: WhitespaceMode) -> Result<Self, TreeError> {
        if heads.is_empty() {
            return Err(TreeError::Empty);
        }
        if heads.len() != relations.len() {
            return Err(TreeError::Lengths { heads: heads.len(), relations: relations.len() });
        }
        let n = u32::try_from(heads.len()).map_err(|_| TreeError::Lengths { heads: heads.len(), relations: 0 })?;
        let mut roots = 0;
        let mut root = 0;
        for (word, (&head, relation)) in (0..n).zip(heads.iter().zip(&relations)) {
            match head {
                Head::Root => {
                    roots += 1;
                    root = word;
                }
                Head::Word(h) if h >= n => return Err(TreeError::HeadOutOfRange { word, head: h }),
                Head::Word(h) if h == word => return Err(TreeError::SelfHead { word }),
                Head::Word(_) => {}
            }
            if relation.as_str().chars().all(|c| mode.is_whitespace(c as u32)) {
                return Err(TreeError::BlankRelation { word });
            }
        }
        if roots != 1 {
            return Err(TreeError::Roots { count: roots });
        }
        check_acyclic(&heads)?;
        Ok(DependencyTree { heads, relations, root })
    }

    /// The number of words.
    pub fn len(&self) -> usize {
        self.heads.len()
    }

    /// Always false: a tree has a word.
    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    pub fn heads(&self) -> &[Head] {
        &self.heads
    }

    pub fn relations(&self) -> &[Field<'a>] {
        &self.relations
    }

    /// The word whose head is the root.
    pub fn root(&self) -> u32 {
        self.root
    }

    /// The arcs in word order, one per word.
    pub fn arcs(&self) -> impl ExactSizeIterator<Item = Arc<'_, 'a>> + '_ {
        self.heads.iter().zip(&self.relations).enumerate().map(|(dependent, (&head, relation))| Arc {
            dependent: dependent as u32,
            head,
            relation,
        })
    }
}

/// States of `checkAcyclic`'s walk.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Walk {
    Unvisited,
    Visiting,
    Visited,
}

/// `DependencyTree.checkAcyclic`: from each word, follows its heads until
/// the root or a word already known to reach it; coming back to a word on
/// the chain being followed is a cycle. Every head is already a word of
/// the sentence.
fn check_acyclic(heads: &[Head]) -> Result<(), TreeError> {
    let mut states = vec![Walk::Unvisited; heads.len()];
    let next = |w: u32| match heads[w as usize] {
        Head::Root => None,
        Head::Word(h) => Some(h),
    };
    for start in 0..heads.len() as u32 {
        let mut current = Some(start);
        while let Some(w) = current.filter(|&w| states[w as usize] == Walk::Unvisited) {
            states[w as usize] = Walk::Visiting;
            current = next(w);
        }
        if let Some(w) = current
            && states[w as usize] == Walk::Visiting
        {
            return Err(TreeError::Cycle { word: w });
        }
        let mut current = Some(start);
        while let Some(w) = current.filter(|&w| states[w as usize] == Walk::Visiting) {
            states[w as usize] = Walk::Visited;
            current = next(w);
        }
    }
    Ok(())
}

/// A sentence with a part-of-speech tag per token and its dependency
/// tree: `DependencySample`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencySample<'a> {
    tokens: Vec<Field<'a>>,
    tags: Vec<Field<'a>>,
    tree: DependencyTree<'a>,
    source: Span,
}

impl<'a> DependencySample<'a> {
    /// A sample of `tokens` with their `tags` and `tree`, read from
    /// `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::Lengths`] when the tags or the tree do not have one
    /// entry per token.
    pub fn new(
        tokens: Vec<Field<'a>>,
        tags: Vec<Field<'a>>,
        tree: DependencyTree<'a>,
        source: Span,
    ) -> Result<DependencySample<'a>, FormatError> {
        if tags.len() != tokens.len() {
            return Err(FormatError::Lengths { what: "tags", expected: tokens.len(), found: tags.len() });
        }
        if tree.len() != tokens.len() {
            return Err(FormatError::Lengths { what: "tree", expected: tokens.len(), found: tree.len() });
        }
        Ok(DependencySample { tokens, tags, tree, source })
    }

    pub fn tokens(&self) -> &[Field<'a>] {
        &self.tokens
    }

    /// The tags, one per token.
    pub fn tags(&self) -> &[Field<'a>] {
        &self.tags
    }

    pub fn tree(&self) -> &DependencyTree<'a> {
        &self.tree
    }

    /// The bytes of the source the sample was read from, from the start
    /// of the sentence's first word line to the end of its last.
    pub fn source(&self) -> Span {
        self.source
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Always false: a sample has a token.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// `DependencySample.toString`: a line per token of its number, the
/// token, the tag, the head's number (`0` for the root) and the relation,
/// separated by tabs, each line ended by a line feed (the line separator
/// of the systems the goldens are made on).
impl fmt::Display for DependencySample<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, ((token, tag), arc)) in self.tokens.iter().zip(&self.tags).zip(self.tree.arcs()).enumerate() {
            writeln!(f, "{}\t{token}\t{tag}\t{}\t{}", i + 1, arc.head.column(), arc.relation)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(s: &'static str) -> Field<'static> {
        Field::made(s.to_owned(), Span::new(0, 0))
    }

    fn tree(heads: &[Head], rels: &[&'static str]) -> Result<DependencyTree<'static>, TreeError> {
        DependencyTree::new(heads.to_vec(), rels.iter().map(|r| rel(r)).collect(), WhitespaceMode::Unicode)
    }

    use Head::{Root, Word};

    #[test]
    fn a_tree_has_one_root_and_no_cycle() {
        let t = tree(&[Word(1), Root, Word(1)], &["nsubj", "root", "obj"]).unwrap();
        assert_eq!(t.root(), 1);
        let arcs: Vec<(u32, Head, &str)> = t.arcs().map(|a| (a.dependent, a.head, a.relation.as_str())).collect();
        assert_eq!(arcs, [(0, Word(1), "nsubj"), (1, Root, "root"), (2, Word(1), "obj")]);
    }

    #[test]
    fn refusals_in_java_order() {
        assert_eq!(tree(&[], &[]), Err(TreeError::Empty));
        assert_eq!(tree(&[Root], &[]), Err(TreeError::Lengths { heads: 1, relations: 0 }));
        // The head of a word is checked before its relation, and both
        // before the number of roots.
        assert_eq!(tree(&[Word(5), Root], &[" ", "x"]), Err(TreeError::HeadOutOfRange { word: 0, head: 5 }));
        assert_eq!(tree(&[Root, Word(1)], &["x", "y"]), Err(TreeError::SelfHead { word: 1 }));
        assert_eq!(tree(&[Root, Root], &["\u{a0}", "y"]), Err(TreeError::BlankRelation { word: 0 }));
        assert_eq!(tree(&[Root, Root], &["x", "y"]), Err(TreeError::Roots { count: 2 }));
        assert_eq!(tree(&[Word(1), Word(0)], &["x", "y"]), Err(TreeError::Roots { count: 0 }));
        // 0 -> root, 1 -> 2 -> 3 -> 1: the walk from 1 comes back to 1.
        assert_eq!(tree(&[Root, Word(2), Word(3), Word(1)], &["a", "b", "c", "d"]), Err(TreeError::Cycle { word: 1 }));
        // A chain into a cycle reports the first word of the cycle it meets.
        assert_eq!(tree(&[Word(1), Word(2), Word(1), Root], &["a", "b", "c", "d"]), Err(TreeError::Cycle { word: 1 }));
    }

    #[test]
    fn blank_follows_the_whitespace_mode() {
        // U+0085 is whitespace in the Unicode mode only, U+001C in the
        // legacy mode only.
        let heads = vec![Root];
        let unicode = DependencyTree::new(heads.clone(), vec![rel("\u{85}")], WhitespaceMode::Unicode);
        assert_eq!(unicode, Err(TreeError::BlankRelation { word: 0 }));
        assert!(DependencyTree::new(heads.clone(), vec![rel("\u{85}")], WhitespaceMode::Legacy).is_ok());
        assert!(DependencyTree::new(heads, vec![rel("\u{1c}")], WhitespaceMode::Legacy).is_err());
    }

    #[test]
    fn long_chains_are_walked_once() {
        let n = 100_000u32;
        let heads: Vec<Head> = (0..n).map(|i| if i == n - 1 { Root } else { Word(i + 1) }).collect();
        let rels = (0..n).map(|_| rel("dep")).collect();
        let t = DependencyTree::new(heads, rels, WhitespaceMode::Unicode).unwrap();
        assert_eq!(t.root(), n - 1);
    }

    #[test]
    fn written_as_java_writes_it() {
        let src = "a b";
        let tokens = vec![Field::of(src, Span::new(0, 1)), Field::of(src, Span::new(2, 3))];
        let tags = vec![rel("X"), rel("Y")];
        let t = tree(&[Root, Word(0)], &["root", "dep"]).unwrap();
        let s = DependencySample::new(tokens, tags, t, Span::new(0, 3)).unwrap();
        assert_eq!(s.to_string(), "1\ta\tX\t0\troot\n2\tb\tY\t1\tdep\n");
    }
}
