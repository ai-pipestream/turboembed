//! Chunk samples from Arvores Deitadas: `ADChunkSampleStream`.

use super::sentence::{AdSentence, SentenceRead, SentenceStream};
use super::tree::{Element, Leaf, Tree};
use crate::error::FormatError;
use crate::sample::ChunkSample;
use crate::source::{Field, Lines};

const OTHER: &str = "O";

/// The phrase type of a node's tag (`getChunkTag(Node)`): the part after
/// its last colon without trailing hyphens, upper-cased when it is `np`,
/// `vp`, `pp`, `ap`, `advp` or `adjp`, else `O`.
fn node_phrase(tag: &str) -> &'static str {
    let kind = tag.rsplit(':').next().unwrap_or(tag).trim_end_matches('-');
    match kind {
        "np" => "NP",
        "vp" => "VP",
        "pp" => "PP",
        "ap" => "AP",
        "advp" => "ADVP",
        "adjp" => "ADJP",
        _ => OTHER,
    }
}

/// Whether a chunk tag, as its prefix and phrase type, ends with `-` and
/// `phrase` (Java's `endsWith("-" + phraseTag)`). No phrase type has a
/// hyphen, so that is a `B-` or `I-` tag of that type.
fn ends_with_phrase((prefix, kind): (&str, &str), phrase: &str) -> bool {
    !prefix.is_empty() && kind == phrase
}

/// The tokens, tags and chunk tags of a sentence as they are built.
struct Columns<'a> {
    tokens: Vec<Field<'a>>,
    tags: Vec<Field<'a>>,
    /// The chunk tags, as a prefix (`B-`, `I-` or none) and a phrase type.
    chunks: Vec<(&'static str, &'static str)>,
    /// The line of each token's leaf.
    lines: Vec<turbo_analysis::Span>,
}

impl<'a> Columns<'a> {
    fn last_ends_with(&self, phrase: &str) -> bool {
        self.chunks.last().is_some_and(|&chunk| ends_with_phrase(chunk, phrase))
    }

    /// `processLeaf`: the leaf's word, its word class (itself when it has
    /// no syntactic or functional tag), and its chunk tag.
    fn leaf(&mut self, leaf: &Leaf<'a>, intermediate: bool, mut phrase: &'static str) {
        if let Some(f) = leaf.functional.filter(|_| phrase == OTHER) {
            phrase = match f.text {
                "v-fin" => "VP",
                "n" => "NP",
                _ => OTHER,
            };
        }
        let prefix = match (phrase == OTHER, intermediate) {
            (true, _) => "",
            (false, true) => "I-",
            (false, false) => "B-",
        };
        self.tokens.push(leaf.lexeme.field());
        let tag = match (leaf.syntactic, leaf.functional) {
            (Some(_), Some(f)) => f,
            _ => leaf.lexeme,
        };
        self.tags.push(tag.field());
        self.chunks.push((prefix, phrase));
        self.lines.push(leaf.line);
    }
}

/// Chunk samples read from an Arvores Deitadas text, one per sentence
/// with at least one leaf: `ADChunkSampleStream`.
///
/// A token is a leaf's word, its tag the leaf's word class (punctuation is
/// tagged with itself), and its chunk tag comes from the nodes above it:
/// the phrase types `NP`, `VP`, `PP`, `AP`, `ADVP` and `ADJP` of a node's
/// tag, passed down to nodes of other types, `VP` for a leaf whose
/// syntactic tag is `P`, and for a leaf outside any phrase `VP` for a
/// finite verb and `NP` for a noun. Punctuation at the edge of a phrase is
/// `O`. These rules follow Fernandes, dos Santos and Milidiu, "A Machine
/// Learning Approach to Portuguese Clause Identification".
///
/// `start` skips the sentences before it, and `end` stops the reading at
/// it; both count the sentences with leaves, and the skipped ones
/// whatever they hold.
///
/// Tokens and tags are fields of the text; a chunk tag has a text of its
/// own and the span of its leaf's line.
pub struct AdChunkReader<'a> {
    sentences: SentenceStream<'a>,
    start: Option<u32>,
    end: Option<u32>,
    index: u32,
    done: bool,
}

impl<'a> AdChunkReader<'a> {
    /// A reader of `source`, which must fit `u32` offsets, from the
    /// sentence `start` to before `end`.
    pub fn new(source: &'a str, start: Option<u32>, end: Option<u32>) -> Result<AdChunkReader<'a>, FormatError> {
        Ok(AdChunkReader { sentences: SentenceStream::new(Lines::new(source)?), start, end, index: 0, done: false })
    }

    fn sample(sentence: &AdSentence<'a>) -> Columns<'a> {
        let tree = &sentence.tree;
        let mut columns = Columns { tokens: Vec::new(), tags: Vec::new(), chunks: Vec::new(), lines: Vec::new() };
        for &child in tree.root_children() {
            match &tree.elements[child] {
                Element::Leaf(leaf) => columns.leaf(leaf, false, OTHER),
                Element::Node { .. } => node(tree, child, &mut columns, None),
            }
        }
        columns
    }
}

/// `processNode`.
fn node<'a>(tree: &Tree<'a>, index: usize, columns: &mut Columns<'a>, inherited: Option<&'static str>) {
    let Element::Node { tag, children } = &tree.elements[index] else { unreachable!("a node") };
    let mut phrase = node_phrase(tag.map_or("", |t| t.text));
    let mut is_inherited = false;
    if let (OTHER, Some(i)) = (phrase, inherited) {
        phrase = i;
        is_inherited = true;
    }
    let is_leaf = |i: usize| matches!(tree.elements[children[i]], Element::Leaf(_));
    for (i, &child) in children.iter().enumerate() {
        match &tree.elements[child] {
            Element::Leaf(leaf) => {
                let mut tag = phrase;
                if leaf.syntactic.is_some_and(|s| s.text == "P") && tag != "VP" {
                    tag = "VP";
                }
                let mut intermediate = columns.last_ends_with(tag) && (is_inherited || i > 0);
                let next_is_leaf = i + 1 < children.len() && is_leaf(i + 1);
                let previous_is_leaf = i > 0 && is_leaf(i - 1);
                if leaf.functional.is_none() && (!next_is_leaf || !previous_is_leaf) {
                    intermediate = false;
                    tag = OTHER;
                }
                columns.leaf(leaf, intermediate, tag);
            }
            Element::Node { .. } => {
                let before = columns.chunks.len();
                node(tree, child, columns, Some(phrase));
                // A child of another type breaks the phrase.
                let broken = columns.chunks[before..].iter().rev().any(|&chunk| !ends_with_phrase(chunk, phrase));
                if broken {
                    phrase = OTHER;
                }
            }
        }
    }
}

impl<'a> Iterator for AdChunkReader<'a> {
    type Item = Result<ChunkSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.done {
            let sentence = match self.sentences.read() {
                Some(SentenceRead::Sentence(s)) => s,
                Some(SentenceRead::Failed) | None => break,
            };
            if self.end.is_some_and(|end| self.index >= end) {
                break;
            }
            if self.start.is_some_and(|start| self.index < start) {
                self.index += 1;
                continue;
            }
            let c = AdChunkReader::sample(&sentence);
            if !c.tokens.is_empty() {
                self.index += 1;
                let chunks = c
                    .chunks
                    .iter()
                    .zip(&c.lines)
                    .map(|(&(prefix, kind), &line)| Field::made(format!("{prefix}{kind}"), line))
                    .collect();
                return Some(ChunkSample::new(c.tokens, c.tags, chunks, sentence.source));
            }
        }
        self.done = true;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phrase_types() {
        assert_eq!(node_phrase("P<:np"), "NP");
        assert_eq!(node_phrase("X:advp--"), "ADVP");
        assert_eq!(node_phrase("STA:fcl"), "O");
        assert!(ends_with_phrase(("B-", "NP"), "NP"));
        assert!(!ends_with_phrase(("B-", "ADVP"), "VP"));
        assert!(!ends_with_phrase(("", "O"), "O"));
    }

    #[test]
    fn samples() {
        let src = "<s>\nSOURCE: x\n1 O gato come.\nSTA:fcl\n=SUBJ:np\n==>N:art('o' <artd> M S) O\n==H:n('gato' M S) gato\n=P:v-fin('comer' PR 3S IND) come\n.\n</s>\n";
        let samples: Vec<_> = AdChunkReader::new(src, None, None).unwrap().map(|s| s.unwrap().to_string()).collect();
        assert_eq!(samples, ["O art B-NP\ngato n I-NP\ncome v-fin B-VP\n. . O\n"]);
    }
}
