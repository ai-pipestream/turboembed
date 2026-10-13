//! The Viterbi search: OpenNLP's `LatticeTokenizer.analyze` over a loaded
//! [`MecabDictionary`].

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use super::chardef::AssignmentId;
use super::dictionary::{BOUNDARY, EntryId, MecabDictionary};
use super::error::NoCandidate;
use crate::TokenizerOptions;
use crate::whitespace::whitespace_spans;

/// No node.
const NONE: u32 = u32::MAX;

/// One morpheme of a segmentation: its span in the text, the dictionary
/// entry it is, and whether unknown word handling made it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Morpheme {
    pub span: Span,
    pub entry: EntryId,
    pub unknown: bool,
}

/// One lattice node: a candidate morpheme of the current stretch with the
/// cost of the cheapest path that ends with it. The nodes ending at one
/// position are chained through `next`, the latest first, as the Java
/// chains them.
#[derive(Copy, Clone, Debug)]
struct Node {
    /// Byte offsets in the stretch.
    start: u32,
    end: u32,
    entry: u32,
    /// The entry's right context id, which the nodes after it look up.
    right: u32,
    unknown: bool,
    cost: i64,
    previous: u32,
    next: u32,
}

/// The buffers one segmentation reuses, so that repeated calls allocate
/// nothing once they have grown: the whitespace-free stretches, the
/// lattice nodes, the head of the chain of nodes ending at each byte of a
/// stretch, each character's categories and run end, and the morphemes
/// [`LatticeTokenizer::tokenize_with`] takes the spans of.
#[derive(Debug, Default, Clone)]
pub struct Scratch {
    stretches: Vec<Span>,
    nodes: Vec<Node>,
    head: Vec<u32>,
    assignment: Vec<AssignmentId>,
    run_end: Vec<u32>,
    morphemes: Vec<Morpheme>,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch::default()
    }
}

/// The lattice tokenizer: a dictionary and the whitespace mode the text is
/// split in.
#[derive(Debug, Clone)]
pub struct LatticeTokenizer {
    pub(crate) dictionary: MecabDictionary,
    pub(crate) whitespace: WhitespaceMode,
}

impl LatticeTokenizer {
    /// A tokenizer over `dictionary` that splits the text at whitespace in
    /// `whitespace` mode. The dictionary keeps the mode its files were
    /// split in; OpenNLP reads one process-wide mode for both.
    pub fn new(dictionary: MecabDictionary, whitespace: WhitespaceMode) -> LatticeTokenizer {
        LatticeTokenizer { dictionary, whitespace }
    }

    pub fn dictionary(&self) -> &MecabDictionary {
        &self.dictionary
    }

    pub fn whitespace(&self) -> WhitespaceMode {
        self.whitespace
    }

    /// Appends the morphemes of `text` to `out`, in text order, with byte
    /// spans of `text`. Whitespace is never part of a morpheme. Fails, with
    /// `out` holding the morphemes of the stretches before, where the
    /// dictionary offers no candidate.
    pub fn analyze(&self, text: &str, out: &mut Vec<Morpheme>) -> Result<(), NoCandidate> {
        self.analyze_with(text, &mut Scratch::new(), out)
    }

    /// [`analyze`](Self::analyze) with the caller's buffers.
    pub fn analyze_with(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<Morpheme>) -> Result<(), NoCandidate> {
        scratch.stretches.clear();
        let options = TokenizerOptions { whitespace: self.whitespace, keep_newlines: false };
        whitespace_spans(text, options, &mut scratch.stretches);
        for i in 0..scratch.stretches.len() {
            let stretch = scratch.stretches[i];
            self.decode(text, stretch, scratch, out)?;
        }
        Ok(())
    }

    /// The spans of the morphemes of `text`, appended to `out`; fails as
    /// [`analyze`](Self::analyze) fails.
    pub fn tokenize_with(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<Span>) -> Result<(), NoCandidate> {
        let mut morphemes = std::mem::take(&mut scratch.morphemes);
        morphemes.clear();
        let result = self.analyze_with(text, scratch, &mut morphemes);
        out.extend(morphemes.iter().map(|m| m.span));
        scratch.morphemes = morphemes;
        result
    }

    /// The cheapest path through one whitespace-free stretch.
    ///
    /// From each character boundary that a node ends at (and the start),
    /// in order, the candidates are the lexicon entries of every surface
    /// that starts there, shortest surface first and each surface's
    /// entries in file order, then the unknown word candidates. Each
    /// candidate takes the predecessor whose path plus connection cost is
    /// strictly lowest, the latest added first, and is then chained before
    /// the nodes already ending where it ends. At the end of the stretch
    /// the first node of that chain with the strictly lowest total wins.
    fn decode(
        &self,
        text: &str,
        stretch: Span,
        scratch: &mut Scratch,
        out: &mut Vec<Morpheme>,
    ) -> Result<(), NoCandidate> {
        let dict = &self.dictionary;
        let part = &text[stretch.range()];
        let bytes = part.as_bytes();
        let n = bytes.len();
        let Scratch { nodes, head, assignment, run_end, .. } = scratch;
        nodes.clear();
        head.clear();
        head.resize(n + 1, NONE);
        self.category_runs(part, assignment, run_end);

        let mut i = 0;
        while i < n {
            let width = char_width(bytes[i]);
            if i > 0 && head[i] == NONE {
                i += width;
                continue;
            }
            let first = nodes.len();
            let start = i as u32;
            dict.trie.prefixes(&bytes[i..], |len, value| {
                for e in dict.group(value) {
                    let right = dict.entries[e as usize].right;
                    nodes.push(Node::new(start, (i + len) as u32, e, right, false));
                }
            });
            let matched = nodes.len() > first;
            let primary = dict.chars.assignment(assignment[i]).primary;
            let category = dict.category(primary);
            if (!matched || category.invoke)
                && let Some(templates) = dict.templates(primary)
            {
                let run = run_end[i] as usize;
                let mut push = |end: usize| {
                    for e in templates.clone() {
                        let right = dict.entries[e as usize].right;
                        nodes.push(Node::new(start, end as u32, e, right, true));
                    }
                };
                if category.group {
                    push(run);
                }
                let mut end = i;
                for _ in 0..category.length {
                    if end >= run {
                        break;
                    }
                    end += char_width(bytes[end]);
                    if category.group && end == run {
                        continue;
                    }
                    push(end);
                }
            }
            if nodes.len() == first {
                // A category that offered nothing still gives its templates
                // one character; without templates, DEFAULT's do.
                let fallback = dict.templates(primary).or_else(|| dict.templates(dict.default_category()));
                for e in fallback.into_iter().flatten() {
                    let right = dict.entries[e as usize].right;
                    nodes.push(Node::new(start, (i + width) as u32, e, right, true));
                }
            }
            if nodes.len() == first {
                return Err(NoCandidate { offset: stretch.start as usize + i });
            }
            let predecessors = if i == 0 { None } else { Some(head[i]) };
            for k in first..nodes.len() {
                self.relax(nodes, k, predecessors);
                if nodes[k].cost < i64::MAX {
                    let end = nodes[k].end as usize;
                    nodes[k].next = head[end];
                    head[end] = k as u32;
                }
            }
            i += width;
        }

        let mut best = NONE;
        let mut best_total = i64::MAX;
        let mut at = head[n];
        while at != NONE {
            let node = &nodes[at as usize];
            let total = node.cost + i64::from(dict.matrix.cost(node.right, BOUNDARY));
            if best == NONE || total < best_total {
                best = at;
                best_total = total;
            }
            at = node.next;
        }
        // Every reachable position offers a candidate that ends further on
        // and no later than the stretch's end, so the end is reached.
        debug_assert!(best != NONE || n == 0);
        let first = out.len();
        while best != NONE {
            let node = &nodes[best as usize];
            out.push(Morpheme {
                span: Span { start: stretch.start + node.start, end: stretch.start + node.end },
                entry: EntryId(node.entry),
                unknown: node.unknown,
            });
            best = node.previous;
        }
        out[first..].reverse();
        Ok(())
    }

    /// Gives node `k` the cheapest path to it: from the start of the
    /// stretch when `predecessors` is `None`, else through the chain of
    /// nodes that starts at `predecessors`, a strictly lower total
    /// replacing the one found before.
    #[inline(always)]
    fn relax(&self, nodes: &mut [Node], k: usize, predecessors: Option<u32>) {
        let dict = &self.dictionary;
        let entry = dict.entries[nodes[k].entry as usize];
        let own = i64::from(entry.cost);
        let Some(mut at) = predecessors else {
            nodes[k].cost = own + i64::from(dict.matrix.cost(BOUNDARY, entry.left));
            return;
        };
        let mut best = i64::MAX;
        let mut previous = NONE;
        while at != NONE {
            let p = &nodes[at as usize];
            let total = p.cost + i64::from(dict.matrix.cost(p.right, entry.left)) + own;
            if total < best {
                best = total;
                previous = at;
            }
            at = p.next;
        }
        nodes[k].cost = best;
        nodes[k].previous = previous;
    }

    /// Each character's categories, and the end of the run that starts at
    /// it: the run goes on while each character shares a category with the
    /// one after it, as MeCab's `seekToOtherType` does. One pass from the
    /// end of the stretch.
    fn category_runs(&self, part: &str, assignment: &mut Vec<AssignmentId>, run_end: &mut Vec<u32>) {
        let dict = &self.dictionary;
        let n = part.len();
        assignment.clear();
        assignment.resize(n, AssignmentId::NONE);
        run_end.clear();
        run_end.resize(n, 0);
        let mut next_mask = 0u32;
        let mut next_run_end = n as u32;
        for (i, c) in part.char_indices().rev() {
            let id = dict.assignment_of(c as u32);
            let mask = dict.chars.assignment(id).mask;
            let end = if mask & next_mask != 0 { next_run_end } else { (i + c.len_utf8()) as u32 };
            assignment[i] = id;
            run_end[i] = end;
            next_mask = mask;
            next_run_end = end;
        }
    }
}

impl Node {
    #[inline(always)]
    fn new(start: u32, end: u32, entry: u32, right: u32, unknown: bool) -> Node {
        Node { start, end, entry, right, unknown, cost: i64::MAX, previous: NONE, next: NONE }
    }
}

/// The length of the UTF-8 character whose first byte is `b`.
#[inline(always)]
fn char_width(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{DictionaryFiles, DictionaryLimits};

    /// OpenNLP's miniature dictionary from `LatticeTokenizerTest`.
    const LEXICON: &str = "東京,0,0,3000,noun,proper\n京都,0,0,3000,noun,proper\n東,0,0,6000,noun,common\n\
        都,0,0,4000,noun,suffix\nに,0,0,1000,particle,case\n行く,0,0,3000,verb,base\n";
    const CHAR_DEF: &str = "DEFAULT 0 1 0\nKANJI 0 0 2\nHIRAGANA 0 1 0\nLATIN 1 1 0\n\n\
        0x3041..0x3096 HIRAGANA\n0x4E00..0x9FFF KANJI\n0x0041..0x005A LATIN\n0x0061..0x007A LATIN\n";
    const UNK_DEF: &str = "DEFAULT,0,0,10000,symbol,unknown\nLATIN,0,0,4000,noun,foreign\n\
        KANJI,0,0,8000,noun,unknown\nHIRAGANA,0,0,9000,particle,unknown\n";

    fn tokenizer_of(matrix: &str, lexicon: &str, char_def: &str, unk_def: &str) -> LatticeTokenizer {
        let lexicons = [("lexicon.csv", lexicon.as_bytes())];
        let files = DictionaryFiles {
            matrix: matrix.as_bytes(),
            lexicons: &lexicons,
            char_def: char_def.as_bytes(),
            unk_def: unk_def.as_bytes(),
        };
        let d = MecabDictionary::from_files(files, WhitespaceMode::Unicode, DictionaryLimits::default()).unwrap();
        LatticeTokenizer::new(d, WhitespaceMode::Unicode)
    }

    fn tokenizer() -> LatticeTokenizer {
        tokenizer_of("1 1\n0 0 0\n", LEXICON, CHAR_DEF, UNK_DEF)
    }

    /// A morpheme as the tests compare it: surface, span, features and
    /// whether it is unknown.
    type Seen = (String, (u32, u32), Vec<String>, bool);

    fn analyze(t: &LatticeTokenizer, text: &str) -> Vec<Seen> {
        let mut out = Vec::new();
        t.analyze(text, &mut out).unwrap();
        out.iter()
            .map(|m| {
                let f = t.dictionary().features(m.entry).map(str::to_string).collect();
                (text[m.span.range()].to_string(), (m.span.start, m.span.end), f, m.unknown)
            })
            .collect()
    }

    fn words(t: &LatticeTokenizer, text: &str) -> Vec<String> {
        analyze(t, text).into_iter().map(|m| m.0).collect()
    }

    // The cases of OpenNLP's LatticeTokenizerTest, with UTF-8 offsets
    // where the Java has UTF-16 ones.

    #[test]
    fn prefers_the_cheaper_segmentation() {
        let t = tokenizer();
        let got = analyze(&t, "東京都に行く");
        assert_eq!(words(&t, "東京都に行く"), ["東京", "都", "に", "行く"]);
        assert_eq!(got.iter().map(|m| m.1).collect::<Vec<_>>(), [(0, 6), (6, 9), (9, 12), (12, 18)]);
        assert_eq!(got[0].2, ["noun", "proper"]);
        assert_eq!(got[2].2, ["particle", "case"]);
        assert!(!got[0].3);
    }

    #[test]
    fn unknown_runs() {
        let t = tokenizer();
        let got = analyze(&t, "ABCに行く");
        assert_eq!(got[0].0, "ABC");
        assert!(got[0].3);
        assert_eq!(got[0].2, ["noun", "foreign"]);
        let got = analyze(&t, "峠道に行く");
        assert_eq!((got.len(), got[0].0.as_str(), got[0].3), (3, "峠道", true));
        // Never across a category boundary, but within one all lengths.
        assert_eq!(words(&t, "峠a"), ["峠", "a"]);
        assert_eq!(words(&t, "峠道a"), ["峠道", "a"]);
        let got = analyze(&t, "①②③");
        assert_eq!((got.len(), got[0].1, got[0].2.clone()), (1, (0, 9), vec!["symbol".into(), "unknown".into()]));
        let got = analyze(&t, "東京①に行く");
        assert_eq!(got.iter().map(|m| m.3).collect::<Vec<_>>(), [false, true, false, false]);
    }

    #[test]
    fn whitespace_separates_and_is_never_a_morpheme() {
        let t = tokenizer();
        let got = analyze(&t, "東京 に 行く");
        assert_eq!(got.iter().map(|m| m.1).collect::<Vec<_>>(), [(0, 6), (7, 10), (11, 17)]);
        assert!(analyze(&t, "   ").is_empty());
        assert!(analyze(&t, "").is_empty());
        let got = analyze(&t, "  東京都に行く");
        assert_eq!(got[0].1, (2, 8));
    }

    #[test]
    fn single_characters() {
        let t = tokenizer();
        let got = analyze(&t, "に");
        assert_eq!((got.len(), got[0].1, got[0].3), (1, (0, 3), false));
        let got = analyze(&t, "峠");
        assert_eq!((got.len(), got[0].1, got[0].3), (1, (0, 3), true));
    }

    #[test]
    fn a_missing_default_template_fails_where_nothing_is_offered() {
        let t = tokenizer_of(
            "1 1\n0 0 0\n",
            "東,0,0,3000,noun\n",
            "DEFAULT 0 1 0\nKANJI 0 0 2\n0x4E00..0x9FFF KANJI\n",
            "KANJI,0,0,8000,noun\n",
        );
        let mut out = Vec::new();
        assert_eq!(t.analyze("峠 \u{2460}", &mut out), Err(NoCandidate { offset: 4 }));
        // The stretch before the failure was segmented.
        assert_eq!(out.len(), 1);
        assert!(t.analyze("東峠", &mut Vec::new()).is_ok());
    }

    #[test]
    fn equal_paths_keep_the_one_the_java_keeps() {
        // Two entries of one surface with the same costs and context ids:
        // both candidates relax to the same cost; at the end the chain
        // holds the later one first, which the strict comparison keeps.
        let t = tokenizer_of(
            "1 1\n0 0 0\n",
            "ab,0,0,10,first\nab,0,0,10,second\n",
            "DEFAULT 0 1 0\n",
            "DEFAULT,0,0,99,unk\n",
        );
        let got = analyze(&t, "ab");
        assert_eq!(got[0].2, ["second"]);
        // a|b and ab cost the same: the end chain holds b (added after ab,
        // from a later position) first, so a|b wins.
        let t = tokenizer_of(
            "1 1\n0 0 0\n",
            "ab,0,0,10,ab\na,0,0,5,a\nb,0,0,5,b\n",
            "DEFAULT 0 1 0\n",
            "DEFAULT,0,0,99,unk\n",
        );
        assert_eq!(words(&t, "ab"), ["a", "b"]);
        // Two predecessors of equal cost: the later added one (a2, from the
        // second entry of "a") is first in the chain and is kept.
        let t = tokenizer_of(
            "1 1\n0 0 0\n",
            "a,0,0,5,a1\na,0,0,5,a2\nb,0,0,5,b\n",
            "DEFAULT 0 1 0\n",
            "DEFAULT,0,0,99,unk\n",
        );
        let got = analyze(&t, "ab");
        assert_eq!(got[0].2, ["a2"]);
    }

    #[test]
    fn connection_costs_choose_between_entries() {
        // Context 1 then 2 is cheap, 1 then 1 dear: the particle reading of
        // the second word wins although its own cost is higher.
        let matrix = "3 3\n0 0 0\n0 1 0\n0 2 0\n1 0 0\n1 1 1000\n1 2 -1000\n2 0 0\n2 1 0\n2 2 0\n";
        let t = tokenizer_of(
            matrix,
            "x,1,1,10,noun\ny,1,1,10,noun\ny,2,2,500,particle\n",
            "DEFAULT 0 1 0\n",
            "DEFAULT,0,0,99,unk\n",
        );
        assert_eq!(analyze(&t, "xy")[1].2, ["particle"]);
        // The end boundary is charged too: 2 then the end is dear.
        let matrix = "3 3\n0 0 0\n0 1 0\n0 2 0\n1 0 0\n1 1 0\n1 2 0\n2 0 5000\n2 1 0\n2 2 0\n";
        let t = tokenizer_of(
            matrix,
            "x,1,1,10,noun\ny,1,1,10,noun\ny,2,2,5,particle\n",
            "DEFAULT 0 1 0\n",
            "DEFAULT,0,0,99,unk\n",
        );
        assert_eq!(analyze(&t, "xy")[1].2, ["noun"]);
    }

    #[test]
    fn invoke_group_and_length() {
        // ALPHA invokes beside a lexicon match and groups; GREEK offers
        // lengths 1 to 3 only; KANA groups and offers lengths 1 and 2.
        let char_def = "DEFAULT 0 1 0\nALPHA 1 1 0\nGREEK 1 0 3\nKANA 0 1 2\n\
            0x0061..0x007A ALPHA\n0x03B1..0x03C9 GREEK\n0x30A1..0x30FF KANA\n";
        let unk = "DEFAULT,0,0,99,unk\nALPHA,0,0,10,alpha\nGREEK,0,0,10,greek\nKANA,0,0,10,kana\n";
        // "ab" listed dear: the grouped unknown "abc" (cost 10) beats it.
        let t = tokenizer_of("1 1\n0 0 0\n", "ab,0,0,5000,listed\n", char_def, unk);
        assert_eq!(words(&t, "abc"), ["abc"]);
        // GREEK: no group, at most three letters a morpheme.
        assert_eq!(words(&t, "αβγδε"), ["αβγ", "δε"]);
        // KANA groups; with a listed match and INVOKE 0 no unknown
        // candidate starts there.
        let t = tokenizer_of("1 1\n0 0 0\n", "アイ,0,0,5000,listed\n", char_def, unk);
        assert_eq!(
            analyze(&t, "アイウ").iter().map(|m| (m.0.clone(), m.3)).collect::<Vec<_>>(),
            [("アイ".to_string(), false), ("ウ".to_string(), true)]
        );
    }

    #[test]
    fn a_run_continues_while_neighbours_share_a_category() {
        // The mark belongs to both scripts, so x ー y is one run: the run
        // from x reaches y. It does not reach z, which shares nothing with
        // y.
        let char_def = "DEFAULT 0 1 0\nA 0 1 0\nB 0 1 0\nC 0 1 0\n0x0078 A\n0x0079 B\n0x007A C\n0x30FC A B\n";
        let unk = "DEFAULT,0,0,99,unk\nA,0,0,10,a\nB,0,0,10,b\nC,0,0,10,c\n";
        let t = tokenizer_of("1 1\n0 0 0\n", "q,0,0,1,q\n", char_def, unk);
        assert_eq!(words(&t, "xーyz"), ["xーy", "z"]);
        assert_eq!(words(&t, "yーx"), ["yーx"]);
    }

    #[test]
    fn supplementary_characters_are_whole() {
        let char_def = "DEFAULT 0 1 0\nKANJI 0 0 2\nEMOJI 0 1 0\n0x20000..0x2A6DF KANJI\n0x1F600..0x1F64F EMOJI\n";
        let unk = "DEFAULT,0,0,99,unk\nKANJI,0,0,10,kanji\nEMOJI,0,0,10,emoji\n";
        let t = tokenizer_of("1 1\n0 0 0\n", "𠮷野,0,0,1,listed\n", char_def, unk);
        let got = analyze(&t, "𠮷野😀😃𠀀𠀁𠀂");
        assert_eq!(got.iter().map(|m| m.1).collect::<Vec<_>>(), [(0, 7), (7, 15), (15, 23), (23, 27)]);
    }

    #[test]
    fn scratch_reuse_gives_the_same_morphemes() {
        let t = tokenizer();
        let mut scratch = Scratch::new();
        let texts = ["東京都に行く", "x", "峠道 ABCに行く", "", "①②③ 東"];
        for text in texts.iter().chain(texts.iter().rev()) {
            let mut fresh = Vec::new();
            t.analyze(text, &mut fresh).unwrap();
            let mut reused = Vec::new();
            t.analyze_with(text, &mut scratch, &mut reused).unwrap();
            assert_eq!(reused, fresh, "{text}");
            let mut spans = vec![Span::new(1, 2)];
            t.tokenize_with(text, &mut scratch, &mut spans).unwrap();
            assert_eq!(spans[1..], fresh.iter().map(|m| m.span).collect::<Vec<_>>()[..]);
        }
    }

    #[test]
    fn every_byte_is_covered_once_outside_whitespace() {
        let t = tokenizer();
        let text = "a\u{e9}東京\u{1f600}\u{20bb7}峠 \t\u{3000}に行くx\u{301}\u{85}①";
        let mut out = Vec::new();
        t.analyze(text, &mut out).unwrap();
        let mut at = 0usize;
        for m in &out {
            let (a, b) = (m.span.start as usize, m.span.end as usize);
            assert!(text.is_char_boundary(a) && text.is_char_boundary(b) && a < b);
            assert!(text[at..a].chars().all(|c| WhitespaceMode::Unicode.is_whitespace(c as u32)));
            assert!(!text[a..b].chars().any(|c| WhitespaceMode::Unicode.is_whitespace(c as u32)));
            at = b;
        }
        assert_eq!(at, text.len());
    }
}
