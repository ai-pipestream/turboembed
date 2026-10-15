//! One MASC document: `MascDocument.parseDocument` and what it builds.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_jvm::hash_map::put_order;
use turbo_maxent::xml::{self, Attribute, Handler, Names};

use super::error::{AnnotationFile, MascError};
use super::java::{decode_utf8, equals_ignore_case_ascii, parse_int, split_on_spaces, split_space};

/// The bytes of one document's files. The text is required; a document
/// whose header names no segmentation or sentence file fails as OpenNLP
/// fails on it ([`MascError::NoAnnotationStream`]), and one without a
/// Penn or named entity file has no tokens or names.
#[derive(Copy, Clone, Debug, Default)]
pub struct MascFiles<'b> {
    /// The primary data, `f.text`.
    pub text: &'b [u8],
    /// The quarks, `f.seg`.
    pub segments: Option<&'b [u8]>,
    /// The sentence regions, `f.s`.
    pub sentences: Option<&'b [u8]>,
    /// The Penn tokens, `f.penn`.
    pub penn: Option<&'b [u8]>,
    /// The named entities, `f.ne`.
    pub named_entities: Option<&'b [u8]>,
}

/// A quark (`MascWord`), MASC's smallest unit of text: a region of the
/// segmentation file, in UTF-16 units of the text.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Quark {
    /// The number of the region's `seg-r` id.
    pub id: i32,
    pub start: i32,
    pub end: i32,
}

/// A Penn token (`MascToken`): one or more quarks, from the start of its
/// first to the end of its last (UTF-16 units), with its tag and base
/// form when the Penn file gives them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascToken {
    /// The number of the token's `penn-n` id.
    pub id: i32,
    pub start: i32,
    pub end: i32,
    /// The `msd` feature.
    pub tag: Option<String>,
    /// The `base` feature.
    pub base: Option<String>,
    /// The token's quarks, as places in [`MascDocument::quarks`], in the
    /// order its link lists them.
    pub quarks: Vec<usize>,
}

/// A named entity of a sentence: tokens `start..end` of the sentence's
/// Penn tokens, as OpenNLP computes them (see the module documentation),
/// and the entity's label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascName {
    pub start: u32,
    pub end: u32,
    pub kind: Option<String>,
}

/// A named entity of a sentence as the named entity file gives it, which
/// OpenNLP does not compute: the sentence's Penn tokens the entity links
/// to (places in the sentence's tokens, ascending, each once), the span
/// from its first token to one past its last, and its label.
///
/// This is the reading a corpus user would expect where OpenNLP's names
/// ([`MascSentence::names`]) lose tokens and whole names (see the module
/// documentation). An entity that links to tokens of several sentences
/// is an entity of each, with the tokens in that sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascEntity {
    /// The first token.
    pub start: u32,
    /// One past the last token.
    pub end: u32,
    pub kind: Option<String>,
    pub tokens: Vec<u32>,
}

impl MascEntity {
    /// Whether the entity's tokens are every token from its start to its
    /// end, with no gap.
    pub fn is_contiguous(&self) -> bool {
        self.tokens.len() as u32 == self.end - self.start
    }
}

/// A sentence (`MascSentence`): a region of the sentence file and the
/// quarks that overlap it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascSentence {
    /// The region, in UTF-16 units of the text.
    pub start: i32,
    pub end: i32,
    quarks: Vec<usize>,
    /// Quark id to its place in the document, for the quarks of this
    /// sentence; a repeated id keeps its last quark, as `HashMap.put` does.
    quark_places: HashMap<i32, usize>,
    tokens: Option<Vec<MascToken>>,
    /// Token id to its place in `tokens`.
    token_places: HashMap<i32, usize>,
    names: Vec<MascName>,
    entities: Vec<MascEntity>,
}

impl MascSentence {
    fn new(start: i32, end: i32, quarks: Vec<usize>, all: &[Quark]) -> MascSentence {
        let quark_places = quarks.iter().map(|&q| (all[q].id, q)).collect();
        MascSentence {
            start,
            end,
            quarks,
            quark_places,
            tokens: None,
            token_places: HashMap::new(),
            names: Vec::new(),
            entities: Vec::new(),
        }
    }

    /// The quarks that overlap the region, as places in
    /// [`MascDocument::quarks`].
    pub fn quarks(&self) -> &[usize] {
        &self.quarks
    }

    /// The Penn tokens, `None` when the document has no Penn file.
    pub fn tokens(&self) -> Option<&[MascToken]> {
        self.tokens.as_deref()
    }

    /// The named entities as OpenNLP computes them, which is what
    /// [`MascNameReader`](super::MascNameReader) reads.
    pub fn names(&self) -> &[MascName] {
        &self.names
    }

    /// The named entities as the named entity file gives them, in the
    /// order of their first edge in the file; see [`MascEntity`].
    pub fn entities(&self) -> &[MascEntity] {
        &self.entities
    }
}

/// Something OpenNLP logs as a warning and reads on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MascWarning {
    /// A quark is in more than one Penn token.
    QuarkInSeveralTokens { quark: i32 },
    /// A Penn token of sentence `sentence` has a quark outside it.
    TokenCrossesSentence { sentence: usize, token: i32, quark: i32 },
    /// A named entity file names a token that an earlier entity of
    /// another type named.
    TokenInSeveralTypes { token: i32 },
    /// Named entities of sentence `sentence` overlap, and some were removed.
    OverlappingNames { sentence: usize },
    /// The document has named entities and no Penn tokens, so it has no
    /// names.
    NamesWithoutTokens,
}

/// The UTF-16 offsets of a text: the byte offset of each unit.
#[derive(Clone, Debug, PartialEq, Eq)]
struct TextIndex {
    /// For each unit, and for the end of the text, its byte offset;
    /// `INSIDE` for the second unit of a surrogate pair.
    bytes: Vec<u32>,
}

const INSIDE: u32 = u32::MAX;

impl TextIndex {
    fn new(text: &str) -> TextIndex {
        let mut bytes = Vec::with_capacity(text.len() + 1);
        for (i, c) in text.char_indices() {
            bytes.push(i as u32);
            if c.len_utf16() == 2 {
                bytes.push(INSIDE);
            }
        }
        bytes.push(text.len() as u32);
        TextIndex { bytes }
    }

    fn units(&self) -> usize {
        self.bytes.len() - 1
    }

    /// The bytes of `String.substring(begin, end)`.
    fn substring(&self, begin: i64, end: i64) -> Result<Span, MascError> {
        let len = self.units();
        if begin < 0 || begin > end || end > len as i64 {
            return Err(MascError::Substring { begin, end, len });
        }
        let byte = |u: i64| match self.bytes[u as usize] {
            INSIDE => Err(MascError::SplitsSurrogatePair { unit: u }),
            b => Ok(b),
        };
        Ok(Span::new(byte(begin)?, byte(end)?))
    }
}

/// A MASC document: the text, its quarks, and its sentences with their
/// Penn tokens and named entities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascDocument {
    path: PathBuf,
    text: String,
    index: TextIndex,
    quarks: Vec<Quark>,
    sentences: Vec<MascSentence>,
    has_penn_tags: bool,
    has_named_entities: bool,
    warnings: Vec<MascWarning>,
}

impl MascDocument {
    /// `MascDocument.parseDocument`: reads the text (UTF-8, each malformed
    /// sequence replaced with U+FFFD where Java's decoder replaces it), the
    /// quarks and the sentence regions, gives each sentence the quarks
    /// that overlap it, then adds the Penn tokens and the named entities
    /// when there are files for them. `path` is the document's header,
    /// which only names it.
    ///
    /// # Errors
    ///
    /// What `parseDocument` throws: see [`MascError`].
    pub fn parse(path: impl Into<PathBuf>, files: MascFiles<'_>) -> Result<MascDocument, MascError> {
        let text = decode_utf8(files.text);
        if u32::try_from(text.len()).is_err() {
            return Err(MascError::TooLong { len: text.len() });
        }
        let segments = files.segments.ok_or(MascError::NoAnnotationStream { file: AnnotationFile::Segments })?;
        let quarks = parse_segments(segments)?;
        let sentence_file = files.sentences.ok_or(MascError::NoAnnotationStream { file: AnnotationFile::Sentences })?;
        let regions = parse_sentences(sentence_file)?;
        let sentences = combine(&regions, &quarks)?;
        let index = TextIndex::new(&text);
        let mut doc = MascDocument {
            path: path.into(),
            text,
            index,
            quarks,
            sentences,
            has_penn_tags: false,
            has_named_entities: false,
            warnings: Vec::new(),
        };
        if let Some(penn) = files.penn {
            let penn = parse_penn(penn)?;
            doc.add_penn_tags(&penn)?;
        }
        if let Some(ne) = files.named_entities {
            let ne = parse_named_entities(ne)?;
            doc.warnings.extend(ne.warnings.iter().cloned());
            doc.add_named_entities(&ne);
        }
        Ok(doc)
    }

    /// The header the document was read from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// The text's length in UTF-16 units, the units of every offset of
    /// the annotation.
    pub fn text_units(&self) -> usize {
        self.index.units()
    }

    /// The quarks in the order of the segmentation file.
    pub fn quarks(&self) -> &[Quark] {
        &self.quarks
    }

    /// The sentences in the order of the sentence file, without the
    /// regions OpenNLP drops.
    pub fn sentences(&self) -> &[MascSentence] {
        &self.sentences
    }

    /// Whether the Penn tokens were added (`hasPennTags`).
    pub fn has_penn_tags(&self) -> bool {
        self.has_penn_tags
    }

    /// Whether the named entities were added (`hasNamedEntities`).
    pub fn has_named_entities(&self) -> bool {
        self.has_named_entities
    }

    /// What OpenNLP logged while reading the document.
    pub fn warnings(&self) -> &[MascWarning] {
        &self.warnings
    }

    /// The bytes of the text between UTF-16 offsets `begin` and `end`, as
    /// `String.substring` takes them.
    ///
    /// # Errors
    ///
    /// [`MascError::Substring`] where `substring` throws, and
    /// [`MascError::SplitsSurrogatePair`] for an offset inside a pair.
    pub fn span(&self, begin: i64, end: i64) -> Result<Span, MascError> {
        self.index.substring(begin, end)
    }

    fn add_penn_tags(&mut self, penn: &Penn) -> Result<(), MascError> {
        // Each quark's tokens, built in the order the token map iterates,
        // each token put in front of the ones before it.
        let mut quark_tokens: HashMap<i32, Vec<usize>> = HashMap::new();
        for t in put_order(penn.tokens.iter().map(|t| t.id)).0 {
            for &q in &penn.tokens[t].quarks {
                let tokens = quark_tokens.entry(q).or_default();
                if !tokens.is_empty() {
                    self.warnings.push(MascWarning::QuarkInSeveralTokens { quark: q });
                }
                tokens.insert(0, t);
            }
        }
        for si in 0..self.sentences.len() {
            let tokens = tokenize(si, &self.sentences[si], &self.quarks, penn, &quark_tokens, &mut self.warnings)?;
            let sentence = &mut self.sentences[si];
            sentence.token_places = tokens.iter().enumerate().map(|(i, t)| (t.id, i)).collect();
            sentence.tokens = Some(tokens);
        }
        self.has_penn_tags = true;
        Ok(())
    }

    fn add_named_entities(&mut self, ne: &NamedEntities) {
        if !self.has_penn_tags && !self.sentences.is_empty() {
            // the first sentence throws "Named entity labels provided for an
            // un-tokenized sentence.", which the document logs
            self.warnings.push(MascWarning::NamesWithoutTokens);
            return;
        }
        for si in 0..self.sentences.len() {
            let sentence = &mut self.sentences[si];
            let count = sentence.tokens.as_ref().map_or(0, Vec::len);
            let mut names = Vec::new();
            for &e in &ne.order {
                let entity = &ne.entities[e];
                let kind = ne.types.get(&entity.id).cloned().flatten();
                let mut start = count;
                let mut end = 0;
                let mut inside = false;
                for token in &entity.tokens {
                    if let Some(&i) = sentence.token_places.get(token) {
                        inside = true;
                        if i < start {
                            start = i;
                        }
                        // compared with the end so far, which is one past a
                        // token: a name of the tokens 3 and 4 ends at 4
                        if i > end {
                            end = i + 1;
                        }
                    }
                }
                if inside {
                    names.push(MascName { start: start as u32, end: end as u32, kind });
                }
            }
            names.sort_by_key(|n| n.start);
            if let Some(kept) = remove_overlaps(&names) {
                self.warnings.push(MascWarning::OverlappingNames { sentence: si });
                names = kept;
            }
            sentence.names = names;
            let mut entities = Vec::new();
            for entity in &ne.entities {
                let mut tokens: Vec<u32> =
                    entity.tokens.iter().filter_map(|t| sentence.token_places.get(t)).map(|&i| i as u32).collect();
                tokens.sort_unstable();
                tokens.dedup();
                let (Some(&first), Some(&last)) = (tokens.first(), tokens.last()) else {
                    continue;
                };
                let kind = ne.types.get(&entity.id).cloned().flatten();
                entities.push(MascEntity { start: first, end: last + 1, kind, tokens });
            }
            sentence.entities = entities;
        }
        self.has_named_entities = true;
    }

    /// `getSentDetectText`: the bytes of the sentence's region.
    pub(crate) fn sentence_text(&self, sentence: &MascSentence) -> Result<Span, MascError> {
        self.span(i64::from(sentence.start), i64::from(sentence.end))
    }

    /// `getTokenText`: the bytes from the start of the first token to the
    /// end of the last, or `None` for a sentence without tokens.
    pub(crate) fn token_text(&self, sentence: &MascSentence) -> Result<Option<(Span, i32)>, MascError> {
        let tokens = sentence.tokens.as_deref().ok_or(MascError::Untokenized { method: "isEmpty" })?;
        match (tokens.first(), tokens.last()) {
            (Some(first), Some(last)) => {
                Ok(Some((self.span(i64::from(first.start), i64::from(last.end))?, first.start)))
            }
            _ => Ok(None),
        }
    }

    /// `getTokenStrings`: the bytes of each Penn token of `sentence`, a
    /// sentence of this document, in [`MascDocument::text`].
    ///
    /// # Errors
    ///
    /// [`MascError::Untokenized`] for a document without a Penn file, and
    /// what an offset outside the text or inside a surrogate pair gives.
    pub fn token_spans(&self, sentence: &MascSentence) -> Result<Vec<Span>, MascError> {
        let tokens = sentence.tokens.as_deref().ok_or(MascError::Untokenized { method: "iterator" })?;
        tokens.iter().map(|t| self.span(i64::from(t.start), i64::from(t.end))).collect()
    }
}

/// The list of names `addNamedEntities` keeps of names sorted by start,
/// or `None` when it finds no overlap and keeps them all.
///
/// It walks a left and a right name: when they overlap (the left contains
/// or crosses the right) it marks the shorter (the right on a tie) and
/// moves the right on; otherwise it moves the left on, until the left
/// reaches the right and finds that a name contains itself. Then, if
/// anything was marked, it keeps the unmarked names but the last.
fn remove_overlaps(names: &[MascName]) -> Option<Vec<MascName>> {
    let contains = |a: &MascName, b: &MascName| a.start <= b.start && b.end <= a.end;
    let crosses = |a: &MascName, b: &MascName| {
        !contains(a, b)
            && !contains(b, a)
            && (a.start <= b.start && b.start < a.end || b.start <= a.start && a.start < b.end)
    };
    let mut marked = HashSet::new();
    let mut overlaps = false;
    let mut left = 0;
    let mut right = 1;
    while right < names.len() {
        let (l, r) = (&names[left], &names[right]);
        if contains(l, r) || crosses(l, r) {
            if r.end - r.start > l.end - l.start {
                marked.insert(left);
            } else {
                marked.insert(right);
            }
            overlaps = true;
            right += 1;
        } else {
            left += 1;
        }
    }
    overlaps.then(|| {
        names[..names.len() - 1]
            .iter()
            .enumerate()
            .filter(|(i, _)| !marked.contains(i))
            .map(|(_, n)| n.clone())
            .collect()
    })
}

/// `MascSentence.tokenizePenn` for sentence `si`: the tokens of its
/// quarks, in the order of the quarks and, for a quark in several tokens,
/// of the quark's token list; a token is taken once, and a token whose
/// quarks span nothing is left out.
fn tokenize(
    si: usize,
    sentence: &MascSentence,
    quarks: &[Quark],
    penn: &Penn,
    quark_tokens: &HashMap<i32, Vec<usize>>,
    warnings: &mut Vec<MascWarning>,
) -> Result<Vec<MascToken>, MascError> {
    let mut tokens = Vec::new();
    let mut processed = HashSet::new();
    for &w in &sentence.quarks {
        let Some(list) = quark_tokens.get(&quarks[w].id) else {
            continue;
        };
        for &t in list {
            if processed.contains(&t) {
                continue;
            }
            let token = &penn.tokens[t];
            for &q in &token.quarks {
                if !sentence.quark_places.contains_key(&q) {
                    warnings.push(MascWarning::TokenCrossesSentence { sentence: si, token: token.id, quark: q });
                }
            }
            let mut places = Vec::with_capacity(token.quarks.len());
            for &q in &token.quarks {
                // the sentence's own quark, else the first in the document
                let place = match sentence.quark_places.get(&q) {
                    Some(&p) => p,
                    None => quarks.iter().position(|x| x.id == q).ok_or(MascError::MissingQuark { id: q })?,
                };
                places.push(place);
            }
            let (Some(&first), Some(&last)) = (places.first(), places.last()) else {
                continue;
            };
            let start = quarks[first].start;
            let end = quarks[last].end;
            if end - start > 0 {
                tokens.push(MascToken {
                    id: token.id,
                    start,
                    end,
                    tag: penn.tags.get(&token.id).cloned().flatten(),
                    base: penn.bases.get(&token.id).cloned().flatten(),
                    quarks: places,
                });
                processed.insert(t);
            }
        }
    }
    Ok(tokens)
}

/// `combineAnnotations`: each non-empty region with the quarks that
/// overlap it. The quarks are walked once, in file order, stepping back
/// at the start of a region over the quarks that end after it starts.
fn combine(regions: &[(i32, i32)], quarks: &[Quark]) -> Result<Vec<MascSentence>, MascError> {
    let mut wi = 0usize;
    let count = quarks.len();
    let mut sentences = Vec::new();
    for &(start, end) in regions {
        if end - start <= 0 {
            continue;
        }
        let mut list = Vec::new();
        let mut next = quarks.get(wi).ok_or(MascError::IndexOutOfBounds { index: wi as i64, len: count })?;
        while start < next.end && wi > 0 {
            wi -= 1;
            next = &quarks[wi];
        }
        let mut over = false;
        while !over && wi < count {
            let q = &quarks[wi];
            if end > q.start && start < q.end {
                list.push(wi);
                if end == q.end {
                    over = true;
                }
                wi += 1;
            } else if end <= q.start {
                over = true;
            } else {
                wi += 1;
            }
        }
        // counts the sentences made against the regions, empty ones too
        if !over && sentences.len() != regions.len() - 1 {
            return Err(MascError::QuarksEndInSentence { end });
        }
        sentences.push(MascSentence::new(start, end, list, quarks));
    }
    Ok(sentences)
}

// ---- the annotation files ----

fn bad(offset: usize, message: &'static str) -> turbo_maxent::Error {
    turbo_maxent::Error::Xml { offset, message }
}

fn attr<'x>(attributes: &'x [Attribute<'_>], name: &str) -> Option<&'x str> {
    attributes.iter().find(|a| a.name == name).map(|a| a.value.as_ref())
}

/// `MascIdentifiers.parseId`: `prefix` and one or more ASCII digits, in
/// the `int` range.
fn parse_id(id: Option<&str>, prefix: &str) -> Option<i32> {
    let digits = id?.strip_prefix(prefix)?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<i32>().ok()
}

/// `MascIdentifiers.parseIds`: one or more ids separated by spaces.
fn parse_ids(ids: Option<&str>, prefix: &str) -> Option<Vec<i32>> {
    let items = split_on_spaces(ids?);
    if items.is_empty() {
        return None;
    }
    items.into_iter().map(|i| parse_id(Some(i), prefix)).collect()
}

/// Whether `Span(start, end)` accepts the offsets.
fn valid_span(start: i32, end: i32) -> bool {
    start >= 0 && end >= 0 && start <= end
}

fn read_xml(bytes: &[u8], file: AnnotationFile, handler: &mut impl Handler) -> Result<(), MascError> {
    let fail = |cause| MascError::Annotation { file, cause };
    let text = xml::decode(bytes).map_err(fail)?;
    xml::parse_with(&text, Names::Qualified, handler).map_err(fail)
}

/// `MascWordParser`: every `region` element (any case) is a quark.
struct Segments(Vec<Quark>);

impl Handler for Segments {
    fn start_element(&mut self, name: &str, a: &[Attribute<'_>], at: usize) -> turbo_maxent::Result<()> {
        if !equals_ignore_case_ascii(name, "region") {
            return Ok(());
        }
        let id = parse_id(attr(a, "xml:id"), "seg-r").ok_or(bad(at, "a region id is not seg-r and digits"))?;
        let anchors = split_on_spaces(attr(a, "anchors").ok_or(bad(at, "a region has no anchors"))?);
        let [left, right] = anchors[..] else {
            return Err(bad(at, "a region's anchors are not two numbers"));
        };
        let (Some(start), Some(end)) = (parse_int(left), parse_int(right)) else {
            return Err(bad(at, "a region anchor is not an int"));
        };
        if !valid_span(start, end) {
            return Err(bad(at, "a region's anchors are not a span"));
        }
        self.0.push(Quark { id, start, end });
        Ok(())
    }

    fn end_element(&mut self, _: &str, _: usize) -> turbo_maxent::Result<()> {
        Ok(())
    }

    fn characters(&mut self, _: &str) {}
}

fn parse_segments(bytes: &[u8]) -> Result<Vec<Quark>, MascError> {
    let mut h = Segments(Vec::new());
    read_xml(bytes, AnnotationFile::Segments, &mut h)?;
    Ok(h.0)
}

/// `MascSentenceParser`: every `region` element (any case) is a sentence
/// region, the first two fields of its anchors split at each space.
struct Sentences(Vec<(i32, i32)>);

impl Handler for Sentences {
    fn start_element(&mut self, name: &str, a: &[Attribute<'_>], at: usize) -> turbo_maxent::Result<()> {
        if !equals_ignore_case_ascii(name, "region") {
            return Ok(());
        }
        let fields = split_space(attr(a, "anchors").ok_or(bad(at, "a region has no anchors"))?);
        let (Some(start), Some(end)) =
            (fields.first().and_then(|f| parse_int(f)), fields.get(1).and_then(|f| parse_int(f)))
        else {
            return Err(bad(at, "a region's first two anchors are not ints"));
        };
        if !valid_span(start, end) {
            return Err(bad(at, "a region's anchors are not a span"));
        }
        self.0.push((start, end));
        Ok(())
    }

    fn end_element(&mut self, _: &str, _: usize) -> turbo_maxent::Result<()> {
        Ok(())
    }

    fn characters(&mut self, _: &str) {}
}

/// The regions, without each one that does not end before the next
/// starts (the last is always kept).
fn parse_sentences(bytes: &[u8]) -> Result<Vec<(i32, i32)>, MascError> {
    let mut h = Sentences(Vec::new());
    read_xml(bytes, AnnotationFile::Sentences, &mut h)?;
    let anchors = h.0;
    let Some(&last) = anchors.last() else {
        return Err(MascError::IndexOutOfBounds { index: -1, len: 0 });
    };
    let mut kept: Vec<(i32, i32)> = anchors.windows(2).filter(|w| w[0].1 < w[1].0).map(|w| w[0]).collect();
    kept.push(last);
    Ok(kept)
}

/// One token of the Penn file and its quark ids.
struct PennToken {
    id: i32,
    quarks: Vec<i32>,
}

/// What `MascPennTagParser` reads.
#[derive(Default)]
struct Penn {
    /// Each token once, at its first link; a later link replaces its
    /// quarks.
    tokens: Vec<PennToken>,
    places: HashMap<i32, usize>,
    tags: HashMap<i32, Option<String>>,
    bases: HashMap<i32, Option<String>>,
    /// The ids of the `node` elements not linked yet.
    nodes: Vec<i32>,
    /// The ids of the open `a` elements.
    annotations: Vec<i32>,
}

impl Handler for Penn {
    fn start_element(&mut self, name: &str, a: &[Attribute<'_>], at: usize) -> turbo_maxent::Result<()> {
        match name {
            "node" => {
                let id = parse_id(attr(a, "xml:id"), "penn-n").ok_or(bad(at, "a node id is not penn-n and digits"))?;
                self.nodes.push(id);
            }
            "link" => {
                if self.nodes.is_empty() {
                    return Err(bad(at, "a link with no node to link"));
                }
                let quarks =
                    parse_ids(attr(a, "targets"), "seg-r").ok_or(bad(at, "a link's targets are not seg-r ids"))?;
                let id = self.nodes.pop().ok_or(bad(at, "a link with no node to link"))?;
                match self.places.get(&id) {
                    Some(&p) => self.tokens[p].quarks = quarks,
                    None => {
                        self.places.insert(id, self.tokens.len());
                        self.tokens.push(PennToken { id, quarks });
                    }
                }
            }
            "a" => {
                let id =
                    parse_id(attr(a, "ref"), "penn-n").ok_or(bad(at, "an annotation ref is not penn-n and digits"))?;
                self.annotations.push(id);
            }
            "f" => {
                let token = *self.annotations.last().ok_or(bad(at, "a feature outside an annotation"))?;
                let value = attr(a, "value").map(str::to_owned);
                match attr(a, "name").ok_or(bad(at, "a feature without a name"))? {
                    "msd" => {
                        self.tags.insert(token, value);
                    }
                    "base" => {
                        self.bases.insert(token, value);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn end_element(&mut self, name: &str, _: usize) -> turbo_maxent::Result<()> {
        if name == "a" {
            self.annotations.pop();
        }
        Ok(())
    }

    fn characters(&mut self, _: &str) {}
}

fn parse_penn(bytes: &[u8]) -> Result<Penn, MascError> {
    let mut h = Penn::default();
    read_xml(bytes, AnnotationFile::Penn, &mut h)?;
    Ok(h)
}

/// One named entity and the ids of its tokens, in edge order.
struct Entity {
    id: i32,
    tokens: Vec<i32>,
}

/// What `MascNamedEntityParser` reads.
#[derive(Default)]
struct NamedEntities {
    /// The label of each entity that has an `a` element.
    types: HashMap<i32, Option<String>>,
    /// Each entity once, at its first edge.
    entities: Vec<Entity>,
    places: HashMap<i32, usize>,
    /// The iteration order of the entity map.
    order: Vec<usize>,
    /// The type each token was last given.
    token_types: HashMap<i32, Option<String>>,
    warnings: Vec<MascWarning>,
}

impl Handler for NamedEntities {
    fn start_element(&mut self, name: &str, a: &[Attribute<'_>], at: usize) -> turbo_maxent::Result<()> {
        match name {
            "a" => {
                let id = parse_id(attr(a, "ref"), "ne-n").ok_or(bad(at, "an annotation ref is not ne-n and digits"))?;
                if self.types.contains_key(&id) {
                    return Err(bad(at, "two labels for one named entity"));
                }
                self.types.insert(id, attr(a, "label").map(str::to_owned));
            }
            "edge" => {
                let id = parse_id(attr(a, "from"), "ne-n").ok_or(bad(at, "an edge's from is not ne-n and digits"))?;
                let token =
                    parse_id(attr(a, "to"), "penn-n").ok_or(bad(at, "an edge's to is not penn-n and digits"))?;
                match self.places.get(&id) {
                    Some(&p) => self.entities[p].tokens.push(token),
                    None => {
                        self.places.insert(id, self.entities.len());
                        self.entities.push(Entity { id, tokens: vec![token] });
                    }
                }
                let kind = self.types.get(&id).cloned().flatten();
                if let Some(before) = self.token_types.get(&token) {
                    match &kind {
                        // `type.equals` on a null type: the entity has no
                        // label, or no `a` element before this edge
                        None => return Err(bad(at, "an edge of an entity without a label to a named token")),
                        Some(k) if before.as_deref() != Some(k.as_str()) => {
                            self.warnings.push(MascWarning::TokenInSeveralTypes { token });
                        }
                        Some(_) => {}
                    }
                }
                self.token_types.insert(token, kind);
            }
            _ => {}
        }
        Ok(())
    }

    fn end_element(&mut self, _: &str, _: usize) -> turbo_maxent::Result<()> {
        Ok(())
    }

    fn characters(&mut self, _: &str) {}
}

fn parse_named_entities(bytes: &[u8]) -> Result<NamedEntities, MascError> {
    let mut h = NamedEntities::default();
    read_xml(bytes, AnnotationFile::NamedEntities, &mut h)?;
    h.order = put_order(h.entities.iter().map(|e| e.id)).0;
    Ok(h)
}

/// A file a document header names: an `annotation` or `primaryData`
/// element's `f.id` and `loc`, either of which may be missing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct HeaderFile {
    pub(crate) id: Option<String>,
    pub(crate) loc: Option<String>,
}

/// The files of a header (`MascDocumentStream.HeaderHandler`), one per
/// `f.id` (a later element with the same `f.id` replaces the `loc`), in the
/// iteration order of its map, or `None` for a header without an
/// `annotation` or `primaryData` element.
pub(crate) fn parse_header(text: &str) -> Result<Option<Vec<HeaderFile>>, turbo_maxent::Error> {
    #[derive(Default)]
    struct Header {
        files: Option<Vec<HeaderFile>>,
        loc: Option<String>,
        id: Option<String>,
    }
    fn file_element(name: &str) -> bool {
        equals_ignore_case_ascii(name, "annotation") || equals_ignore_case_ascii(name, "primaryData")
    }
    impl Handler for Header {
        fn start_element(&mut self, name: &str, a: &[Attribute<'_>], _: usize) -> turbo_maxent::Result<()> {
            if file_element(name) {
                self.loc = attr(a, "loc").map(str::to_owned);
                self.id = attr(a, "f.id").map(str::to_owned);
                self.files.get_or_insert_with(Vec::new);
            }
            Ok(())
        }

        fn end_element(&mut self, name: &str, _: usize) -> turbo_maxent::Result<()> {
            if file_element(name) {
                let files = self.files.get_or_insert_with(Vec::new);
                match files.iter_mut().find(|f| f.id == self.id) {
                    Some(file) => file.loc = self.loc.clone(),
                    None => {
                        files.push(HeaderFile { id: self.id.clone(), loc: self.loc.clone() });
                    }
                }
            }
            Ok(())
        }

        fn characters(&mut self, _: &str) {}
    }
    let mut h = Header::default();
    xml::parse_with(text, Names::Qualified, &mut h)?;
    Ok(h.files.map(|mut files| {
        let (order, _) = put_order(files.iter().map(|f| f.id.as_deref()));
        order.into_iter().map(|i| std::mem::take(&mut files[i])).collect()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(start: u32, end: u32) -> MascName {
        MascName { start, end, kind: None }
    }

    fn spans(names: &[MascName]) -> Vec<(u32, u32)> {
        names.iter().map(|n| (n.start, n.end)).collect()
    }

    #[test]
    fn one_name_is_kept() {
        assert_eq!(remove_overlaps(&[name(0, 2)]), None);
        assert_eq!(remove_overlaps(&[]), None);
    }

    #[test]
    fn names_apart_lose_all_but_the_first() {
        // the left index reaches the right, the name contains itself and is
        // marked, and the last name is dropped as well
        let kept = remove_overlaps(&[name(0, 1), name(2, 3), name(4, 6)]).unwrap();
        assert_eq!(spans(&kept), [(0, 1)]);
    }

    #[test]
    fn the_shorter_of_two_overlapping_names_is_marked() {
        // (1, 4) contains (2, 3): the right is marked, then the left is
        // compared with (5, 6), which moves it on to itself
        let kept = remove_overlaps(&[name(1, 4), name(2, 3), name(5, 6), name(7, 8)]).unwrap();
        assert_eq!(spans(&kept), [(1, 4)]);
        // the longer name on the right marks the left one
        let kept = remove_overlaps(&[name(0, 2), name(1, 5), name(9, 9)]).unwrap();
        assert_eq!(spans(&kept), [(1, 5)]);
    }

    #[test]
    fn two_names_over_the_same_tokens_can_both_survive() {
        // next to each other, the right copy is marked
        let kept = remove_overlaps(&[name(0, 2), name(0, 2)]).unwrap();
        assert_eq!(spans(&kept), [(0, 2)]);
        // a shorter name before the copies crosses both and is marked twice,
        // so the walk moves the right index past the copies without
        // comparing them with each other; both stay, and `NameSample`
        // refuses them
        let kept = remove_overlaps(&[name(0, 2), name(1, 4), name(1, 4), name(9, 10)]).unwrap();
        assert_eq!(spans(&kept), [(1, 4), (1, 4)]);
    }

    #[test]
    fn a_name_ends_one_past_a_token_only_when_past_the_end_so_far() {
        let doc = |penn: &str, ne: &str| {
            let text = b"a b c d e";
            let seg = br#"<graph><region xml:id="seg-r0" anchors="0 1"/><region xml:id="seg-r1" anchors="2 3"/><region xml:id="seg-r2" anchors="4 5"/><region xml:id="seg-r3" anchors="6 7"/><region xml:id="seg-r4" anchors="8 9"/></graph>"#;
            let s = br#"<graph><region anchors="0 9"/></graph>"#;
            MascDocument::parse(
                "t.hdr",
                MascFiles {
                    text,
                    segments: Some(seg),
                    sentences: Some(s),
                    penn: Some(penn.as_bytes()),
                    named_entities: Some(ne.as_bytes()),
                },
            )
            .unwrap()
        };
        let mut penn = String::from("<graph>");
        for i in 0..5 {
            penn.push_str(&format!(r#"<node xml:id="penn-n{i}"><link targets="seg-r{i}"/></node>"#));
        }
        penn.push_str("</graph>");
        let names = |edges: &[u32]| {
            let mut ne = String::from(r#"<graph><a ref="ne-n0" label="x"/>"#);
            for e in edges {
                ne.push_str(&format!(r#"<edge from="ne-n0" to="penn-n{e}"/>"#));
            }
            ne.push_str("</graph>");
            spans(doc(&penn, &ne).sentences()[0].names())
        };
        assert_eq!(names(&[3]), [(3, 4)]);
        // tokens 3 and 4: 4 is not past the end 4 the token 3 gave
        assert_eq!(names(&[3, 4]), [(3, 4)]);
        assert_eq!(names(&[1, 3]), [(1, 4)]);
        // the first token alone is empty
        assert_eq!(names(&[0]), [(0, 0)]);
        assert_eq!(names(&[4, 2]), [(2, 5)]);
    }

    #[test]
    fn an_entity_spans_its_first_token_to_one_past_its_last_and_none_is_removed() {
        let text = b"a b c d e. f g";
        let mut seg = String::from("<graph>");
        let mut penn = String::from("<graph>");
        for (i, at) in [0, 2, 4, 6, 8, 11, 13].iter().enumerate() {
            seg.push_str(&format!(r#"<region xml:id="seg-r{i}" anchors="{at} {}"/>"#, at + 1));
            penn.push_str(&format!(r#"<node xml:id="penn-n{i}"><link targets="seg-r{i}"/></node>"#));
        }
        seg.push_str("</graph>");
        penn.push_str("</graph>");
        let s = br#"<graph><region anchors="0 10"/><region anchors="11 14"/></graph>"#;
        // ne-n0: tokens 0 and 1; ne-n1: token 0, inside ne-n0; ne-n2: tokens
        // 4 and 2, with a gap; ne-n3: tokens 4 and 5, across the sentence
        // end; ne-n4: no label.
        let ne = r#"<graph><a ref="ne-n0" label="person"/><a ref="ne-n1" label="org"/><a ref="ne-n2" label="date"/>
            <a ref="ne-n3" label="location"/>
            <edge from="ne-n0" to="penn-n1"/><edge from="ne-n0" to="penn-n0"/><edge from="ne-n1" to="penn-n0"/>
            <edge from="ne-n2" to="penn-n4"/><edge from="ne-n2" to="penn-n2"/>
            <edge from="ne-n3" to="penn-n4"/><edge from="ne-n3" to="penn-n5"/><edge from="ne-n4" to="penn-n6"/></graph>"#;
        let doc = MascDocument::parse(
            "t.hdr",
            MascFiles {
                text,
                segments: Some(seg.as_bytes()),
                sentences: Some(s),
                penn: Some(penn.as_bytes()),
                named_entities: Some(ne.as_bytes()),
            },
        )
        .unwrap();
        let entity = |start, end, kind: Option<&str>, tokens: &[u32]| MascEntity {
            start,
            end,
            kind: kind.map(str::to_owned),
            tokens: tokens.to_vec(),
        };
        let first = doc.sentences()[0].entities();
        assert_eq!(
            first,
            [
                entity(0, 2, Some("person"), &[0, 1]),
                entity(0, 1, Some("org"), &[0]),
                entity(2, 5, Some("date"), &[2, 4]),
                entity(4, 5, Some("location"), &[4]),
            ]
        );
        assert!(first[0].is_contiguous() && !first[2].is_contiguous());
        // OpenNLP's names of the same sentence lose most of this.
        assert_ne!(spans(doc.sentences()[0].names()), [(0, 2), (0, 1), (2, 5), (4, 5)]);
        assert_eq!(doc.sentences()[1].entities(), [entity(0, 1, Some("location"), &[0]), entity(1, 2, None, &[1])]);
    }
}
