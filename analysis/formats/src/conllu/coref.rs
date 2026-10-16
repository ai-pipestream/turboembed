//! Gold coreference chains and speakers read from CoNLL-U `Entity=...`
//! and `# speaker =` annotations:
//! `opennlp.tools.formats.conllu.ConlluCorefDocumentStream` and
//! `opennlp.tools.coref.CorefMention`.
//!
//! Each `# newdoc` comment starts a document; a file without one is a
//! single document, and a `# newdoc` block without token rows is skipped.
//! The document text is rebuilt from the word forms, separated by single
//! spaces except after a token marked `SpaceAfter=No` in `MISC`, with
//! sentences separated by `\n`. Multiword token ranges and empty nodes
//! are validated and skipped.
//!
//! Every returned [`Document`] carries:
//! - [`Sentences`] under [`Key::DEFAULT`],
//! - [`Tokens`] under [`Key::DEFAULT`],
//! - [`PosTags`] from the chosen [`Tagset`] under [`Key::DEFAULT`],
//! - [`CorefChains`] under [`GOLD_CHAINS`] (`Key::named("gold")`), and
//! - when `# speaker =` comments are present, [`Speakers`] under
//!   [`SPEAKERS`] (`Key::DEFAULT`).

use std::collections::HashMap;

use turbo_analysis::{Document, Ix, Key, Layer, Overlap, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{PosTags, Sentences, Strs, Tokens};

use super::Tagset;
use super::dependency::split_tab_all;
use crate::error::FormatError;
use crate::sample::dependency::is_blank;
use crate::source::{Line, Lines, java_trim};

const NEWDOC: &str = "# newdoc";
const SPEAKER: &str = "# speaker =";
const ENTITY: &str = "Entity";
const SPACE_AFTER: &str = "SpaceAfter";

/// The key for gold coreference chains (`CorefAnnotator.GOLD_CHAINS`,
/// `"opennlp:chains#gold"`).
pub const GOLD_CHAINS: Key<CorefChains> = Key::named("gold");

/// The key for the speakers layer (`CorefAnnotator.SPEAKERS`,
/// `"opennlp:speakers"`).
pub const SPEAKERS: Key<Speakers> = Key::DEFAULT;

/// One mention in a coreference chain (`opennlp.tools.coref.CorefMention`).
///
/// Chain identifiers start at zero in first-mention order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CorefMention {
    pub chain: u32,
    pub kind: String,
    pub entity: i32,
}

impl CorefMention {
    /// A mention lifted from the entity layer (`CorefMention.KIND_ENTITY`).
    pub const KIND_ENTITY: &'static str = "entity";
    /// A noun phrase mention lifted from the chunk layer (`CorefMention.KIND_NOMINAL`).
    pub const KIND_NOMINAL: &'static str = "nominal";
    /// A pronoun mention (`CorefMention.KIND_PRONOUN`).
    pub const KIND_PRONOUN: &'static str = "pronoun";
    /// A mention read from a corpus annotation (`CorefMention.KIND_GOLD`).
    pub const KIND_GOLD: &'static str = "gold";
    /// The entity index of a mention not backed by an entity (`CorefMention.NO_ENTITY`).
    pub const NO_ENTITY: i32 = -1;

    /// Validates and constructs a [`CorefMention`].
    ///
    /// # Errors
    ///
    /// [`FormatError::ConlluCoref`] when `chain < 0`, `kind` is blank, or
    /// `entity < NO_ENTITY`.
    pub fn new(chain: i32, kind: impl Into<String>, entity: i32) -> Result<CorefMention, FormatError> {
        if chain < 0 {
            return Err(FormatError::ConlluCoref { line: 0, message: format!("chain must not be negative: {chain}") });
        }
        let kind = kind.into();
        if is_blank(&kind) {
            return Err(FormatError::ConlluCoref { line: 0, message: "kind must not be null or blank".into() });
        }
        if entity < Self::NO_ENTITY {
            return Err(FormatError::ConlluCoref {
                line: 0,
                message: format!("entity must be an index or NO_ENTITY: {entity}"),
            });
        }
        Ok(CorefMention { chain: chain as u32, kind, entity })
    }
}

/// Coreference mentions over a document (`CorefAnnotator.CHAINS`,
/// `"opennlp:chains"`).
#[derive(Clone, Debug)]
pub struct CorefChains {
    pub spans: Spans<CorefChains>,
    pub mentions: Vec<CorefMention>,
}

impl CorefChains {
    /// Iterates over `(span, mention)` pairs in layer order.
    pub fn iter(&self) -> impl Iterator<Item = (Span, &CorefMention)> {
        (0..self.spans.len()).map(|i| (self.spans.get(Ix::new(i as u32)), &self.mentions[i]))
    }
}

impl Layer for CorefChains {
    const ID: &'static str = "opennlp:chains";
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<turbo_analysis::TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        if self.mentions.len() != self.spans.len() {
            return Err("a mention for every coreference span".into());
        }
        for (i, m) in self.mentions.iter().enumerate() {
            if is_blank(&m.kind) {
                return Err(format!("coreference mention {i} has a blank kind"));
            }
            if m.entity < CorefMention::NO_ENTITY {
                return Err(format!("coreference mention {i} has invalid entity index {}", m.entity));
            }
        }
        self.spans.check(Self::OVERLAP)
    }
}

/// Speaker turns or sentences (`CorefAnnotator.SPEAKERS`,
/// `"opennlp:speakers"`).
#[derive(Clone, Debug)]
pub struct Speakers {
    pub spans: Spans<Speakers>,
    pub speakers: Strs,
}

impl Speakers {
    /// Iterates over `(span, speaker)` pairs in layer order.
    pub fn iter(&self) -> impl Iterator<Item = (Span, &str)> {
        (0..self.spans.len()).map(|i| (self.spans.get(Ix::new(i as u32)), self.speakers.get(i)))
    }
}

impl Layer for Speakers {
    const ID: &'static str = "opennlp:speakers";

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<turbo_analysis::TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        if self.speakers.len() != self.spans.len() {
            return Err("a speaker label for every speaker span".into());
        }
        for i in 0..self.speakers.len() {
            if is_blank(self.speakers.get(i)) {
                return Err(format!("speaker {i} must not be blank"));
            }
        }
        self.spans.check(Self::OVERLAP)
    }
}

/// Reads [`Document`]s with gold coreference chains from CoNLL-U text
/// (`ConlluCorefDocumentStream`).
#[derive(Clone, Debug)]
pub struct ConlluCorefReader<'a> {
    source: &'a str,
    lines: Lines<'a>,
    tagset: Tagset,
    pending_line: Option<Line<'a>>,
}

impl<'a> ConlluCorefReader<'a> {
    /// Opens a CoNLL-U coreference document stream over `source` using
    /// `tagset` for the [`PosTags`] layer.
    pub fn new(source: &'a str, tagset: Tagset) -> Result<ConlluCorefReader<'a>, FormatError> {
        Ok(ConlluCorefReader { source, lines: Lines::new(source)?, tagset, pending_line: None })
    }

    /// Rewinds the reader to the start of `source`
    /// (`ConlluCorefDocumentStream.reset`).
    pub fn reset(&mut self) {
        self.lines = Lines::new(self.source).expect("source length already checked");
        self.pending_line = None;
    }

    fn next_line(&mut self) -> Option<Line<'a>> {
        self.pending_line.take().or_else(|| self.lines.next())
    }

    fn parse(&self, lines: &[Line<'a>]) -> Result<Document, FormatError> {
        let mut text = String::new();
        let mut sentences: Vec<Span> = Vec::new();
        let mut tokens: Vec<(Span, Ix<Sentences>)> = Vec::new();
        let mut tags: Vec<&'a str> = Vec::new();
        let mut speakers: Vec<(Span, String)> = Vec::new();

        // Open mention starts by canonical entity id, preserving first-seen key order.
        let mut open_starts: HashMap<String, Vec<u32>> = HashMap::new();
        let mut open_order: Vec<String> = Vec::new();
        // Completed mention spans by canonical entity id, in first-mention order (LinkedHashMap).
        let mut entity_order: Vec<String> = Vec::new();
        let mut entities: HashMap<String, Vec<Span>> = HashMap::new();

        let mut speaker: Option<String> = None;
        let mut sentence_start: Option<u32> = None;
        let mut expected_word_id: usize = 1;
        let mut glue = false;
        let mut last_line_num = lines.first().map_or(1, |l| l.number);

        for &line in lines {
            last_line_num = line.number;
            let raw = line.text();
            if raw.is_empty() {
                if let Some(sent_start) = sentence_start.take() {
                    reject_open_mentions(&open_order, &open_starts, line.number, " crosses a sentence boundary")?;
                    let sent_span = Span::new(sent_start, text.len() as u32);
                    sentences.push(sent_span);
                    if let Some(spk) = speaker.take() {
                        speakers.push((sent_span, spk));
                    }
                }
                expected_word_id = 1;
                continue;
            }
            if raw.as_bytes()[0] == b'#' {
                if raw.starts_with(SPEAKER) {
                    let trimmed = java_trim(line.piece.slice(SPEAKER.len(), raw.len())).text;
                    if is_blank(trimmed) {
                        return Err(FormatError::ConlluCoref {
                            line: line.number,
                            message: "speaker label must not be blank".into(),
                        });
                    }
                    speaker = Some(trimmed.to_owned());
                }
                continue;
            }
            let fields = split_tab_all(line.piece);
            if fields.len() != 10 {
                return Err(FormatError::ConlluCoref {
                    line: line.number,
                    message: format!("expected 10 columns: {raw}"),
                });
            }
            for (f, field) in fields.iter().enumerate() {
                if is_blank(field.text) {
                    return Err(FormatError::ConlluCoref {
                        line: line.number,
                        message: format!("column {} must not be blank: {raw}", f + 1),
                    });
                }
            }
            if !plain_id(fields[0].text, line.number)? {
                continue;
            }
            if fields[0].text != expected_word_id.to_string() {
                return Err(FormatError::ConlluCoref {
                    line: line.number,
                    message: format!("expected token id {expected_word_id} but found {}", fields[0].text),
                });
            }
            expected_word_id += 1;
            if sentence_start.is_none() {
                if !text.is_empty() {
                    text.push('\n');
                }
                sentence_start = Some(text.len() as u32);
                glue = true;
            }
            if !glue {
                text.push(' ');
            }
            let start = text.len() as u32;
            text.push_str(fields[1].text);
            let end = text.len() as u32;
            let sent_ix = Ix::<Sentences>::new(sentences.len() as u32);
            tokens.push((Span::new(start, end), sent_ix));
            tags.push(match self.tagset {
                Tagset::U => fields[3].text,
                Tagset::X => fields[4].text,
            });
            glue = misc(fields[9].text, SPACE_AFTER) == Some("No");
            if let Some(entity) = misc(fields[9].text, ENTITY) {
                brackets(
                    entity,
                    start,
                    end,
                    line.number,
                    &mut open_order,
                    &mut open_starts,
                    &mut entity_order,
                    &mut entities,
                )?;
            }
        }
        if let Some(sent_start) = sentence_start.take() {
            let sent_span = Span::new(sent_start, text.len() as u32);
            sentences.push(sent_span);
            if let Some(spk) = speaker.take() {
                speakers.push((sent_span, spk));
            }
        }
        reject_open_mentions(&open_order, &open_starts, last_line_num, " is never closed")?;

        let mut doc = Document::new(text);

        let mut sent_spans = Spans::<Sentences>::with_capacity(ViewId::SOURCE, sentences.len());
        for s in sentences {
            sent_spans.push(s);
        }
        doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("conllu-coref"))
            .map_err(|e| FormatError::ConlluDocument { line: last_line_num, detail: e.to_string() })?;

        let mut tok_spans = Spans::<Tokens>::with_capacity(ViewId::SOURCE, tokens.len());
        let mut tok_sents = Vec::with_capacity(tokens.len());
        for (span, s_ix) in tokens {
            tok_spans.push(span);
            tok_sents.push(s_ix);
        }
        doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: Some(tok_sents) }, Provenance::by("conllu-coref"))
            .map_err(|e| FormatError::ConlluDocument { line: last_line_num, detail: e.to_string() })?;

        let mut unique_tags: Vec<&str> = Vec::new();
        for &t in &tags {
            if !unique_tags.contains(&t) {
                unique_tags.push(t);
            }
        }
        let tagset_name = match self.tagset {
            Tagset::U => "upos",
            Tagset::X => "xpos",
        };
        let tag_set = TagSet::new(tagset_name, &unique_tags);
        let tag_ids: Vec<Tag> = tags.iter().map(|&t| tag_set.tag(t).unwrap_or(Tag(0))).collect();
        doc.add(&Key::DEFAULT, PosTags { tagset: tag_set, tags: tag_ids, prob: None }, Provenance::by("conllu-coref"))
            .map_err(|e| FormatError::ConlluDocument { line: last_line_num, detail: e.to_string() })?;

        let chains_layer = build_chains(&entity_order, &entities);
        doc.add(&GOLD_CHAINS, chains_layer, Provenance::by("conllu-coref"))
            .map_err(|e| FormatError::ConlluDocument { line: last_line_num, detail: e.to_string() })?;

        if !speakers.is_empty() {
            let mut spk_spans = Spans::<Speakers>::with_capacity(ViewId::SOURCE, speakers.len());
            let mut spk_strs = Strs::default();
            for (span, label) in speakers {
                spk_spans.push(span);
                spk_strs.push(&label);
            }
            doc.add(&SPEAKERS, Speakers { spans: spk_spans, speakers: spk_strs }, Provenance::by("conllu-coref"))
                .map_err(|e| FormatError::ConlluDocument { line: last_line_num, detail: e.to_string() })?;
        }

        Ok(doc)
    }
}

impl Iterator for ConlluCorefReader<'_> {
    type Item = Result<Document, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut lines = Vec::new();
            let mut line = self.next_line();
            while let Some(cur) = line {
                if cur.text().starts_with(NEWDOC) && !lines.is_empty() {
                    self.pending_line = Some(cur);
                    break;
                }
                lines.push(cur);
                line = self.lines.next();
            }
            if lines.iter().any(|l| !l.text().is_empty() && l.text().as_bytes()[0] != b'#') {
                return Some(self.parse(&lines));
            }
            self.pending_line?;
        }
    }
}

fn build_chains(entity_order: &[String], entities: &HashMap<String, Vec<Span>>) -> CorefChains {
    let mut items: Vec<(Span, CorefMention)> = Vec::new();
    for (chain, id) in entity_order.iter().enumerate() {
        if let Some(spans) = entities.get(id) {
            for &span in spans {
                items.push((
                    span,
                    CorefMention {
                        chain: chain as u32,
                        kind: CorefMention::KIND_GOLD.to_owned(),
                        entity: CorefMention::NO_ENTITY,
                    },
                ));
            }
        }
    }
    items.sort_by(|(a, _), (b, _)| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));
    let mut spans = Spans::<CorefChains>::with_capacity(ViewId::SOURCE, items.len());
    let mut mentions = Vec::with_capacity(items.len());
    for (span, mention) in items {
        spans.push(span);
        mentions.push(mention);
    }
    CorefChains { spans, mentions }
}

fn plain_id(id: &str, line: u32) -> Result<bool, FormatError> {
    if positive_integer(id) {
        return Ok(true);
    }
    if compound_id(id, '-') || compound_id(id, '.') {
        return Ok(false);
    }
    Err(FormatError::ConlluCoref { line, message: format!("invalid token id: {id}") })
}

fn positive_integer(value: &str) -> bool {
    let bytes = value.as_bytes();
    matches!(bytes.first(), Some(b'1'..=b'9')) && bytes[1..].iter().all(u8::is_ascii_digit)
}

fn nonnegative_integer(value: &str) -> bool {
    value == "0" || positive_integer(value)
}

fn less_than(first: &str, second: &str) -> bool {
    if first.len() != second.len() { first.len() < second.len() } else { first < second }
}

fn compound_id(id: &str, separator: char) -> bool {
    let Some(split) = id.find(separator) else {
        return false;
    };
    if split == 0 || id[split + 1..].contains(separator) {
        return false;
    }
    let prefix = &id[..split];
    let suffix = &id[split + 1..];
    let valid_prefix = if separator == '.' { nonnegative_integer(prefix) } else { positive_integer(prefix) };
    if !valid_prefix || !positive_integer(suffix) {
        return false;
    }
    separator == '.' || less_than(prefix, suffix)
}

fn misc<'a>(misc_col: &'a str, key: &str) -> Option<&'a str> {
    let mut start = 0;
    while start < misc_col.len() {
        let end = misc_col[start..].find('|').map_or(misc_col.len(), |i| start + i);
        let attr = &misc_col[start..end];
        if let Some(rest) = attr.strip_prefix(key)
            && let Some(val) = rest.strip_prefix('=')
        {
            return Some(val);
        }
        start = end + 1;
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn brackets(
    entity: &str,
    start: u32,
    end: u32,
    line: u32,
    open_order: &mut Vec<String>,
    open_starts: &mut HashMap<String, Vec<u32>>,
    entity_order: &mut Vec<String>,
    entities: &mut HashMap<String, Vec<Span>>,
) -> Result<(), FormatError> {
    if entity.is_empty() {
        return Err(FormatError::ConlluCoref { line, message: "entity id must not be empty".into() });
    }
    let bytes = entity.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'(' && bytes[j] != b')' {
                j += 1;
            }
            let id = canonical_id(&entity[i + 1..j], line)?;
            if !entities.contains_key(id) {
                entity_order.push(id.to_owned());
                entities.insert(id.to_owned(), Vec::new());
            }
            if j < bytes.len() && bytes[j] == b')' {
                entities.get_mut(id).expect("inserted above").push(Span::new(start, end));
                j += 1;
            } else {
                if !open_starts.contains_key(id) {
                    open_order.push(id.to_owned());
                    open_starts.insert(id.to_owned(), Vec::new());
                }
                open_starts.get_mut(id).expect("inserted above").push(start);
            }
            i = j;
        } else {
            let mut j = i;
            while j < bytes.len() && bytes[j] != b')' {
                j += 1;
            }
            if j == bytes.len() {
                return Err(FormatError::ConlluCoref {
                    line,
                    message: format!("entity close is missing ')': {entity}"),
                });
            }
            let id = canonical_id(&entity[i..j], line)?;
            let Some(starts) = open_starts.get_mut(id) else {
                return Err(FormatError::ConlluCoref { line, message: format!("entity {id} is closed but not open") });
            };
            let Some(mention_start) = starts.pop() else {
                return Err(FormatError::ConlluCoref { line, message: format!("entity {id} is closed but not open") });
            };
            if !entities.contains_key(id) {
                entity_order.push(id.to_owned());
                entities.insert(id.to_owned(), Vec::new());
            }
            entities.get_mut(id).expect("inserted above").push(Span::new(mention_start, end));
            i = j + 1;
        }
    }
    Ok(())
}

fn reject_open_mentions(
    open_order: &[String],
    open_starts: &HashMap<String, Vec<u32>>,
    line: u32,
    reason: &str,
) -> Result<(), FormatError> {
    for id in open_order {
        if open_starts.get(id).is_some_and(|s| !s.is_empty()) {
            return Err(FormatError::ConlluCoref { line, message: format!("entity {id}{reason}") });
        }
    }
    Ok(())
}

fn canonical_id(id: &str, line: u32) -> Result<&str, FormatError> {
    let end = id.find(['-', '[']).unwrap_or(id.len());
    if end == 0 {
        return Err(FormatError::ConlluCoref { line, message: "entity id must not be empty".into() });
    }
    Ok(&id[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONLLU: &str = "\
# newdoc id = doc1
# sent_id = doc1-1
# speaker = Ann
1\tByron\tByron\tPROPN\tNNP\t_\t2\tnsubj\t_\tEntity=(1)
2\tmet\tmeet\tVERB\tVBD\t_\t0\troot\t_\t_
3\this\the\tPRON\tPRP$\t_\t4\tnmod:poss\t_\tEntity=(2-person-new(1)
4\tmother\tmother\tNOUN\tNN\t_\t2\tobj\t_\tEntity=2)|SpaceAfter=No
5\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_

# sent_id = doc1-2
1-2\tShe's\t_\t_\t_\t_\t_\t_\t_\t_
1\tShe\tshe\tPRON\tPRP\t_\t3\tnsubj\t_\tEntity=(2)
2\t's\tbe\tAUX\tVBZ\t_\t3\tcop\t_\t_
3\thome\thome\tNOUN\tNN\t_\t0\troot\t_\tSpaceAfter=No
4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_

# newdoc id = doc2
1\tAcme\tAcme\tPROPN\tNNP\t_\t2\tnsubj\t_\tEntity=(7)
2\tgrew\tgrow\tVERB\tVBD\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_
";

    #[test]
    fn reads_documents_with_layers_and_gold_chains() {
        let mut reader = ConlluCorefReader::new(CONLLU, Tagset::X).unwrap();
        let first = reader.next().unwrap().unwrap();
        assert_eq!(first.source(), "Byron met his mother.\nShe 's home.");

        let sents = first.get::<Sentences>().unwrap();
        let sent_texts: Vec<&str> =
            (0..sents.len()).map(|i| &first.source()[sents.spans.get(Ix::new(i as u32)).range()]).collect();
        assert_eq!(sent_texts, ["Byron met his mother.", "She 's home."]);

        let toks = first.get::<Tokens>().unwrap();
        let tok_texts: Vec<&str> =
            (0..toks.len()).map(|i| &first.source()[toks.spans.get(Ix::new(i as u32)).range()]).collect();
        assert_eq!(tok_texts, ["Byron", "met", "his", "mother", ".", "She", "'s", "home", "."]);

        let pos = first.get::<PosTags>().unwrap();
        let pos_texts: Vec<&str> = pos.tags.iter().map(|&t| pos.tagset.label(t)).collect();
        assert_eq!(pos_texts, ["NNP", "VBD", "PRP$", "NN", ".", "PRP", "VBZ", "NN", "."]);

        let spk = first.get_key(&SPEAKERS).unwrap();
        let spk_items: Vec<(Span, &str)> = spk.iter().collect();
        assert_eq!(spk_items, [(Span::new(0, 21), "Ann")]);

        let chains = first.get_key(&GOLD_CHAINS).unwrap();
        let mention_texts: Vec<&str> = chains.iter().map(|(span, _)| &first.source()[span.range()]).collect();
        assert_eq!(mention_texts, ["Byron", "his mother", "his", "She"]);
        let chain_ids: Vec<u32> = chains.iter().map(|(_, m)| m.chain).collect();
        assert_eq!(chain_ids, [0, 1, 0, 1]);
        assert_eq!(chains.mentions[0].kind, CorefMention::KIND_GOLD);

        let second = reader.next().unwrap().unwrap();
        assert_eq!(second.source(), "Acme grew.");
        assert!(second.get_key(&SPEAKERS).is_none());
        assert_eq!(second.get_key(&GOLD_CHAINS).unwrap().len(), 1);
        assert!(reader.next().is_none());

        reader.reset();
        assert_eq!(reader.next().unwrap().unwrap().source(), "Byron met his mother.\nShe 's home.");
    }

    #[test]
    fn nested_discontinuous_and_empty_nodes() {
        let nested = "\
1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\tEntity=(outer
2\tcompany\tcompany\tNOUN\tNN\t_\t3\tnsubj\t_\tEntity=(inner)
3\tgrew\tgrow\tVERB\tVBD\t_\t0\troot\t_\tEntity=outer)
";
        let doc = ConlluCorefReader::new(nested, Tagset::X).unwrap().next().unwrap().unwrap();
        let chains = doc.get_key(&GOLD_CHAINS).unwrap();
        let items: Vec<(Span, u32)> = chains.iter().map(|(s, m)| (s, m.chain)).collect();
        assert_eq!(items, [(Span::new(0, 16), 0), (Span::new(4, 11), 1)]);

        let parts = "\
1\tKim\tKim\tPROPN\tNNP\t_\t0\troot\t_\tEntity=(e5[1/2]-person-1)
2\tand\tand\tCCONJ\tCC\t_\t3\tcc\t_\t_
3\tLee\tLee\tPROPN\tNNP\t_\t1\tconj\t_\tEntity=(e5[2/2]-person-1)
";
        let doc = ConlluCorefReader::new(parts, Tagset::X).unwrap().next().unwrap().unwrap();
        let chains = doc.get_key(&GOLD_CHAINS).unwrap();
        assert_eq!(chains.len(), 2);
        assert_eq!(chains.mentions[0].chain, chains.mentions[1].chain);

        let empty_node = "\
0.1\thad\thave\tAUX\tVBD\t_\t_\t_\t_\t_
1\tspoke\tspeak\tVERB\tVBD\t_\t0\troot\t_\t_
1.1\thad\thave\tAUX\tVBD\t_\t_\t_\t_\t_
";
        let doc = ConlluCorefReader::new(empty_node, Tagset::X).unwrap().next().unwrap().unwrap();
        assert_eq!(doc.source(), "spoke");
    }

    #[test]
    fn rejects_malformed_inputs() {
        for bad in [
            "1\tAcme\tAcme\tPROPN\tNNP\t_\t0\troot\t_\tEntity=\n",
            "1\tAcme\tAcme\tPROPN\tNNP\t_\t0\troot\t_\tEntity=(3\n",
            "1\tAcme\tAcme\tPROPN\tNNP\t_\t0\troot\t_\tEntity=3)\n",
            "1\tAcme\tAcme\tPROPN\tNNP\t_\t2\tnsubj\t_\tEntity=(3\n2\tgrew\tgrow\tVERB\tVBD\t_\t0\troot\t_\tEntity=3\n",
            "1\tAcme\tAcme\tPROPN\tNNP\t_\t0\troot\t_\tEntity=(3\n\n1\tIt\tit\tPRON\tPRP\t_\t0\troot\t_\tEntity=3)\n",
            "1\tAcme\n",
            "1\tAcme\tAcme\tPROPN\tNNP\t_\t0\troot\t_\t_\textra\n",
            "1\tAcme\tAcme\tPROPN\tNNP\t_\t0\troot\t_\tEntity=()\n",
            "1\t\t_\tPROPN\tNNP\t_\t0\troot\t_\t_\n",
            "# speaker =   \n1\tAcme\t_\tPROPN\tNNP\t_\t0\troot\t_\t_\n",
        ] {
            let err = ConlluCorefReader::new(bad, Tagset::X).unwrap().next().unwrap().err().unwrap();
            assert_eq!(err.java_class(), Some("opennlp.tools.util.InvalidFormatException"), "{bad}");
        }

        for id in ["word", "0", "01", "1.0", "01.1", "1.01", "1.1.1", "1-1", "2-1", "0-1", "1-0", "1--2"] {
            let row = format!("{id}\tAcme\tAcme\tPROPN\tNNP\t_\t0\troot\t_\t_\n");
            let err = ConlluCorefReader::new(&row, Tagset::X).unwrap().next().unwrap().err().unwrap();
            assert_eq!(err.to_string(), format!("invalid token id: {id}"));
        }
    }
}
