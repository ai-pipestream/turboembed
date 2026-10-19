//! Princeton WordNet WNDB 3.0 / 3.1 directory reader (`WndbReader`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::error::WordNetError;
use crate::lexicon::{InMemoryWordNetLexicon, LemmaKey, Synset};
use crate::pos::WordNetPos;
use crate::relation::WordNetRelation;

const FILE_POS_LIST: [(WordNetPos, char, &str); 4] = [
    (WordNetPos::Noun, 'n', "noun"),
    (WordNetPos::Verb, 'v', "verb"),
    (WordNetPos::Adjective, 'a', "adj"),
    (WordNetPos::Adverb, 'r', "adv"),
];

const INDEX_POINTER_SYMBOLS: [&str; 20] =
    ["!", "@", "~", "#m", "#s", "#p", "%m", "%s", "%p", "=", "+", "*", ">", "^", "$", "&", "<", "\\", ";", "-"];

/// Reader for Princeton WordNet 3.0 / 3.1 database directories (`data.*` and `index.*`).
pub struct WndbReader;

impl WndbReader {
    /// Reads a WNDB directory containing the eight `data.{noun,verb,adj,adv}` and
    /// `index.{noun,verb,adj,adv}` files into a validated [`InMemoryWordNetLexicon`].
    pub fn read(directory: impl AsRef<Path>) -> Result<InMemoryWordNetLexicon, WordNetError> {
        Self::read_directory(directory)
    }

    /// Reads a WNDB directory containing the eight `data.{noun,verb,adj,adv}` and
    /// `index.{noun,verb,adj,adv}` files into a validated [`InMemoryWordNetLexicon`].
    pub fn read_directory(directory: impl AsRef<Path>) -> Result<InMemoryWordNetLexicon, WordNetError> {
        let dir = directory.as_ref();
        if !dir.is_dir() {
            return Err(WordNetError::invalid(format!(
                "directory does not exist or is not a directory: {}",
                dir.display()
            )));
        }

        let mut raw_synsets: Vec<(String, RawSynset)> = Vec::new();
        let mut raw_index: HashMap<String, usize> = HashMap::new();
        for &(pos, pos_char, suffix) in &FILE_POS_LIST {
            parse_data_file(dir, pos, pos_char, suffix, &mut raw_synsets, &mut raw_index)?;
        }

        let synsets_by_id = resolve_synsets(&raw_synsets, &raw_index)?;
        let mut sense_order: HashMap<LemmaKey, Vec<String>> = HashMap::new();
        for &(pos, pos_char, suffix) in &FILE_POS_LIST {
            parse_index_file(dir, pos, pos_char, suffix, &raw_index, &mut sense_order)?;
        }

        InMemoryWordNetLexicon::new(synsets_by_id, sense_order)
    }
}

struct RawPointer {
    relation: WordNetRelation,
    target_id: String,
    target_word: usize,
    line_number: usize,
}

struct RawSynset {
    id: String,
    pos: WordNetPos,
    word_count: usize,
    lemmas: Vec<String>,
    gloss: String,
    pointers: Vec<RawPointer>,
    file_name: String,
}

fn synset_id(offset: &str, pos_char: char) -> String {
    format!("wndb-{offset}-{pos_char}")
}

fn read_all(file: &Path) -> Result<Vec<u8>, WordNetError> {
    if !file.is_file() {
        return Err(WordNetError::invalid(format!("Missing WNDB database file: {}", file.display())));
    }
    Ok(fs::read(file)?)
}

fn decode_latin1(bytes: &[u8]) -> std::borrow::Cow<'_, str> {
    if bytes.is_ascii() {
        std::borrow::Cow::Borrowed(std::str::from_utf8(bytes).expect("ASCII bytes are valid UTF-8"))
    } else {
        std::borrow::Cow::Owned(bytes.iter().map(|&b| b as char).collect())
    }
}

fn malformed(file_name: &str, line_number: usize, message: impl AsRef<str>) -> WordNetError {
    WordNetError::invalid(format!("Malformed WNDB file {file_name} at line {line_number}: {}", message.as_ref()))
}

fn parse_data_file(
    directory: &Path,
    pos: WordNetPos,
    pos_char: char,
    suffix: &str,
    raw_synsets: &mut Vec<(String, RawSynset)>,
    raw_index: &mut HashMap<String, usize>,
) -> Result<(), WordNetError> {
    let file_name = format!("data.{suffix}");
    let bytes = read_all(&directory.join(&file_name))?;
    let mut line_start = 0usize;
    let mut lf_offset = 0usize;
    let mut line_number = 0usize;
    while line_start < bytes.len() {
        line_number += 1;
        let line_end = bytes[line_start..].iter().position(|&b| b == b'\n').map_or(bytes.len(), |i| line_start + i);
        let raw_line = &bytes[line_start..line_end];
        let trimmed = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        let line = decode_latin1(trimmed);
        if !line.is_empty() && !line.starts_with(' ') {
            parse_data_line(
                &line,
                line_start,
                lf_offset,
                &file_name,
                line_number,
                pos,
                pos_char,
                raw_synsets,
                raw_index,
            )?;
        }
        line_start = line_end + 1;
        lf_offset += trimmed.len() + 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn parse_data_line(
    line: &str,
    byte_offset: usize,
    lf_offset: usize,
    file_name: &str,
    line_number: usize,
    pos: WordNetPos,
    pos_char: char,
    raw_synsets: &mut Vec<(String, RawSynset)>,
    raw_index: &mut HashMap<String, usize>,
) -> Result<(), WordNetError> {
    let mut tokens = Tokenizer::new(line, file_name, line_number);
    let offset_field = tokens.next_field("synset_offset")?;
    let parsed_offset = parse_offset(offset_field, &tokens)?;
    if parsed_offset != byte_offset && parsed_offset != lf_offset {
        return Err(malformed(
            file_name,
            line_number,
            format!("Synset offset field {offset_field} disagrees with the actual byte position {byte_offset}"),
        ));
    }
    let lex_filenum = tokens.next_fixed_int("lex_filenum", 10, 2)?;
    if lex_filenum > 44 {
        return Err(malformed(
            file_name,
            line_number,
            format!("lex_filenum must be between 00 and 44, got: {lex_filenum}"),
        ));
    }
    let ss_type = tokens.next_field("ss_type (synset type)")?;
    let valid_type = if pos == WordNetPos::Adjective {
        ss_type == "a" || ss_type == "s"
    } else {
        ss_type.len() == 1 && ss_type.starts_with(pos_char)
    };
    if !valid_type {
        return Err(malformed(file_name, line_number, format!("Synset type {ss_type} does not belong in {file_name}")));
    }
    let word_count = tokens.next_fixed_count("w_cnt", 16, 2, "Word count")?;
    if word_count < 1 {
        return Err(malformed(file_name, line_number, format!("Word count must be at least 1, got: {word_count}")));
    }
    let mut lemmas = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        let word = tokens.next_field("word")?;
        let lemma = clean_lemma(word, file_name, line_number, pos)?;
        tokens.next_fixed_int("lex_id", 16, 1)?;
        if !lemmas.contains(&lemma) {
            lemmas.push(lemma);
        }
    }
    let pointer_count = tokens.next_fixed_count("p_cnt", 10, 3, "Pointer count")?;
    let mut pointers = Vec::with_capacity(pointer_count);
    for _ in 0..pointer_count {
        let symbol = tokens.next_field("pointer_symbol")?;
        let Some(relation) = map_pointer(symbol) else {
            return Err(malformed(file_name, line_number, format!("Undeclared pointer symbol: {symbol}")));
        };
        let target_offset = tokens.next_field("pointer synset_offset")?;
        parse_offset(target_offset, &tokens)?;
        let target_pos_tok = tokens.next_field("pointer pos")?;
        let target_pos = parse_pointer_pos_char(target_pos_tok, &tokens)?;
        let source_target = tokens.next_fixed_int("pointer source/target", 16, 4)?;
        let source_word = source_target >> 8;
        let target_word = source_target & 0xff;
        if source_word > word_count {
            return Err(malformed(
                file_name,
                line_number,
                format!("Pointer source word {source_word} exceeds word count {word_count}"),
            ));
        }
        if (source_word == 0) != (target_word == 0) {
            return Err(malformed(
                file_name,
                line_number,
                "Pointer source and target words must both be zero or nonzero",
            ));
        }
        pointers.push(RawPointer {
            relation,
            target_id: synset_id(target_offset, target_pos),
            target_word,
            line_number,
        });
    }
    if pos == WordNetPos::Verb {
        let frame_count = tokens.next_fixed_count("f_cnt", 10, 2, "Verb frame count")?;
        for _ in 0..frame_count {
            let marker = tokens.next_field("verb frame marker")?;
            if marker != "+" {
                return Err(malformed(
                    file_name,
                    line_number,
                    format!("Expected + before a verb frame, got: {marker}"),
                ));
            }
            let frame_number = tokens.next_fixed_int("f_num", 10, 2)?;
            if !(1..=35).contains(&frame_number) {
                return Err(malformed(
                    file_name,
                    line_number,
                    format!("f_num must be between 01 and 35, got: {frame_number}"),
                ));
            }
            let word_number = tokens.next_fixed_int("w_num", 16, 2)?;
            if word_number > word_count {
                return Err(malformed(
                    file_name,
                    line_number,
                    format!("Verb frame word {word_number} exceeds word count {word_count}"),
                ));
            }
        }
    }
    let gloss = tokens.gloss()?;
    let id = synset_id(offset_field, pos_char);
    let raw = RawSynset { id: id.clone(), pos, word_count, lemmas, gloss, pointers, file_name: file_name.to_string() };
    if let Some(&existing_idx) = raw_index.get(&id) {
        raw_synsets[existing_idx] = (id, raw);
    } else {
        let idx = raw_synsets.len();
        raw_index.insert(id.clone(), idx);
        raw_synsets.push((id, raw));
    }
    Ok(())
}

fn resolve_synsets(
    raw_synsets: &[(String, RawSynset)],
    raw_index: &HashMap<String, usize>,
) -> Result<HashMap<String, Synset>, WordNetError> {
    let mut synsets_by_id = HashMap::with_capacity(raw_synsets.len());
    for (_, raw) in raw_synsets {
        let mut relations: BTreeMap<WordNetRelation, Vec<String>> = BTreeMap::new();
        for pointer in &raw.pointers {
            let Some(&target_idx) = raw_index.get(&pointer.target_id) else {
                return Err(malformed(
                    &raw.file_name,
                    pointer.line_number,
                    format!(
                        "Synset {} has a {:?} pointer to nonexistent synset {}",
                        raw.id, pointer.relation, pointer.target_id
                    ),
                ));
            };
            let target = &raw_synsets[target_idx].1;
            if pointer.target_word > target.word_count {
                return Err(malformed(
                    &raw.file_name,
                    pointer.line_number,
                    format!(
                        "Pointer target word {} exceeds target word count {}",
                        pointer.target_word, target.word_count
                    ),
                ));
            }
            let list = relations.entry(pointer.relation).or_default();
            if !list.contains(&target.id) {
                list.push(target.id.clone());
            }
        }
        let synset = Synset::new(raw.id.clone(), raw.pos, raw.lemmas.clone(), raw.gloss.clone(), relations)?;
        synsets_by_id.insert(raw.id.clone(), synset);
    }
    Ok(synsets_by_id)
}

fn parse_index_file(
    directory: &Path,
    pos: WordNetPos,
    pos_char: char,
    suffix: &str,
    raw_index: &HashMap<String, usize>,
    senses: &mut HashMap<LemmaKey, Vec<String>>,
) -> Result<(), WordNetError> {
    let file_name = format!("index.{suffix}");
    let bytes = read_all(&directory.join(&file_name))?;
    let mut line_number = 0usize;
    let mut line_start = 0usize;
    while line_start < bytes.len() {
        line_number += 1;
        let line_end = bytes[line_start..].iter().position(|&b| b == b'\n').map_or(bytes.len(), |i| line_start + i);
        let raw_line = &bytes[line_start..line_end];
        let trimmed = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        let line = decode_latin1(trimmed);
        if !line.is_empty() && !line.starts_with(' ') {
            parse_index_line(&line, &file_name, line_number, pos, pos_char, suffix, raw_index, senses)?;
        }
        line_start = line_end + 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn parse_index_line(
    line: &str,
    file_name: &str,
    line_number: usize,
    pos: WordNetPos,
    pos_char: char,
    suffix: &str,
    raw_index: &HashMap<String, usize>,
    senses: &mut HashMap<LemmaKey, Vec<String>>,
) -> Result<(), WordNetError> {
    let mut tokens = Tokenizer::new(line, file_name, line_number);
    let lemma = tokens.next_field("lemma")?;
    let pos_tok = tokens.next_field("pos")?;
    if pos_tok.len() != 1 || !pos_tok.starts_with(pos_char) {
        return Err(malformed(file_name, line_number, format!("Index pos {pos_tok} does not belong in {file_name}")));
    }
    let synset_count = tokens.next_non_negative_int("synset count", "Synset count")?;
    if synset_count < 1 {
        return Err(malformed(file_name, line_number, format!("Synset count must be at least 1, got: {synset_count}")));
    }
    let pointer_type_count = tokens.next_non_negative_int("pointer count", "Pointer count")?;
    let mut pointer_types = HashSet::with_capacity(pointer_type_count);
    for _ in 0..pointer_type_count {
        let symbol = tokens.next_field("ptr_symbol (pointer symbol)")?;
        if !INDEX_POINTER_SYMBOLS.contains(&symbol) {
            return Err(malformed(file_name, line_number, format!("Undeclared pointer symbol: {symbol}")));
        }
        if !pointer_types.insert(symbol) {
            return Err(malformed(file_name, line_number, format!("Duplicate pointer symbol: {symbol}")));
        }
    }
    let sense_count = tokens.next_non_negative_int("sense count", "Sense count")?;
    if sense_count != synset_count {
        return Err(malformed(
            file_name,
            line_number,
            format!("Sense count {sense_count} does not match synset count {synset_count}"),
        ));
    }
    let tagged_sense_count = tokens.next_non_negative_int("tagged-sense count", "Tagged-sense count")?;
    if tagged_sense_count > sense_count {
        return Err(malformed(
            file_name,
            line_number,
            format!("Tagged-sense count {tagged_sense_count} exceeds sense count {sense_count}"),
        ));
    }
    if synset_count > tokens.remaining_fixed_width_fields(8) {
        return Err(malformed(
            file_name,
            line_number,
            format!("Synset count {synset_count} exceeds the available offset fields"),
        ));
    }
    let mut order = Vec::with_capacity(synset_count);
    for _ in 0..synset_count {
        let offset = tokens.next_field("synset_offset")?;
        parse_offset(offset, &tokens)?;
        let id = synset_id(offset, pos_char);
        if !raw_index.contains_key(&id) {
            return Err(malformed(
                file_name,
                line_number,
                format!("Lemma {lemma} references offset {offset} with no data.{suffix} line"),
            ));
        }
        if order.contains(&id) {
            return Err(malformed(file_name, line_number, format!("Duplicate synset offset {offset}")));
        }
        order.push(id);
    }
    tokens.require_end("synset offsets")?;
    let key = LemmaKey::of(lemma, pos);
    let existing = senses.entry(key).or_default();
    for id in order {
        if !existing.contains(&id) {
            existing.push(id);
        }
    }
    Ok(())
}

fn clean_lemma(word: &str, file_name: &str, line_number: usize, file_pos: WordNetPos) -> Result<String, WordNetError> {
    let mut cleaned = word;
    if cleaned.ends_with(')') {
        let marker = match cleaned.rfind('(') {
            Some(open) => &cleaned[open..],
            None => "",
        };
        if marker != "(p)" && marker != "(a)" && marker != "(ip)" {
            return Err(malformed(file_name, line_number, format!("Unknown syntactic marker on word: {word}")));
        }
        if file_pos != WordNetPos::Adjective {
            return Err(malformed(
                file_name,
                line_number,
                format!("Syntactic marker {marker} is only valid in data.adj"),
            ));
        }
        cleaned = &cleaned[..cleaned.len() - marker.len()];
    }
    if cleaned.is_empty() {
        return Err(malformed(file_name, line_number, "Empty word field"));
    }
    Ok(cleaned.replace('_', " "))
}

fn parse_offset(offset: &str, tokens: &Tokenizer<'_>) -> Result<usize, WordNetError> {
    if offset.len() != 8 || !offset.bytes().all(|b| b.is_ascii_digit()) {
        return Err(tokens.malformed_token(format!("Synset offset must be 8 digits, got: {offset}")));
    }
    offset
        .parse::<usize>()
        .map_err(|_| tokens.malformed_token(format!("Synset offset must be 8 digits, got: {offset}")))
}

fn parse_pointer_pos_char(pos: &str, tokens: &Tokenizer<'_>) -> Result<char, WordNetError> {
    if pos.len() == 1 {
        let c = pos.chars().next().unwrap();
        if matches!(c, 'n' | 'v' | 'a' | 'r') {
            return Ok(c);
        }
    }
    Err(tokens.malformed_token(format!("Pointer pos must be one of n, v, a, r, got: {pos}")))
}

fn map_pointer(symbol: &str) -> Option<WordNetRelation> {
    match symbol {
        "!" => Some(WordNetRelation::Antonym),
        "@" => Some(WordNetRelation::Hypernym),
        "@i" => Some(WordNetRelation::InstanceHypernym),
        "~" => Some(WordNetRelation::Hyponym),
        "~i" => Some(WordNetRelation::InstanceHyponym),
        "#m" => Some(WordNetRelation::MemberHolonym),
        "#s" => Some(WordNetRelation::SubstanceHolonym),
        "#p" => Some(WordNetRelation::PartHolonym),
        "%m" => Some(WordNetRelation::MemberMeronym),
        "%s" => Some(WordNetRelation::SubstanceMeronym),
        "%p" => Some(WordNetRelation::PartMeronym),
        "=" => Some(WordNetRelation::Attribute),
        "+" => Some(WordNetRelation::DerivationallyRelated),
        "*" => Some(WordNetRelation::Entailment),
        ">" => Some(WordNetRelation::Cause),
        "^" => Some(WordNetRelation::AlsoSee),
        "$" => Some(WordNetRelation::VerbGroup),
        "&" => Some(WordNetRelation::SimilarTo),
        "<" => Some(WordNetRelation::Participle),
        "\\" => Some(WordNetRelation::Pertainym),
        ";c" => Some(WordNetRelation::DomainTopic),
        "-c" => Some(WordNetRelation::MemberOfDomainTopic),
        ";r" => Some(WordNetRelation::DomainRegion),
        "-r" => Some(WordNetRelation::MemberOfDomainRegion),
        ";u" => Some(WordNetRelation::DomainUsage),
        "-u" => Some(WordNetRelation::MemberOfDomainUsage),
        _ => None,
    }
}

struct Tokenizer<'a> {
    line: &'a str,
    file_name: &'a str,
    line_number: usize,
    position: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(line: &'a str, file_name: &'a str, line_number: usize) -> Self {
        Self { line, file_name, line_number, position: 0 }
    }

    fn next_field(&mut self, field: &str) -> Result<&'a str, WordNetError> {
        let bytes = self.line.as_bytes();
        while self.position < bytes.len() && bytes[self.position] == b' ' {
            self.position += 1;
        }
        if self.position >= bytes.len() {
            return Err(malformed(self.file_name, self.line_number, format!("Truncated line, missing field: {field}")));
        }
        let start = self.position;
        while self.position < bytes.len() && bytes[self.position] != b' ' {
            self.position += 1;
        }
        Ok(&self.line[start..self.position])
    }

    fn next_non_negative_int(&mut self, field: &str, label: &str) -> Result<usize, WordNetError> {
        let token = self.next_field(field)?;
        if token.starts_with('-') {
            return Err(malformed(
                self.file_name,
                self.line_number,
                format!("{label} must not be negative, got: {token}"),
            ));
        }
        if !token.bytes().all(|b| b.is_ascii_digit()) {
            return Err(malformed(
                self.file_name,
                self.line_number,
                format!("{field} is not an unsigned decimal integer: {token}"),
            ));
        }
        let val: i32 = token.parse().map_err(|_| {
            malformed(self.file_name, self.line_number, format!("{field} is not an unsigned decimal integer: {token}"))
        })?;
        Ok(val as usize)
    }

    fn next_fixed_int(&mut self, field: &str, radix: u32, width: usize) -> Result<usize, WordNetError> {
        let token = self.next_field(field)?;
        self.parse_fixed_int(field, radix, width, token)
    }

    fn next_fixed_count(&mut self, field: &str, radix: u32, width: usize, label: &str) -> Result<usize, WordNetError> {
        let token = self.next_field(field)?;
        if token.starts_with('-') {
            return Err(malformed(
                self.file_name,
                self.line_number,
                format!("{label} must not be negative, got: {token}"),
            ));
        }
        self.parse_fixed_int(field, radix, width, token)
    }

    fn parse_fixed_int(&self, field: &str, radix: u32, width: usize, token: &str) -> Result<usize, WordNetError> {
        if token.len() != width || !token.chars().all(|c| c.is_digit(radix)) {
            return Err(malformed(
                self.file_name,
                self.line_number,
                format!("{field} must be a {width}-digit base-{radix} integer, got: {token}"),
            ));
        }
        usize::from_str_radix(token, radix).map_err(|_| {
            malformed(
                self.file_name,
                self.line_number,
                format!("{field} must be a {width}-digit base-{radix} integer, got: {token}"),
            )
        })
    }

    fn remaining_fixed_width_fields(&self, width: usize) -> usize {
        (self.line.len().saturating_sub(self.position)) / (width + 1)
    }

    fn require_end(&mut self, parsed_fields: &str) -> Result<(), WordNetError> {
        let bytes = self.line.as_bytes();
        while self.position < bytes.len() && bytes[self.position] == b' ' {
            self.position += 1;
        }
        if self.position < bytes.len() {
            let extra = self.next_field("field")?;
            return Err(malformed(
                self.file_name,
                self.line_number,
                format!("Unexpected field after {parsed_fields}: {extra}"),
            ));
        }
        Ok(())
    }

    fn gloss(&mut self) -> Result<String, WordNetError> {
        let separator = self.next_field("gloss separator")?;
        if separator != "|" {
            return Err(malformed(
                self.file_name,
                self.line_number,
                format!("Expected the | gloss separator, got: {separator}"),
            ));
        }
        let bytes = self.line.as_bytes();
        let mut start = self.position;
        while start < bytes.len() && bytes[start] == b' ' {
            start += 1;
        }
        let mut end = bytes.len();
        while end > start && bytes[end - 1] == b' ' {
            end -= 1;
        }
        Ok(self.line[start..end].to_string())
    }

    fn malformed_token(&self, message: impl AsRef<str>) -> WordNetError {
        malformed(self.file_name, self.line_number, message)
    }
}
