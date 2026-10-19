//! Princeton WordNet WNDB 3.0 / 3.1 directory reader (`WndbReader`).

mod parse;

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use self::parse::{Tokenizer, clean_lemma, map_pointer, parse_index_line, parse_offset, parse_pointer_pos_char};
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

pub(super) fn synset_id(offset: &str, pos_char: char) -> String {
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

pub(super) fn malformed(file_name: &str, line_number: usize, message: impl AsRef<str>) -> WordNetError {
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
