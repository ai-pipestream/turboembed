use std::collections::{HashMap, HashSet};

use super::{malformed, synset_id};
use crate::error::WordNetError;
use crate::lexicon::LemmaKey;
use crate::pos::WordNetPos;
use crate::relation::WordNetRelation;

const INDEX_POINTER_SYMBOLS: [&str; 20] =
    ["!", "@", "~", "#m", "#s", "#p", "%m", "%s", "%p", "=", "+", "*", ">", "^", "$", "&", "<", "\\", ";", "-"];

#[allow(clippy::too_many_arguments)]
pub(super) fn parse_index_line(
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
    if pointer_type_count > tokens.remaining_fixed_width_fields(1) {
        return Err(malformed(
            file_name,
            line_number,
            format!("Pointer count {pointer_type_count} exceeds the available pointer fields"),
        ));
    }
    let mut pointer_types = HashSet::with_capacity(pointer_type_count.min(INDEX_POINTER_SYMBOLS.len()));
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

pub(super) fn clean_lemma(
    word: &str,
    file_name: &str,
    line_number: usize,
    file_pos: WordNetPos,
) -> Result<String, WordNetError> {
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
    if !cleaned.chars().any(|c| c != '_') {
        return Err(malformed(file_name, line_number, "Empty word field"));
    }
    Ok(cleaned.replace('_', " "))
}

pub(super) fn parse_offset(offset: &str, tokens: &Tokenizer<'_>) -> Result<usize, WordNetError> {
    if offset.len() != 8 || !offset.bytes().all(|b| b.is_ascii_digit()) {
        return Err(tokens.malformed_token(format!("Synset offset must be 8 digits, got: {offset}")));
    }
    offset
        .parse::<usize>()
        .map_err(|_| tokens.malformed_token(format!("Synset offset must be 8 digits, got: {offset}")))
}

pub(super) fn parse_pointer_pos_char(pos: &str, tokens: &Tokenizer<'_>) -> Result<char, WordNetError> {
    if pos.len() == 1 {
        let c = pos.chars().next().unwrap();
        if matches!(c, 'n' | 'v' | 'a' | 'r') {
            return Ok(c);
        }
    }
    Err(tokens.malformed_token(format!("Pointer pos must be one of n, v, a, r, got: {pos}")))
}

pub(super) fn map_pointer(symbol: &str) -> Option<WordNetRelation> {
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

pub(super) struct Tokenizer<'a> {
    line: &'a str,
    file_name: &'a str,
    line_number: usize,
    position: usize,
}

impl<'a> Tokenizer<'a> {
    pub(super) fn new(line: &'a str, file_name: &'a str, line_number: usize) -> Self {
        Self { line, file_name, line_number, position: 0 }
    }

    pub(super) fn next_field(&mut self, field: &str) -> Result<&'a str, WordNetError> {
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

    pub(super) fn next_non_negative_int(&mut self, field: &str, label: &str) -> Result<usize, WordNetError> {
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

    pub(super) fn next_fixed_int(&mut self, field: &str, radix: u32, width: usize) -> Result<usize, WordNetError> {
        let token = self.next_field(field)?;
        self.parse_fixed_int(field, radix, width, token)
    }

    pub(super) fn next_fixed_count(
        &mut self,
        field: &str,
        radix: u32,
        width: usize,
        label: &str,
    ) -> Result<usize, WordNetError> {
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

    pub(super) fn remaining_fixed_width_fields(&self, width: usize) -> usize {
        (self.line.len().saturating_sub(self.position)) / (width + 1)
    }

    pub(super) fn require_end(&mut self, parsed_fields: &str) -> Result<(), WordNetError> {
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

    pub(super) fn gloss(&mut self) -> Result<String, WordNetError> {
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

    pub(super) fn malformed_token(&self, message: impl AsRef<str>) -> WordNetError {
        malformed(self.file_name, self.line_number, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_index_line_rejects_oversized_pointer_count_without_allocating() {
        let raw_index = HashMap::new();
        let mut senses = HashMap::new();
        let err = parse_index_line(
            "berry n 1 2147483647 1 0 00001564",
            "index.noun",
            1,
            WordNetPos::Noun,
            'n',
            "noun",
            &raw_index,
            &mut senses,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Pointer count 2147483647 exceeds the available pointer fields"), "{err}");
    }

    #[test]
    fn clean_lemma_rejects_underscore_only_words() {
        for word in ["_", "__", "_(p)"] {
            let err = clean_lemma(word, "data.adj", 1, WordNetPos::Adjective).unwrap_err();
            assert!(err.to_string().contains("Empty word field"), "{word}: {err}");
        }
    }
}
