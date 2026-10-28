//! LitBank coreference in CoNLL-2012 format.

use std::collections::HashMap;
use std::ops::Range;

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::source::{Field, Lines, split_fields};

/// A token of a LitBank CoNLL coreference document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorefToken<'a> {
    pub form: Field<'a>,
    pub sentence: u32,
    pub index: u32,
}

/// A mention of an entity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mention {
    pub chain: u32,
    pub tokens: Range<u32>,
    pub text: Span,
}

/// A document with gold coreference.
#[derive(Clone, Debug)]
pub struct LitbankCorefDocument<'a> {
    pub tokens: Vec<CorefToken<'a>>,
    pub mentions: Vec<Mention>,
    pub text: String,
}

/// Reads LitBank CoNLL-2012 coreference annotations.
pub struct LitbankCorefReader<'a> {
    lines: Lines<'a>,
}

impl<'a> LitbankCorefReader<'a> {
    /// A reader of `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(source: &'a str) -> Result<LitbankCorefReader<'a>, FormatError> {
        Ok(LitbankCorefReader { lines: Lines::new(source)? })
    }

    /// Reads the next document.
    ///
    /// # Errors
    ///
    /// [`FormatError::Ner`] for malformed coreference lines.
    pub fn read_document(&mut self) -> Result<Option<LitbankCorefDocument<'a>>, FormatError> {
        let mut tokens = Vec::new();
        let mut text = String::new();
        let mut mentions = Vec::new();
        let mut open: HashMap<u32, u32> = HashMap::new();
        let mut sentence = 0u32;
        let mut index = 0u32;
        let mut began = false;

        for line in self.lines.by_ref() {
            let text_line = line.text();
            if text_line.starts_with("#begin document") {
                began = true;
                continue;
            }
            if text_line.starts_with("#end document") {
                break;
            }
            if text_line.is_empty() {
                if began {
                    sentence += 1;
                }
                continue;
            }
            if text_line.starts_with('#') {
                continue;
            }
            began = true;

            let fields = split_fields(line.piece, '\t');
            if fields.len() < 12 {
                return Err(FormatError::Ner(super::NerError::LitbankCorefFields {
                    line: line.number,
                    count: fields.len(),
                    text: text_line.to_owned(),
                }));
            }

            let token_start = text.len() as u32;
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(fields[3].text);
            let token_end = text.len() as u32;

            tokens.push(CorefToken { form: fields[3].field(), sentence, index });
            index += 1;

            let coref = fields[fields.len() - 1].text;
            if coref != "_" && coref != "-" {
                for part in coref.split('|') {
                    if let Some(id) = part.strip_prefix('(') {
                        if let Some(id) = id.strip_suffix(')') {
                            let chain: u32 = id.parse().map_err(|_| {
                                FormatError::Ner(super::NerError::LitbankCorefChain {
                                    line: line.number,
                                    text: part.to_owned(),
                                })
                            })?;
                            mentions.push(Mention {
                                chain,
                                tokens: (tokens.len() as u32 - 1)..(tokens.len() as u32),
                                text: Span::new(token_start, token_end),
                            });
                        } else {
                            let chain: u32 = id.parse().map_err(|_| {
                                FormatError::Ner(super::NerError::LitbankCorefChain {
                                    line: line.number,
                                    text: part.to_owned(),
                                })
                            })?;
                            open.insert(chain, tokens.len() as u32 - 1);
                        }
                    } else if let Some(id) = part.strip_suffix(')') {
                        let chain: u32 = id.parse().map_err(|_| {
                            FormatError::Ner(super::NerError::LitbankCorefChain {
                                line: line.number,
                                text: part.to_owned(),
                            })
                        })?;
                        if let Some(start) = open.remove(&chain) {
                            mentions.push(Mention {
                                chain,
                                tokens: start..(tokens.len() as u32),
                                text: Span::new(token_start, token_end),
                            });
                        }
                    }
                }
            }
        }

        if tokens.is_empty() {
            return Ok(None);
        }
        Ok(Some(LitbankCorefDocument { tokens, mentions, text }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "#begin document (test); part 0\ntest\t0\t0\tHello\t_\t_\t_\t_\t_\t_\t_\t_\t(0)\ntest\t0\t1\tworld\t_\t_\t_\t_\t_\t_\t_\t_\t_\n\n#end document\n";

    #[test]
    fn reads_singleton_mention() {
        let mut reader = LitbankCorefReader::new(TEXT).unwrap();
        let doc = reader.read_document().unwrap().unwrap();
        assert_eq!(doc.tokens.len(), 2);
        assert_eq!(doc.mentions.len(), 1);
        assert_eq!(doc.mentions[0].chain, 0);
    }
}
