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
    pub start: u32,
    pub end: u32,
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
        let mut open: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
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

            if !text.is_empty() {
                text.push(' ');
            }
            let token_start = text.len() as u32;
            text.push_str(fields[3].text);
            let token_end = text.len() as u32;

            tokens.push(CorefToken { form: fields[3].field(), sentence, index, start: token_start, end: token_end });
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
                            open.entry(chain).or_default().push((tokens.len() as u32 - 1, line.number));
                        }
                    } else if let Some(id) = part.strip_suffix(')') {
                        let chain: u32 = id.parse().map_err(|_| {
                            FormatError::Ner(super::NerError::LitbankCorefChain {
                                line: line.number,
                                text: part.to_owned(),
                            })
                        })?;
                        let Some(stack) = open.get_mut(&chain) else {
                            return Err(FormatError::Ner(super::NerError::LitbankCorefChain {
                                line: line.number,
                                text: format!("unmatched closing bracket for chain {chain}"),
                            }));
                        };
                        let Some((start, _)) = stack.pop() else {
                            return Err(FormatError::Ner(super::NerError::LitbankCorefChain {
                                line: line.number,
                                text: format!("unmatched closing bracket for chain {chain}"),
                            }));
                        };
                        let end = index as usize - 1;
                        mentions.push(Mention {
                            chain,
                            tokens: start..index,
                            text: Span::new(tokens[start as usize].start, tokens[end].end),
                        });
                    }
                }
            }
        }

        if let Some((&chain, stack)) = open.iter().find(|(_, stack)| !stack.is_empty()) {
            let opening_line = stack.last().map(|(_, line)| *line).unwrap_or(0);
            return Err(FormatError::Ner(super::NerError::LitbankCorefChain {
                line: opening_line,
                text: format!("unclosed opening bracket for chain {chain}"),
            }));
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
    use crate::ner::NerError;

    const TEXT: &str = "#begin document (test); part 0\ntest\t0\t0\tHello\t_\t_\t_\t_\t_\t_\t_\t_\t(0)\ntest\t0\t1\tworld\t_\t_\t_\t_\t_\t_\t_\t_\t_\n\n#end document\n";

    #[test]
    fn reads_singleton_mention() {
        let mut reader = LitbankCorefReader::new(TEXT).unwrap();
        let doc = reader.read_document().unwrap().unwrap();
        assert_eq!(doc.tokens.len(), 2);
        assert_eq!(doc.mentions.len(), 1);
        assert_eq!(doc.mentions[0].chain, 0);
    }

    #[test]
    fn multi_token_mention_spans_first_to_last_token() {
        let text = "#begin document (test); part 0\ntest\t0\t0\tJohn\t_\t_\t_\t_\t_\t_\t_\t_\t(0\ntest\t0\t1\tSmith\t_\t_\t_\t_\t_\t_\t_\t_\t0)\n\n#end document\n";
        let mut reader = LitbankCorefReader::new(text).unwrap();
        let doc = reader.read_document().unwrap().unwrap();
        assert_eq!(doc.mentions.len(), 1);
        assert_eq!(doc.mentions[0].tokens, 0..2);
        assert_eq!(doc.mentions[0].text, Span::new(0, 10));
    }

    #[test]
    fn nested_same_chain_mentions() {
        let text = "#begin document (test); part 0\ntest\t0\t0\this\t_\t_\t_\t_\t_\t_\t_\t_\t(0|(0\ntest\t0\t1\tsister\t_\t_\t_\t_\t_\t_\t_\t_\t0)|0)\n\n#end document\n";
        let mut reader = LitbankCorefReader::new(text).unwrap();
        let doc = reader.read_document().unwrap().unwrap();
        assert_eq!(doc.mentions.len(), 2);
        let inner = &doc.mentions[0];
        assert_eq!(inner.tokens, 0..2);
        assert_eq!(inner.text, Span::new(0, 10));
        let outer = &doc.mentions[1];
        assert_eq!(outer.tokens, 0..2);
        assert_eq!(outer.text, Span::new(0, 10));
    }

    #[test]
    fn nested_inner_mention_ends_on_its_closing_token() {
        let text = "#begin document (test); part 0\ntest\t0\t0\ta\t_\t_\t_\t_\t_\t_\t_\t_\t(0\ntest\t0\t1\tb\t_\t_\t_\t_\t_\t_\t_\t_\t_\ntest\t0\t2\tc\t_\t_\t_\t_\t_\t_\t_\t_\t_\ntest\t0\t3\td\t_\t_\t_\t_\t_\t_\t_\t_\t(0\ntest\t0\t4\te\t_\t_\t_\t_\t_\t_\t_\t_\t_\ntest\t0\t5\tf\t_\t_\t_\t_\t_\t_\t_\t_\t0)\ntest\t0\t6\tg\t_\t_\t_\t_\t_\t_\t_\t_\t_\ntest\t0\t7\th\t_\t_\t_\t_\t_\t_\t_\t_\t0)\n\n#end document\n";
        let mut reader = LitbankCorefReader::new(text).unwrap();
        let doc = reader.read_document().unwrap().unwrap();
        assert_eq!(doc.mentions.len(), 2);
        assert_eq!(doc.mentions[0].tokens, 3..6);
        assert_eq!(doc.mentions[0].text, Span::new(6, 11));
        assert_eq!(doc.mentions[1].tokens, 0..8);
        assert_eq!(doc.mentions[1].text, Span::new(0, 15));
    }

    #[test]
    fn unmatched_closing_bracket_is_an_error() {
        let text = "#begin document (test); part 0\ntest\t0\t0\tHello\t_\t_\t_\t_\t_\t_\t_\t_\t0)\n\n#end document\n";
        let mut reader = LitbankCorefReader::new(text).unwrap();
        let err = reader.read_document().unwrap_err();
        match err {
            FormatError::Ner(NerError::LitbankCorefChain { line, .. }) => assert_eq!(line, 2),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn unclosed_opening_bracket_is_an_error() {
        let text = "#begin document (test); part 0\ntest\t0\t0\tHello\t_\t_\t_\t_\t_\t_\t_\t_\t(0\n\n#end document\n";
        let mut reader = LitbankCorefReader::new(text).unwrap();
        let err = reader.read_document().unwrap_err();
        match err {
            FormatError::Ner(NerError::LitbankCorefChain { line, .. }) => assert_eq!(line, 2),
            other => panic!("unexpected error: {other}"),
        }
    }
}
