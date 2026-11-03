//! CoNLL-U reading for the peer analysis benchmark: word lines with
//! sequential ids and in-sentence heads, multiword ranges and empty
//! nodes skipped after their own shape is checked.

use crate::Result;

/// One CoNLL-U token row (word lines only; multiword-token ranges and
/// empty nodes are skipped by the parser).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConlluToken {
    pub form: String,
    pub lemma: String,
    pub upos: String,
    pub xpos: String,
    pub feats: String,
    pub head: Option<i64>,
    pub deprel: String,
}

/// One CoNLL-U sentence: its `# text` line when present, and its tokens.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConlluSentence {
    pub text: Option<String>,
    pub tokens: Vec<ConlluToken>,
    /// Whether a `# newdoc` comment starts a document at this sentence.
    pub new_doc: bool,
}

/// Reads CoNLL-U sentences. Lines with a range (`1-2`) or decimal
/// (`3.1`) id are not word lines and are skipped; a word line must have
/// the ten tab-separated fields.
pub fn parse_conllu(text: &str) -> Result<Vec<ConlluSentence>> {
    let mut out = Vec::new();
    let mut current = ConlluSentence::default();
    let mut started = false;
    let mut token_lines: Vec<usize> = Vec::new();
    let mut expected_id = 1u32;
    for (line_no, line) in text.lines().enumerate() {
        let line_no = line_no + 1;
        if line.is_empty() {
            if started {
                validate_heads(&current, &token_lines)?;
                out.push(std::mem::take(&mut current));
                token_lines.clear();
                expected_id = 1;
                started = false;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            started = true;
            if let Some(t) = rest.trim().strip_prefix("text =") {
                current.text = Some(t.trim().to_string());
            }
            if rest.trim().starts_with("newdoc") {
                current.new_doc = true;
            }
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 10 {
            return Err(format!("line {line_no}: expected 10 CoNLL-U fields, got {}", fields.len()));
        }
        if fields[0].contains('-') || fields[0].contains('.') {
            // A multiword range covers the next word ids; an empty node
            // sits between them. Neither is a scored word, but a range
            // must start where the next word id belongs.
            if let Some((a, b)) = fields[0].split_once('-') {
                let (a, b) = (a.parse::<u32>(), b.parse::<u32>());
                match (a, b) {
                    (Ok(a), Ok(b)) if a == expected_id && b >= a => {}
                    _ => {
                        return Err(format!(
                            "line {line_no}: multiword id {:?} does not cover word {expected_id}",
                            fields[0]
                        ));
                    }
                }
            }
            started = true;
            continue;
        }
        let id = fields[0]
            .parse::<u32>()
            .map_err(|_| format!("line {line_no}: token id {:?} is not a positive integer", fields[0]))?;
        if id != expected_id {
            return Err(format!("line {line_no}: token id {id}, expected {expected_id}"));
        }
        expected_id += 1;
        let head = match fields[6] {
            "_" => None,
            raw => Some(raw.parse::<i64>().map_err(|_| format!("line {line_no}: head {raw:?} is not an integer"))?),
        };
        started = true;
        token_lines.push(line_no);
        current.tokens.push(ConlluToken {
            form: fields[1].to_string(),
            lemma: fields[2].to_string(),
            upos: fields[3].to_string(),
            xpos: fields[4].to_string(),
            feats: fields[5].to_string(),
            head,
            deprel: fields[7].to_string(),
        });
    }
    if started {
        validate_heads(&current, &token_lines)?;
        out.push(current);
    }
    if out.is_empty() {
        return Err("no CoNLL-U sentences".into());
    }
    Ok(out)
}

/// Every stated head must name the root (0) or another word of its
/// sentence, so flattening can safely read HEAD as a row position.
fn validate_heads(sentence: &ConlluSentence, token_lines: &[usize]) -> Result<()> {
    let len = sentence.tokens.len() as i64;
    for (i, token) in sentence.tokens.iter().enumerate() {
        if let Some(head) = token.head
            && (head < 0 || head > len || head == i as i64 + 1)
        {
            return Err(format!("line {}: head {head} does not name the root or another word", token_lines[i]));
        }
    }
    Ok(())
}
