use crate::{
    manifest::{Family, PromptRole, Qa, Truncation},
    status::{Error, INVALID_ARGUMENT, Result},
    tokenizer::{Encode, Tokenizer},
};
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct QaWindow {
    pub ids: Vec<i32>,
    pub types: Vec<i32>,
    pub context: Vec<Option<Range<usize>>>,
    pub max_context: Vec<bool>,
}

/// Encode overlapping joint inputs. Questions use a separate truncation budget;
/// context offsets always refer to the original context bytes.
pub fn windows(tok: &Tokenizer, family: Family, q: &Qa, question: &str, text: &str) -> Result<Vec<QaWindow>> {
    let invalid = |s: &str| Error::new(INVALID_ARGUMENT, s);
    let specials = if family == Family::Roberta { 4 } else { 3 };
    if q.max_query_tokens == 0
        || q.max_seq > tok.max_seq
        || q.max_seq <= specials + 1
        || q.max_query_tokens >= q.max_seq - specials
    {
        return Err(invalid("QA query and sequence budgets leave no context capacity"));
    }
    let opts = Encode {
        add_special_tokens: false,
        truncation: Truncation::Right,
        max_tokens: q.max_query_tokens,
        prompt: PromptRole::None,
    };
    let query = tok.encode(question, opts)?;
    let mut ids = Vec::new();
    let mut spans = Vec::new();
    tok.encode_spans(
        text,
        Encode { max_tokens: u32::MAX, truncation: Truncation::None, ..opts },
        &mut ids,
        &mut spans,
    )?;
    let room = q.max_seq as usize - specials as usize - query.len();
    if q.doc_stride as usize >= room {
        return Err(invalid("QA doc_stride overlap must be smaller than context capacity"));
    }
    let (head, tail) = tok.template_ends();
    if head.len() != 1 || tail.len() != 1 {
        return Err(invalid("QA needs one start and end special token"));
    }
    let step = room - q.doc_stride as usize;
    let mut intervals = Vec::new();
    let mut start = 0usize;
    loop {
        let end = start.saturating_add(room).min(ids.len());
        intervals.push(start..end);
        if end == ids.len() {
            break;
        }
        start += step;
    }
    let mut owners = vec![(0usize, (0usize, 0usize)); ids.len()];
    for (wi, range) in intervals.iter().enumerate() {
        for i in range.clone() {
            // Interior context wins; length only breaks equal-distance ties.
            let score = ((i - range.start).min(range.end - i - 1), range.len());
            if score > owners[i].1 {
                owners[i] = (wi, score);
            }
        }
    }
    let mut result = Vec::with_capacity(intervals.len());
    for (wi, range) in intervals.into_iter().enumerate() {
        let mut row = vec![head[0]];
        row.extend_from_slice(&query);
        row.push(tail[0]);
        if family == Family::Roberta {
            row.push(tail[0]);
        }
        let boundary = row.len();
        let mut context = vec![None; boundary];
        let mut max_context = vec![false; boundary];
        for i in range.clone() {
            let [a, b] = spans[i];
            if a >= b || text.get(a as usize..b as usize).is_none() {
                return Err(invalid("QA tokenizer offset is empty, outside context, or splits UTF-8"));
            }
            context.push(Some(a as usize..b as usize));
            max_context.push(owners[i].0 == wi);
        }
        row.extend_from_slice(&ids[range]);
        row.push(tail[0]);
        context.push(None);
        max_context.push(false);
        let mut types = vec![0; row.len()];
        if family == Family::Bert {
            types[boundary..].fill(1);
        }
        result.push(QaWindow { ids: row, types, context, max_context });
    }
    Ok(result)
}
