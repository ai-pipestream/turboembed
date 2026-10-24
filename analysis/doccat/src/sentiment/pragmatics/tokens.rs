use super::*;

pub(super) fn sentence_token_range(sentence_index: usize, sentence_span: Span, tokens: &Tokens) -> IxRange<Tokens> {
    if let Some(tok_sents) = &tokens.sentence {
        let target = Ix::<Sentences>::new(sentence_index as u32);
        let a = tok_sents.partition_point(|&s| s < target);
        let b = a + tok_sents[a..].partition_point(|&s| s == target);
        return IxRange::new(a as u32, b as u32);
    }
    let n = tokens.len();
    let mut a = 0usize;
    while a < n && tokens.spans.get(Ix::new(a as u32)).end <= sentence_span.start {
        a += 1;
    }
    let mut b = a;
    while b < n {
        let sp = tokens.spans.get(Ix::new(b as u32));
        if sp.start >= sentence_span.end || sp.end > sentence_span.end {
            break;
        }
        b += 1;
    }
    IxRange::new(a as u32, b as u32)
}

pub(super) fn emoji_polarity(sequence: &str, cldr_name: &str) -> i8 {
    if let Some(ann) = opennlp_annotation(sequence)
        && ann.sentiment != 0
    {
        return ann.sentiment;
    }
    for ch in sequence.chars() {
        match ch {
            // Positive emoji common in sarcastic contrast ("everything is on fire 😂🎉👍")
            '\u{1F602}' | '\u{1F923}' | '\u{1F973}' | '\u{1F389}' | '\u{1F38A}' | '\u{1F60D}' | '\u{1F929}'
            | '\u{1F970}' | '\u{1F606}' | '\u{1F607}' => return 2,
            '\u{1F44D}' | '\u{1F44F}' | '\u{1F64C}' | '\u{1F525}' | '\u{2728}' | '\u{1F4AF}' | '\u{1F60E}'
            | '\u{1F60B}' => return 1,
            // Negative / eye-roll / exasperated emoji common in sarcastic contrast ("just fantastic 🙄😒")
            '\u{1F644}' | '\u{1F612}' | '\u{1F621}' | '\u{1F620}' | '\u{1F92C}' | '\u{1F92E}' | '\u{1F922}'
            | '\u{1F480}' | '\u{1F4A9}' => return -2,
            '\u{1F643}' | '\u{1F44E}' | '\u{1F624}' | '\u{1F629}' | '\u{1F62B}' | '\u{1F61E}' | '\u{1F616}'
            | '\u{1F623}' | '\u{1F613}' | '\u{1F614}' => return -1,
            _ => {}
        }
    }
    if !cldr_name.is_empty() {
        if cldr_name.contains("rolling eyes")
            || cldr_name.contains("unamused")
            || cldr_name.contains("angry")
            || cldr_name.contains("enraged")
            || cldr_name.contains("vomiting")
            || cldr_name.contains("nauseated")
            || cldr_name.contains("crying")
            || cldr_name.contains("broken heart")
        {
            return -2;
        }
        if cldr_name.contains("frowning")
            || cldr_name.contains("confused")
            || cldr_name.contains("disappointed")
            || cldr_name.contains("weary")
            || cldr_name.contains("tired")
            || cldr_name.contains("thumbs down")
            || cldr_name.contains("upside-down")
        {
            return -1;
        }
        if cldr_name.contains("tears of joy")
            || cldr_name.contains("rolling on the floor laughing")
            || cldr_name.contains("party popper")
            || cldr_name.contains("partying face")
            || cldr_name.contains("heart-eyes")
            || cldr_name.contains("star-struck")
            || cldr_name.contains("grinning")
        {
            return 2;
        }
        if cldr_name.contains("smiling")
            || cldr_name.contains("thumbs up")
            || cldr_name.contains("clapping")
            || cldr_name.contains("raising hands")
        {
            return 1;
        }
    }
    0
}

pub(super) fn collect_sentence_emoji(
    sentence_span: Span,
    tok_range: IxRange<Tokens>,
    text: &str,
    tokens: &Tokens,
    emoji: Option<&Emoji>,
) -> Vec<(IxRange<Tokens>, i8)> {
    let mut out = Vec::new();
    let start = tok_range.start.get();
    let end = tok_range.end.get();
    if start >= end {
        return out;
    }

    if let Some(em_layer) = emoji {
        for (em_ix, em_span) in em_layer.spans.iter() {
            if em_span.start < sentence_span.start || em_span.end > sentence_span.end {
                continue;
            }
            let seq = &text[em_span.range()];
            let cldr_name = em_layer.name.get(em_ix.get());
            let pol = emoji_polarity(seq, cldr_name);
            if pol == 0 {
                continue;
            }
            if let Some(r) = byte_span_to_token_range(em_span, tok_range, tokens) {
                out.push((r, pol));
            }
        }
    }

    // Also inspect tokens directly in case `EmojiAnnotate` was not run ahead
    // of `PragmaticsAnnotator`.
    for idx in start..end {
        let r = IxRange::new(idx as u32, (idx + 1) as u32);
        if overlaps_any_emoji(r, &out) {
            continue;
        }
        let sp = tokens.spans.get(Ix::new(idx as u32));
        let surface = &text[sp.range()];
        let pol = emoji_polarity(surface, "");
        if pol != 0 {
            out.push((r, pol));
        }
    }

    out
}

pub(super) fn byte_span_to_token_range(
    span: Span,
    sent_tok_range: IxRange<Tokens>,
    tokens: &Tokens,
) -> Option<IxRange<Tokens>> {
    let start = sent_tok_range.start.get();
    let end = sent_tok_range.end.get();
    let mut first = None;
    let mut last = None;
    for i in start..end {
        let tsp = tokens.spans.get(Ix::new(i as u32));
        if tsp.end > span.start && tsp.start < span.end {
            if first.is_none() {
                first = Some(i as u32);
            }
            last = Some((i + 1) as u32);
        }
    }
    match (first, last) {
        (Some(a), Some(b)) if a < b => Some(IxRange::new(a, b)),
        _ => None,
    }
}

pub(super) fn first_non_emoji_token_range(
    tok_range: IxRange<Tokens>,
    emoji_spans: &[(IxRange<Tokens>, i8)],
) -> Option<IxRange<Tokens>> {
    let start = tok_range.start.get() as u32;
    let end = tok_range.end.get() as u32;
    let mut a = None;
    let mut b = None;
    for t in start..end {
        let r = IxRange::new(t, t + 1);
        if !overlaps_any_emoji(r, emoji_spans) {
            if a.is_none() {
                a = Some(t);
            }
            b = Some(t + 1);
        }
    }
    match (a, b) {
        (Some(x), Some(y)) if x < y => Some(IxRange::new(x, y)),
        _ => None,
    }
}

pub(super) fn overlaps_any_emoji(range: IxRange<Tokens>, emoji_spans: &[(IxRange<Tokens>, i8)]) -> bool {
    emoji_spans.iter().any(|&(er, _)| ranges_overlap(range, er))
}

pub(super) fn ranges_overlap(a: IxRange<Tokens>, b: IxRange<Tokens>) -> bool {
    a.start.get() < b.end.get() && b.start.get() < a.end.get()
}

pub(super) fn has_preceding_sarcastic_cue(
    range: IxRange<Tokens>,
    sent_range: IxRange<Tokens>,
    lower_tokens: &[String],
) -> bool {
    let s = sent_range.start.get();
    let pos_start = range.start.get();
    let window_start = pos_start.saturating_sub(2).max(s);
    for w in &lower_tokens[window_start..pos_start] {
        if SARCASTIC_PRE_CUES.contains(&w.as_str()) {
            return true;
        }
    }
    false
}

pub(super) fn has_preceding_negation(token_index: usize, sent_start: usize, lower_tokens: &[String]) -> bool {
    let window_start = token_index.saturating_sub(3).max(sent_start);
    for w in &lower_tokens[window_start..token_index] {
        if matches!(
            w.as_str(),
            "not"
                | "never"
                | "no"
                | "cannot"
                | "cant"
                | "can't"
                | "dont"
                | "don't"
                | "didnt"
                | "didn't"
                | "doesnt"
                | "doesn't"
                | "isnt"
                | "isn't"
                | "wasnt"
                | "wasn't"
                | "wont"
                | "won't"
                | "wouldnt"
                | "wouldn't"
        ) {
            return true;
        }
    }
    false
}

pub(super) fn is_clause_final_tag(
    pos_range: IxRange<Tokens>,
    sent_range: IxRange<Tokens>,
    lower_tokens: &[String],
) -> bool {
    let len = pos_range.end.get() - pos_range.start.get();
    if len == 0 || len > 2 {
        return false;
    }
    let last_word = &lower_tokens[pos_range.end.get() - 1];
    if !CLAUSE_FINAL_POSITIVE_TAGS.contains(&last_word.as_str()) {
        return false;
    }
    // Check that only punctuation (empty cleaned tokens) follows `pos_range`
    // up to the end of the sentence.
    lower_tokens[pos_range.end.get()..sent_range.end.get()].iter().all(|w| w.is_empty())
}

pub(super) fn clean_token(raw: &str) -> String {
    crate::sentiment::token_key::token_key_strict(raw.trim(), false)
}
