//! Conversion between [`SyntacticSentence`]s and [`Document`] layers
//! ([`Sentences`], [`Tokens`], [`Words`], [`WordPosTags`], [`WordLemmas`],
//! [`MorphFeatures`] and [`WordDependencies`]).

use std::fmt::Write as _;

use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{MorphFeatures, Sentences, Strs, Tokens, WordDependencies, WordLemmas, WordPosTags, Words};

use super::WordLine;
use super::words::{SyntacticSentence, escape_conllu_spaces};
use crate::error::FormatError;

impl<'a> SyntacticSentence<'a> {
    /// Converts this sentence into a [`Document`] populated with [`Sentences`],
    /// [`Tokens`], [`Words`], [`WordPosTags`], [`WordLemmas`],
    /// [`MorphFeatures`] and [`WordDependencies`].
    pub fn to_document(&self) -> Result<Document, FormatError> {
        let surface_text = match &self.text {
            Some(t) => t.as_str().to_owned(),
            None => self.reconstruct_surface_text(),
        };

        let mut token_spans = Spans::<Tokens>::new(ViewId::SOURCE);
        let mut words = Words::new();
        let mut search = 0usize;

        let mut w_idx = 0usize;
        let mut mwt_iter = self.multiwords.iter().peekable();
        while w_idx < self.words.len() {
            if let Some(mwt) = mwt_iter.peek()
                && mwt.words.start.get() == w_idx
            {
                let mwt = mwt_iter.next().unwrap();
                let surface = mwt.line_data.form.as_str();
                let Some(at) = surface_text[search..].find(surface).map(|i| search + i) else {
                    return Err(FormatError::TokenNotInText {
                        line: mwt.line_data.line,
                        token: surface.to_owned(),
                        sentence: self.sentence_id.as_ref().map(|f| f.as_str().to_owned()),
                        text: surface_text,
                    });
                };
                let tok_start = at as u32;
                let tok_end = (at + surface.len()) as u32;
                search = at + surface.len();
                let tok_ix = token_spans.push(Span::new(tok_start, tok_end));
                let tok_range = IxRange::new(tok_ix.get() as u32, tok_ix.get() as u32 + 1);

                let slice_words = &self.words[mwt.words.start.get()..mwt.words.end.get()];
                if let Some(subspans) = match_contiguous_subspans(surface, tok_start, slice_words) {
                    for sub in subspans {
                        words.push_word(tok_range, None, Some(sub));
                    }
                } else {
                    for sw in slice_words {
                        words.push_word(tok_range, Some(sw.form.as_str()), None);
                    }
                }
                w_idx = mwt.words.end.get();
            } else {
                let sw = &self.words[w_idx];
                let surface = sw.form.as_str();
                let Some(at) = surface_text[search..].find(surface).map(|i| search + i) else {
                    return Err(FormatError::TokenNotInText {
                        line: sw.line,
                        token: surface.to_owned(),
                        sentence: self.sentence_id.as_ref().map(|f| f.as_str().to_owned()),
                        text: surface_text,
                    });
                };
                let tok_start = at as u32;
                let tok_end = (at + surface.len()) as u32;
                search = at + surface.len();
                let tok_ix = token_spans.push(Span::new(tok_start, tok_end));
                words.push_token(tok_ix);
                w_idx += 1;
            }
        }

        let mut sent_spans = Spans::<Sentences>::new(ViewId::SOURCE);
        sent_spans.push(Span::new(0, surface_text.len() as u32));
        let tok_count = token_spans.len();
        let tokens = Tokens { spans: token_spans, sentence: Some(vec![Ix::new(0); tok_count]) };

        let mut upos_labels: Vec<&str> = Vec::new();
        let mut dep_labels: Vec<&str> = Vec::new();
        for w in &self.words {
            let u = w.upos.as_str();
            if !upos_labels.contains(&u) {
                upos_labels.push(u);
            }
            let d = w.deprel.as_str();
            if !dep_labels.contains(&d) {
                dep_labels.push(d);
            }
        }
        let upos_set = TagSet::new("upos", &upos_labels);
        let dep_set = TagSet::new("deprel", &dep_labels);

        let mut pos_tags = Vec::with_capacity(self.words.len());
        let mut lemmas = Strs::default();
        let mut feats = Strs::default();
        let mut heads = Vec::with_capacity(self.words.len());
        let mut rels = Vec::with_capacity(self.words.len());

        for (i, w) in self.words.iter().enumerate() {
            pos_tags.push(upos_set.tag(w.upos.as_str()).unwrap_or(Tag(0)));
            lemmas.push(w.lemma.as_str());
            feats.push(w.feats.as_str());
            // A head is `0` or `_` for a root, else the 1-based id of
            // another word of the sentence; anything else is refused, not
            // read as a root.
            let head_ix = match w.head.as_str() {
                "_" | "0" => None,
                h => match h.parse::<usize>() {
                    Ok(id) if (1..=self.words.len()).contains(&id) && id != i + 1 => Some(Ix::new(id as u32 - 1)),
                    _ => return Err(FormatError::ConlluHead { line: w.line, head: h.to_owned() }),
                },
            };
            heads.push(head_ix);
            rels.push(dep_set.tag(w.deprel.as_str()).unwrap_or(Tag(0)));
        }

        let line = self.words.first().map_or(1, |w| w.line);
        let refused = |e: turbo_analysis::Error| FormatError::ConlluDocument { line, detail: e.to_string() };
        words.check_against(&surface_text, &tokens).map_err(|detail| FormatError::ConlluDocument { line, detail })?;
        let mut doc = Document::new(surface_text);
        doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("conllu")).map_err(refused)?;
        doc.add(&Key::DEFAULT, tokens, Provenance::by("conllu")).map_err(refused)?;
        doc.add(&Key::DEFAULT, words, Provenance::by("conllu")).map_err(refused)?;
        doc.add(&Key::DEFAULT, WordPosTags { tagset: upos_set, tags: pos_tags, prob: None }, Provenance::by("conllu"))
            .map_err(refused)?;
        doc.add(&Key::DEFAULT, WordLemmas { lemma: lemmas }, Provenance::by("conllu")).map_err(refused)?;
        doc.add(&Key::DEFAULT, MorphFeatures::<Words>::new(feats), Provenance::by("conllu")).map_err(refused)?;
        doc.add(
            &Key::DEFAULT,
            WordDependencies { head: heads, relation: rels, relations: dep_set },
            Provenance::by("conllu"),
        )
        .map_err(refused)?;

        Ok(doc)
    }
}

fn match_contiguous_subspans(surface: &str, tok_start: u32, words: &[WordLine<'_>]) -> Option<Vec<Span>> {
    let mut offset = 0usize;
    let mut out = Vec::with_capacity(words.len());
    for w in words {
        let f = w.form.as_str();
        if f.is_empty() || !surface[offset..].starts_with(f) {
            return None;
        }
        let s = tok_start + offset as u32;
        let e = s + f.len() as u32;
        out.push(Span::new(s, e));
        offset += f.len();
    }
    if offset == surface.len() { Some(out) } else { None }
}

/// Serializes a [`Document`] with [`Tokens`] and [`Words`] (and optional
/// [`Sentences`], [`WordPosTags`], [`WordLemmas`], [`MorphFeatures`],
/// [`WordDependencies`]) to CoNLL-U format, emitting multi-word token range
/// lines (`start-end`) whenever multiple syntactic words share a surface
/// token, `SpaceAfter=No` when the next token abuts, and `SpacesAfter=...`
/// when the gap to it is anything but one space (escaped as `\s`, `\t`,
/// `\n`, `\r`, `\p` and `\\`, as the treebanks write it). The `# text`
/// comment is the sentence's text with line breaks and tabs as spaces,
/// since a comment is one line. A sentence with no tokens is not written,
/// and a token no word covers is in the text but has no line. A head in
/// another sentence is refused.
pub fn write_document_conllu(doc: &Document) -> Result<String, FormatError> {
    let tokens = doc.get::<Tokens>().ok_or(FormatError::NoTokens)?;
    let words = doc.get::<Words>();
    let default_words;
    let words = match words {
        Some(w) => w,
        None => {
            default_words = Words::from_tokens(tokens);
            &default_words
        }
    };
    let view_text = doc.text(tokens.spans.view());
    let pos = doc.get::<WordPosTags>();
    let lemmas = doc.get::<WordLemmas>();
    let feats = doc.get::<MorphFeatures<Words>>();
    let deps = doc.get::<WordDependencies>();

    let sentence_ranges: Vec<(Span, IxRange<Tokens>)> = match doc.get::<Sentences>() {
        Some(sents) if tokens.sentence.is_some() => (0..sents.spans.len() as u32)
            .map(|i| {
                let s_ix = Ix::new(i);
                (sents.spans.get(s_ix), tokens.of(s_ix))
            })
            .collect(),
        _ => vec![(Span::new(0, view_text.len() as u32), IxRange::new(0, tokens.spans.len() as u32))],
    };

    let mut out = String::new();
    for (s_idx, (sent_span, tok_range)) in sentence_ranges.into_iter().enumerate() {
        if tok_range.is_empty() {
            continue;
        }
        let _ = writeln!(out, "# sent_id = {}", s_idx + 1);
        out.push_str("# text = ");
        out.extend(view_text[sent_span.range()].chars().map(|c| if matches!(c, '\n' | '\r' | '\t') { ' ' } else { c }));
        out.push('\n');

        let w_start = words.tokens.partition_point(|r| r.end <= tok_range.start);
        let w_end = w_start + words.tokens[w_start..].partition_point(|r| r.start < tok_range.end);

        let mut w_idx = w_start;
        while w_idx < w_end {
            let t_range = words.tokens[w_idx];
            let group_end = w_idx + words.tokens[w_idx..w_end].iter().take_while(|&&r| r == t_range).count();
            let first_tok = tokens.spans.get(t_range.start);
            let last_tok = tokens.spans.get(Ix::new(t_range.end.get() as u32 - 1));
            let tok_span = Span::new(first_tok.start, last_tok.end);
            let surface = &view_text[tok_span.range()];

            let next_tok_ix = t_range.end;
            let gap = if next_tok_ix < tok_range.end {
                &view_text[tok_span.end as usize..tokens.spans.get(next_tok_ix).start as usize]
            } else {
                " "
            };
            let surface_misc = match gap {
                " " => "_".to_owned(),
                "" => "SpaceAfter=No".to_owned(),
                _ => format!("SpacesAfter={}", escape_conllu_spaces(gap)),
            };

            if group_end - w_idx > 1 {
                let start_1 = (w_idx - w_start) + 1;
                let end_1 = group_end - w_start;
                let _ = writeln!(out, "{start_1}-{end_1}\t{surface}\t_\t_\t_\t_\t_\t_\t_\t{surface_misc}");
            }

            for cur in w_idx..group_end {
                let id_1 = (cur - w_start) + 1;
                let w_ix = Ix::<Words>::new(cur as u32);
                let form = words.form(w_ix, view_text, tokens);
                let lemma = lemmas.map_or("_", |l| l.lemma.get(cur));
                let upos = pos.map_or("_", |p| p.tagset.label(p.tags[cur]));
                let feat = feats.map_or("_", |f| f.feats.get(cur));
                let (head_s, deprel) = match deps {
                    Some(d) => {
                        let h = match d.head[cur] {
                            None => 0usize,
                            Some(h_ix) if (w_start..w_end).contains(&h_ix.get()) => (h_ix.get() - w_start) + 1,
                            Some(h_ix) => {
                                return Err(FormatError::HeadOutsideSentence { word: cur, head: h_ix.get() });
                            }
                        };
                        (h.to_string(), d.relations.label(d.relation[cur]))
                    }
                    None => ("_".to_owned(), "_"),
                };
                let word_misc = if group_end - w_idx == 1 { surface_misc.as_str() } else { "_" };
                let _ = writeln!(out, "{id_1}\t{form}\t{lemma}\t{upos}\t_\t{feat}\t{head_s}\t{deprel}\t_\t{word_misc}");
            }

            w_idx = group_end;
        }
        out.push('\n');
    }

    Ok(out)
}
