use super::*;

use std::sync::Arc;

impl LayerCodec for Language {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_strs(&mut w, &self.code)?;
        put_f64s(&mut w, &self.score)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let code = get_strs(&mut r)?;
        let score = get_f64s(&mut r)?;
        want_len(score.len(), code.len(), "scores")?;
        done(&r, "language")?;
        Ok(Language { code, score })
    }
}

impl LayerCodec for Words {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_ranges(&mut w, cx, &self.tokens_key, &self.tokens)?;
        match &self.form {
            None => w.u8(0),
            Some(forms) => {
                w.u8(1);
                w.usize(forms.len())?;
                for f in forms {
                    match f {
                        None => w.u8(0),
                        Some(f) => {
                            w.u8(1);
                            w.str(f)?;
                        }
                    }
                }
            }
        }
        match &self.subspan {
            None => w.u8(0),
            Some(spans) => {
                w.u8(1);
                w.usize(spans.len())?;
                for s in spans {
                    match s {
                        None => w.u8(0),
                        Some(s) => {
                            w.u8(1);
                            w.span(*s);
                        }
                    }
                }
            }
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        // The token ranges carry their target ref; read it through a
        // reader that also yields the key to restore.
        let target = get_ref(&mut r)?;
        want_target::<Tokens>(&target, cx)?;
        let len = cx.layer_len(&target)?;
        let mut tokens = Vec::new();
        for _ in 0..r.usize()? {
            let (start, end) = (r.u32()?, r.u32()?);
            tokens.push(checked_range(cx, &target, len, start, end)?);
        }
        let tokens_key = cx.key::<Tokens>(&target.instance.clone());
        let n = tokens.len();
        let form = match r.u8()? {
            0 => None,
            1 => {
                let mut forms = Vec::new();
                for _ in 0..r.usize()? {
                    forms.push(match r.u8()? {
                        0 => None,
                        1 => Some(r.string()?.into_boxed_str()),
                        tag => return Err(malformed(format!("an optional form tag of {tag}"))),
                    });
                }
                want_len(forms.len(), n, "word forms")?;
                Some(forms)
            }
            tag => return Err(malformed(format!("an optional column tag of {tag}"))),
        };
        let subspan = match r.u8()? {
            0 => None,
            1 => {
                let mut spans = Vec::new();
                for _ in 0..r.usize()? {
                    spans.push(match r.u8()? {
                        0 => None,
                        1 => Some(r.span()?),
                        tag => return Err(malformed(format!("an optional span tag of {tag}"))),
                    });
                }
                want_len(spans.len(), n, "subspans")?;
                Some(spans)
            }
            tag => return Err(malformed(format!("an optional column tag of {tag}"))),
        };
        done(&r, "words")?;
        let words = Words { tokens, form, subspan, tokens_key };
        // Subspans and explicit forms are only meaningful against the
        // restored tokens and their view text; `Document::add` cannot
        // run that check (the layer is document-anchored), so refuse a
        // subspan that splits a character or leaves its token here,
        // before ordinary access could panic on it.
        let tokens = doc.get_key(&tokens_key).ok_or_else(|| ExportError::DanglingReference {
            layer: cx.layer().to_owned(),
            reference: target.clone(),
        })?;
        words
            .check_against(doc.text(tokens.spans.view()), tokens)
            .map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
        Ok(words)
    }
}

impl<A: WordOrTokenAnchor> LayerCodec for MorphFeatures<A> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_strs(&mut w, &self.feats)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let feats = get_strs(&mut r)?;
        done(&r, "morph features")?;
        Ok(MorphFeatures::new(feats))
    }
}

impl<L: CategoryScope> LayerCodec for Categories<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_tagset(&mut w, &self.categories)?;
        put_tags(&mut w, &self.category)?;
        put_selected(&mut w, &self.selected, &self.selected_end)?;
        put_f64s(&mut w, &self.scores)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let categories = get_tagset(&mut r)?;
        let category = get_category_tags(&mut r, &categories)?;
        let selected = get_selected(&mut r, &categories)?;
        let scores = get_f64s(&mut r)?;
        let want =
            category.len().checked_mul(categories.len()).ok_or_else(|| malformed("a score count that overflows"))?;
        want_len(scores.len(), want, "scores")?;
        want_len(selected.len(), category.len(), "selected label sets")?;
        done(&r, "categories")?;
        let mut out = Categories::with_capacity(categories, category.len());
        let k = out.categories.len();
        for (i, best) in category.into_iter().enumerate() {
            out.push_selected(best, &selected[i], &scores[i * k..(i + 1) * k]);
        }
        Ok(out)
    }
}

/// The best-category column of a [`Categories`] layer: like [`get_tags`],
/// but `Tag::NONE` is the item that abstained (nothing predicted), not an
/// index into the tag set.
fn get_category_tags(r: &mut Reader<'_>, tagset: &TagSet) -> Result<Vec<Tag>, ExportError> {
    let mut tags = Vec::new();
    for _ in 0..r.usize()? {
        let tag = Tag(r.u16()?);
        if tag != Tag::NONE && tag.0 as usize >= tagset.len() {
            return Err(malformed(format!("a tag outside {}", tagset.name)));
        }
        tags.push(tag);
    }
    Ok(tags)
}

/// The selected-set column: one label list per item, in category order,
/// sliced out of the layer's CSR pair.
fn put_selected(w: &mut Writer, selected: &[Tag], ends: &[u32]) -> Result<(), ExportError> {
    w.usize(ends.len())?;
    let mut start = 0usize;
    for &end in ends {
        let set = &selected[start..end as usize];
        w.usize(set.len())?;
        for t in set {
            w.u16(t.0);
        }
        start = end as usize;
    }
    Ok(())
}

/// The selected-set column read back: every tag an index into the set,
/// so no [`Tag::NONE`] (abstention is the empty list).
fn get_selected(r: &mut Reader<'_>, tagset: &TagSet) -> Result<Vec<Vec<Tag>>, ExportError> {
    let mut out = Vec::new();
    for _ in 0..r.usize()? {
        let mut set = Vec::new();
        for _ in 0..r.usize()? {
            let tag = Tag(r.u16()?);
            if tag.0 as usize >= tagset.len() {
                return Err(malformed(format!("a selected tag outside {}", tagset.name)));
            }
            set.push(tag);
        }
        out.push(set);
    }
    Ok(out)
}

fn put_u8s(w: &mut Writer, values: &[u8]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for &v in values {
        w.u8(v);
    }
    Ok(())
}

fn get_u8s(r: &mut Reader<'_>) -> Result<Vec<u8>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.u8()?);
    }
    Ok(values)
}

impl LayerCodec for WordNetSenses {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ix_col(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.token)?;
        put_strs(&mut w, &self.synset_id)?;
        put_strs(&mut w, &self.sense_id)?;
        put_u8s(&mut w, &self.pos)?;
        put_strs(&mut w, &self.lemma)?;
        put_strs(&mut w, &self.synset_lemmas)?;
        put_u32s(&mut w, &self.synset_lemma_row)?;
        put_strs(&mut w, &self.gloss)?;
        put_u32s(&mut w, &self.sense_rank)?;
        w.str(&self.lexicon_id)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = spans.len();
        let token = get_ix_col(&mut r, cx)?;
        want_len(token.len(), n, "token indexes")?;
        let synset_id = get_strs(&mut r)?;
        want_len(synset_id.len(), n, "synset ids")?;
        let sense_id = get_strs(&mut r)?;
        want_len(sense_id.len(), n, "sense ids")?;
        let pos = get_u8s(&mut r)?;
        want_len(pos.len(), n, "parts of speech")?;
        let lemma = get_strs(&mut r)?;
        want_len(lemma.len(), n, "lemmas")?;
        let synset_lemmas = get_strs(&mut r)?;
        let synset_lemma_row = get_u32s(&mut r)?;
        want_len(synset_lemma_row.len(), n + 1, "synset lemma row starts")?;
        let gloss = get_strs(&mut r)?;
        want_len(gloss.len(), n, "glosses")?;
        let sense_rank = get_u32s(&mut r)?;
        want_len(sense_rank.len(), n, "sense ranks")?;
        let lexicon_id: Arc<str> = r.string()?.into();
        done(&r, "wordnet senses")?;
        Ok(WordNetSenses {
            spans,
            token,
            synset_id,
            sense_id,
            pos,
            lemma,
            synset_lemmas,
            synset_lemma_row,
            gloss,
            sense_rank,
            lexicon_id,
        })
    }
}

impl LayerCodec for LexicalExpansions {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ix_col(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.token)?;
        put_strs(&mut w, &self.term)?;
        put_u8s(&mut w, &self.relation)?;
        put_u8s(&mut w, &self.depth)?;
        put_u32s(&mut w, &self.sense_rank)?;
        put_f64s(&mut w, &self.weight)?;
        w.str(&self.lexicon_id)?;
        put_strs(&mut w, &self.synset_id)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = spans.len();
        let token = get_ix_col(&mut r, cx)?;
        want_len(token.len(), n, "token indexes")?;
        let term = get_strs(&mut r)?;
        want_len(term.len(), n, "terms")?;
        let relation = get_u8s(&mut r)?;
        want_len(relation.len(), n, "relations")?;
        let depth = get_u8s(&mut r)?;
        want_len(depth.len(), n, "depths")?;
        let sense_rank = get_u32s(&mut r)?;
        want_len(sense_rank.len(), n, "sense ranks")?;
        let weight = get_f64s(&mut r)?;
        want_len(weight.len(), n, "weights")?;
        let lexicon_id: Arc<str> = r.string()?.into();
        let synset_id = get_strs(&mut r)?;
        want_len(synset_id.len(), n, "synset ids")?;
        done(&r, "lexical expansions")?;
        Ok(LexicalExpansions { spans, token, term, relation, depth, sense_rank, weight, lexicon_id, synset_id })
    }
}
