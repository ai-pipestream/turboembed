use super::*;

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
        put_f64s(&mut w, &self.scores)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let categories = get_tagset(&mut r)?;
        let category = get_category_tags(&mut r, &categories)?;
        let scores = get_f64s(&mut r)?;
        let want =
            category.len().checked_mul(categories.len()).ok_or_else(|| malformed("a score count that overflows"))?;
        want_len(scores.len(), want, "scores")?;
        done(&r, "categories")?;
        let mut out = Categories::with_capacity(categories, category.len());
        let k = out.categories.len();
        for (i, best) in category.into_iter().enumerate() {
            out.push(best, &scores[i * k..(i + 1) * k]);
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
