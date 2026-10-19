use super::*;

impl LayerCodec for Opinions {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_keyed_ranges(&mut w, cx, &self.tokens_key, &self.tokens)?;
        put_f64s(&mut w, &self.polarity)?;
        put_f64s(&mut w, &self.intensity)?;
        put_bools(&mut w, &self.negated)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let (tokens_key, tokens) = get_keyed_ranges(&mut r, cx)?;
        let polarity = get_f64s(&mut r)?;
        let intensity = get_f64s(&mut r)?;
        let negated = get_bools(&mut r)?;
        want_len(polarity.len(), tokens.len(), "polarities")?;
        want_len(intensity.len(), tokens.len(), "intensities")?;
        want_len(negated.len(), tokens.len(), "negation flags")?;
        done(&r, "opinions")?;
        Ok(Opinions { tokens, polarity, intensity, negated, tokens_key })
    }
}

impl LayerCodec for Aspects {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_ref(&mut w, &cx.resolve(&self.tokens_key))?;
        put_ref(&mut w, &cx.resolve(&self.sentences_key))?;
        put_ref(&mut w, &cx.resolve(&self.opinions_key))?;
        w.usize(self.tokens.len())?;
        for t in &self.tokens {
            match t {
                None => w.u8(0),
                Some(range) => {
                    w.u8(1);
                    w.u32(range.start.get() as u32);
                    w.u32(range.end.get() as u32);
                }
            }
        }
        w.usize(self.sentence.len())?;
        for s in &self.sentence {
            match s {
                None => w.u8(0),
                Some(s) => {
                    w.u8(1);
                    w.u32(s.get() as u32);
                }
            }
        }
        put_tagset(&mut w, &self.categories)?;
        w.usize(self.category.len())?;
        for c in &self.category {
            match c {
                None => w.u8(0),
                Some(t) => {
                    w.u8(1);
                    w.u16(t.0);
                }
            }
        }
        put_f64s(&mut w, &self.positive)?;
        put_f64s(&mut w, &self.negative)?;
        put_f64s(&mut w, &self.neutral)?;
        w.usize(self.opinions.len())?;
        for o in &self.opinions {
            w.u32(o.get() as u32);
        }
        put_u32s(&mut w, &self.opinion_end)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let tokens_ref = get_ref(&mut r)?;
        want_target::<Tokens>(&tokens_ref, cx)?;
        let sentences_ref = get_ref(&mut r)?;
        want_target::<Sentences>(&sentences_ref, cx)?;
        let opinions_ref = get_ref(&mut r)?;
        want_target::<Opinions>(&opinions_ref, cx)?;
        let tokens_key = cx.key::<Tokens>(&tokens_ref.instance.clone());
        let sentences_key = cx.key::<Sentences>(&sentences_ref.instance.clone());
        let opinions_key = cx.key::<Opinions>(&opinions_ref.instance.clone());
        let mut tokens = Vec::new();
        for _ in 0..r.usize()? {
            tokens.push(match r.u8()? {
                0 => None,
                1 => {
                    let range = get_checked_range(&mut r, cx, &tokens_ref)?;
                    Some(range)
                }
                tag => return Err(malformed(format!("an optional range tag of {tag}"))),
            });
        }
        let n = tokens.len();
        let mut sentence = Vec::new();
        for _ in 0..r.usize()? {
            sentence.push(match r.u8()? {
                0 => None,
                1 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: sentences_ref.clone(), index })?;
                    Some(Ix::new(index))
                }
                tag => return Err(malformed(format!("an optional index tag of {tag}"))),
            });
        }
        let categories = get_tagset(&mut r)?;
        let mut category = Vec::new();
        for _ in 0..r.usize()? {
            category.push(match r.u8()? {
                0 => None,
                1 => {
                    let tag = Tag(r.u16()?);
                    if tag.0 as usize >= categories.len() {
                        return Err(malformed(format!("a category outside {}", categories.name)));
                    }
                    Some(tag)
                }
                tag => return Err(malformed(format!("an optional category tag of {tag}"))),
            });
        }
        let positive = get_f64s(&mut r)?;
        let negative = get_f64s(&mut r)?;
        let neutral = get_f64s(&mut r)?;
        let mut opinions = Vec::new();
        for _ in 0..r.usize()? {
            let index = r.u32()?;
            cx.require_node(&NodeRef { layer: opinions_ref.clone(), index })?;
            opinions.push(Ix::new(index));
        }
        let opinion_end = get_u32s(&mut r)?;
        want_len(sentence.len(), n, "sentences")?;
        want_len(category.len(), n, "categories")?;
        want_len(positive.len(), n, "positive scores")?;
        want_len(negative.len(), n, "negative scores")?;
        want_len(neutral.len(), n, "neutral scores")?;
        want_len(opinion_end.len(), n, "opinion ends")?;
        if opinion_end.windows(2).any(|w| w[0] > w[1])
            || opinion_end.last().copied().unwrap_or(0) as usize != opinions.len()
        {
            return Err(malformed("opinion ends do not partition the opinions"));
        }
        done(&r, "aspects")?;
        Ok(Aspects {
            tokens,
            sentence,
            category,
            categories,
            positive,
            negative,
            neutral,
            opinions,
            opinion_end,
            tokens_key,
            sentences_key,
            opinions_key,
        })
    }
}

impl LayerCodec for Incivility {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_ref(&mut w, &cx.resolve(&self.tokens_key))?;
        put_ref(&mut w, &cx.resolve(&self.sentences_key))?;
        put_ref(&mut w, &cx.resolve(&self.chunks_key))?;
        put_ref(&mut w, &cx.resolve(&self.structure_key))?;
        w.usize(self.target.len())?;
        for t in &self.target {
            match t {
                IncivilityTarget::Sentence(i) => {
                    w.u8(0);
                    w.u32(i.get() as u32);
                }
                IncivilityTarget::Chunk(i) => {
                    w.u8(1);
                    w.u32(i.get() as u32);
                }
                IncivilityTarget::Block(i) => {
                    w.u8(2);
                    w.u32(i.get() as u32);
                }
            }
        }
        for scores in [
            &self.insult,
            &self.profanity,
            &self.threat,
            &self.identity_attack,
            &self.intolerance,
            &self.severe_toxicity,
        ] {
            put_f64s(&mut w, scores)?;
        }
        w.usize(self.evidence_label.len())?;
        for l in &self.evidence_label {
            w.u8(*l as u8);
        }
        w.usize(self.evidence_tokens.len())?;
        for t in &self.evidence_tokens {
            w.u32(t.start.get() as u32);
            w.u32(t.end.get() as u32);
        }
        put_u32s(&mut w, &self.evidence_end)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let tokens_ref = get_ref(&mut r)?;
        want_target::<Tokens>(&tokens_ref, cx)?;
        let sentences_ref = get_ref(&mut r)?;
        want_target::<Sentences>(&sentences_ref, cx)?;
        let chunks_ref = get_ref(&mut r)?;
        want_target::<Chunks>(&chunks_ref, cx)?;
        let structure_ref = get_ref(&mut r)?;
        want_target::<Structure>(&structure_ref, cx)?;
        let tokens_key = cx.key::<Tokens>(&tokens_ref.instance.clone());
        let sentences_key = cx.key::<Sentences>(&sentences_ref.instance.clone());
        let chunks_key = cx.key::<Chunks>(&chunks_ref.instance.clone());
        let structure_key = cx.key::<Structure>(&structure_ref.instance.clone());
        let mut target = Vec::new();
        for _ in 0..r.usize()? {
            target.push(match r.u8()? {
                0 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: sentences_ref.clone(), index })?;
                    IncivilityTarget::Sentence(Ix::new(index))
                }
                1 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: chunks_ref.clone(), index })?;
                    IncivilityTarget::Chunk(Ix::new(index))
                }
                2 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: structure_ref.clone(), index })?;
                    IncivilityTarget::Block(Ix::new(index))
                }
                tag => return Err(malformed(format!("an incivility target tag of {tag}"))),
            });
        }
        let n = target.len();
        let [insult, profanity, threat, identity_attack, intolerance, severe_toxicity] = [
            get_f64s(&mut r)?,
            get_f64s(&mut r)?,
            get_f64s(&mut r)?,
            get_f64s(&mut r)?,
            get_f64s(&mut r)?,
            get_f64s(&mut r)?,
        ];
        let mut evidence_label = Vec::new();
        for _ in 0..r.usize()? {
            evidence_label.push(
                *IncivilityLabel::ALL
                    .get(r.u8()? as usize)
                    .ok_or_else(|| malformed("an incivility label outside the table"))?,
            );
        }
        let mut evidence_tokens = Vec::new();
        for _ in 0..r.usize()? {
            let range = get_checked_range(&mut r, cx, &tokens_ref)?;
            evidence_tokens.push(range);
        }
        let evidence_end = get_u32s(&mut r)?;
        for (scores, what) in [
            (&insult, "insult scores"),
            (&profanity, "profanity scores"),
            (&threat, "threat scores"),
            (&identity_attack, "identity attack scores"),
            (&intolerance, "intolerance scores"),
            (&severe_toxicity, "severe toxicity scores"),
        ] {
            want_len(scores.len(), n, what)?;
        }
        want_len(evidence_tokens.len(), evidence_label.len(), "evidence spans")?;
        want_len(evidence_end.len(), n, "evidence ends")?;
        if evidence_end.windows(2).any(|w| w[0] > w[1])
            || evidence_end.last().copied().unwrap_or(0) as usize != evidence_label.len()
        {
            return Err(malformed("evidence ends do not partition the evidence"));
        }
        done(&r, "incivility")?;
        Ok(Incivility {
            target,
            insult,
            profanity,
            threat,
            identity_attack,
            intolerance,
            severe_toxicity,
            evidence_label,
            evidence_tokens,
            evidence_end,
            tokens_key,
            sentences_key,
            chunks_key,
            structure_key,
        })
    }
}

impl LayerCodec for Pragmatics {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_ref(&mut w, &cx.resolve(&self.tokens_key))?;
        put_f64s(&mut w, &self.score)?;
        w.usize(self.evidence.len())?;
        for e in &self.evidence {
            match e {
                None => w.u8(0),
                Some(inc) => {
                    w.u8(1);
                    for range in [inc.positive, inc.negative] {
                        w.u32(range.start.get() as u32);
                        w.u32(range.end.get() as u32);
                    }
                }
            }
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let tokens_ref = get_ref(&mut r)?;
        want_target::<Tokens>(&tokens_ref, cx)?;
        let tokens_key = cx.key::<Tokens>(&tokens_ref.instance.clone());
        let score = get_f64s(&mut r)?;
        let mut evidence = Vec::new();
        for _ in 0..r.usize()? {
            evidence.push(match r.u8()? {
                0 => None,
                1 => {
                    let positive = get_checked_range(&mut r, cx, &tokens_ref)?;
                    let negative = get_checked_range(&mut r, cx, &tokens_ref)?;
                    Some(Incongruity { positive, negative })
                }
                tag => return Err(malformed(format!("an optional evidence tag of {tag}"))),
            });
        }
        want_len(evidence.len(), score.len(), "evidence slots")?;
        done(&r, "pragmatics")?;
        Ok(Pragmatics { score, evidence, tokens_key })
    }
}
