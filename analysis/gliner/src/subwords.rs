use std::sync::Arc;

use turbo::sentencepiece::{PieceType, Scratch};
use turbo_analysis::{Annotator, Error, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_embed::align;
use turbo_analysis_layers::{Subwords, Tokens};

use crate::Model;

pub struct GlinerSubwords {
    model: Arc<Model>,
}

impl GlinerSubwords {
    pub fn new(model: Arc<Model>) -> Self {
        Self { model }
    }

    pub fn pieces(&self, view: &View) -> Result<Subwords, Error> {
        self.pieces_with_scratch(view, &mut Scratch::default(), &mut Vec::new())
    }

    fn pieces_with_scratch(
        &self,
        view: &View,
        scratch: &mut Scratch,
        encoded: &mut Vec<turbo::sentencepiece::Piece>,
    ) -> Result<Subwords, Error> {
        encoded.clear();
        self.model.tokenizer.encode(view.text(), scratch, encoded);
        let mut fused: Vec<(u32, u32, u32)> = Vec::new();
        let mut previous_byte = false;
        for piece in encoded.iter() {
            let byte = self.model.tokenizer.kind(piece.id) == Some(PieceType::Byte);
            if byte && previous_byte {
                fused.last_mut().expect("a byte run").2 = piece.end;
                continue;
            }
            let id = if byte { self.model.tokenizer.unk_id() } else { piece.id };
            fused.push((id, piece.begin, piece.end));
            previous_byte = byte;
        }
        if fused.is_empty() && !view.text().is_empty() {
            fused.push((self.model.tokenizer.unk_id(), 0, view.text().len() as u32));
        }
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut ids = Vec::with_capacity(fused.len());
        for (id, mut start, mut end) in fused {
            start = start.min(text.len() as u32);
            end = end.min(text.len() as u32);
            while !text.is_char_boundary(start as usize) {
                start -= 1;
            }
            while !text.is_char_boundary(end as usize) {
                end += 1;
            }
            ids.push(id as i32);
            spans.push(Span::new(start, end));
        }
        Ok(Subwords { spans, ids, model: self.model.identity.clone(), token: None })
    }
}

impl Annotator for GlinerSubwords {
    const ID: &'static str = "turbo:subwords";
    type Reads = (View, Option<Tokens>);
    type Writes = (Subwords,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let (view, tokens) = input;
        let mut subwords = self.pieces(view)?;
        if let Some(tokens) = tokens {
            if tokens.spans.view() != view.id() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: "the tokens and subwords are in different views".into(),
                });
            }
            if !tokens.spans.is_ordered() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: "the tokens are not in text order".into(),
                });
            }
            subwords.token = align(&subwords.spans, &tokens.spans);
        }
        Ok((subwords,))
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let mut scratch = Scratch::default();
        let mut encoded = Vec::new();
        inputs
            .into_iter()
            .map(|(view, tokens)| {
                let mut subwords = self.pieces_with_scratch(view, &mut scratch, &mut encoded)?;
                if let Some(tokens) = tokens {
                    if tokens.spans.view() != view.id() || !tokens.spans.is_ordered() {
                        return Err(Error::Annotator {
                            annotator: Self::ID,
                            detail: "the tokens are not ordered in the subword view".into(),
                        });
                    }
                    subwords.token = align(&subwords.spans, &tokens.spans);
                }
                Ok((subwords,))
            })
            .collect()
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.identity.clone())
    }
}
