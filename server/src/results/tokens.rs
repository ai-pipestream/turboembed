//! TokenVectorsResult: the turbo-analysis-layers `Vectors<Subwords>` and
//! `TokenVectorWindows` layers (issue #709), one contextual vector per
//! model token with the spans of the encoder windows that produced them.

use turbo_analysis::{Layer, ModelRef, Span, ViewId};
use turbo_analysis_layers::{Subwords, TokenVectorWindows, Vectors};

use super::{MAX_TOKEN_VECTOR_BYTES, Wire, mismatched, model_from_proto, model_to_proto, over};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::TokenVectorsResult;

/// The token-vector result unbundled from its two layers: the vectors and
/// the encoder windows beside them. `Vectors<Subwords>` is the layer;
/// `windows` is what `TokenVectorWindows` carries.
#[derive(Debug)]
pub struct TokenVectors {
    pub model: ModelRef,
    pub dim: u32,
    /// rows x dim, row-major.
    pub data: Vec<f32>,
    /// The rows with no vector, ascending; their rows are zeros.
    pub empty: Vec<u32>,
    /// The encoder windows, in text order, non-overlapping.
    pub windows: Vec<Span>,
}

impl TokenVectors {
    /// The two layers the result came from.
    pub fn new(v: &Vectors<Subwords>, windows: &TokenVectorWindows) -> TokenVectors {
        TokenVectors {
            model: v.model.clone(),
            dim: v.dim,
            data: v.data.clone(),
            empty: v.empty.iter().map(|i| i.get() as u32).collect(),
            windows: (0..windows.spans.len()).map(|i| windows.spans.get(turbo_analysis::Ix::new(i as u32))).collect(),
        }
    }

    /// The vectors layer this result's rows make, checked.
    pub fn vectors(&self) -> Result<Vectors<Subwords>, Failure> {
        let rows = self.data.len() / self.dim.max(1) as usize;
        if self.empty.iter().any(|&i| i as usize >= rows) {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "empty: row indices among the rows"));
        }
        let empty: Vec<turbo_analysis::Ix<Subwords>> = self.empty.iter().map(|&i| turbo_analysis::Ix::new(i)).collect();
        let v = Vectors::new(self.dim, self.data.clone(), empty, self.model.clone());
        v.check().map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("token vectors: {e}")))?;
        Ok(v)
    }

    /// The windows layer this result's windows make, checked the way
    /// `TokenVectorWindows` checks itself: at least one window when rows
    /// carry vectors, ordered, non-overlapping, each non-empty.
    pub fn windows_layer(&self, rows: usize) -> Result<TokenVectorWindows, Failure> {
        if rows > 0 && self.windows.is_empty() {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "at least one encoder window for the rows"));
        }
        let mut spans = turbo_analysis::Spans::new(ViewId::SOURCE);
        for &w in &self.windows {
            if w.start >= w.end {
                return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "a window that ends at or before it starts"));
            }
            spans.push(w);
        }
        spans
            .check(turbo_analysis::Overlap::None)
            .map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("windows: {e}")))?;
        Ok(TokenVectorWindows { spans })
    }
}

impl Wire for TokenVectors {
    type Msg = TokenVectorsResult;

    fn to_msg(&self) -> TokenVectorsResult {
        let mut data = Vec::with_capacity(self.data.len() * 4);
        for v in &self.data {
            data.extend_from_slice(&v.to_le_bytes());
        }
        TokenVectorsResult {
            model: Some(model_to_proto(&self.model)),
            dim: self.dim,
            data: data.into(),
            empty: self.empty.clone(),
            window_start: self.windows.iter().map(|s| s.start).collect(),
            window_end: self.windows.iter().map(|s| s.end).collect(),
        }
    }

    fn from_msg(msg: &TokenVectorsResult) -> Result<Self, Failure> {
        if msg.data.len() as u64 > MAX_TOKEN_VECTOR_BYTES {
            return Err(over("max_token_vector_bytes", "token vector data", msg.data.len() as u64));
        }
        if msg.dim == 0 {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "dim: over 0"));
        }
        if !msg.data.len().is_multiple_of(4) {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "data: whole little-endian f32 values"));
        }
        let values = msg.data.len() / 4;
        if !values.is_multiple_of(msg.dim as usize) {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                0,
                format!("data: {} f32 values do not make rows of dim {}", values, msg.dim),
            ));
        }
        if msg.window_start.len() != msg.window_end.len() {
            return Err(mismatched("window_start", msg.window_start.len(), msg.window_end.len()));
        }
        let mut data = Vec::with_capacity(values);
        for c in msg.data.as_chunks::<4>().0 {
            data.push(f32::from_le_bytes(*c));
        }
        let out = TokenVectors {
            model: msg.model.as_ref().map(model_from_proto).unwrap_or(ModelRef {
                id: String::new(),
                revision: String::new(),
                sha256: String::new(),
            }),
            dim: msg.dim,
            data,
            empty: msg.empty.clone(),
            windows: msg.window_start.iter().zip(&msg.window_end).map(|(&s, &e)| Span { start: s, end: e }).collect(),
        };
        let rows = out.data.len() / out.dim as usize;
        out.vectors()?;
        out.windows_layer(rows)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::TURBO_E_CAPACITY;

    fn model() -> ModelRef {
        ModelRef { id: "org/m".into(), revision: "r1".into(), sha256: "abc".into() }
    }

    /// Four rows of dim 3; row 1 has no vector. Windows tile the text.
    fn sample() -> (Vectors<Subwords>, TokenVectorWindows) {
        let data: Vec<f32> = (0..12).map(|i| if i / 3 == 1 { 0.0 } else { i as f32 * 0.25 }).collect();
        let v = Vectors::new(3, data, vec![turbo_analysis::Ix::new(1)], model());
        v.check().unwrap();
        let mut spans = turbo_analysis::Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, 10));
        spans.push(Span::new(10, 20));
        let w = TokenVectorWindows { spans };
        w.check().unwrap();
        (v, w)
    }

    #[test]
    fn the_wire_round_trip_is_unchanged() {
        let (v, w) = sample();
        let t = TokenVectors::new(&v, &w);
        let msg = t.to_msg();
        let back = TokenVectors::from_msg(&msg).unwrap();
        assert_eq!(back.model, t.model);
        assert_eq!(back.dim, 3);
        assert_eq!(back.data, t.data);
        assert_eq!(back.empty, [1]);
        assert_eq!(back.windows, t.windows);
        // The rebuilt layers hold what the originals held.
        let rv = back.vectors().unwrap();
        assert_eq!((rv.dim, rv.data, rv.model), (v.dim, v.data.clone(), v.model.clone()));
        assert_eq!(rv.empty.iter().map(|i| i.get()).collect::<Vec<_>>(), [1]);
        let rw = back.windows_layer(back.data.len() / 3).unwrap();
        assert_eq!(rw.spans.columns(), w.spans.columns());
        assert_eq!(rw.spans.fragment_columns(), w.spans.fragment_columns());
    }

    #[test]
    fn an_empty_result_has_no_windows_and_round_trips() {
        let v = Vectors::new(3, Vec::new(), Vec::new(), model());
        let mut spans = turbo_analysis::Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, 4));
        let t = TokenVectors::new(&v, &TokenVectorWindows { spans });
        let t = TokenVectors { windows: Vec::new(), ..t };
        let back = TokenVectors::from_msg(&t.to_msg()).unwrap();
        assert!(back.data.is_empty() && back.empty.is_empty() && back.windows.is_empty());
        assert!(back.vectors().unwrap().len() == 0);
    }

    fn refused(mut f: impl FnMut(&mut TokenVectorsResult)) -> Failure {
        let (v, w) = sample();
        let mut msg = TokenVectors::new(&v, &w).to_msg();
        f(&mut msg);
        TokenVectors::from_msg(&msg).unwrap_err()
    }

    #[test]
    fn garbage_is_refused_without_panicking() {
        let e = refused(|m| m.dim = 0);
        assert!(e.message.contains("dim"), "{e:?}");
        let e = refused(|m| {
            m.data = vec![1, 2, 3].into();
        });
        assert!(e.message.contains("f32"), "{e:?}");
        let e = refused(|m| {
            m.data = m.data.slice(..m.data.len() - 4);
        });
        assert!(e.message.contains("rows"), "{e:?}");
        let e = refused(|m| {
            m.empty = vec![4];
        });
        assert!(e.message.contains("among the rows"), "{e:?}");
        let e = refused(|m| {
            m.empty = vec![3];
        });
        assert!(e.message.contains("zeros"), "{e:?}");
        let e = refused(|m| m.empty = vec![1, 1]);
        assert!(e.message.contains("order"), "{e:?}");
        let e = refused(|m| m.window_start = vec![10, 0]);
        assert!(e.message.contains("ends at or before"), "{e:?}");
        let e = refused(|m| m.window_start = vec![0, 5]);
        assert!(e.message.contains("windows"), "{e:?}");
        let e = refused(|m| m.window_end[0] = 30);
        assert!(e.message.contains("windows"), "{e:?}");
        let e = refused(|m| {
            m.window_end.pop();
        });
        assert!(e.message.contains("window_start"), "{e:?}");
        // Row 1 is listed empty but carries values.
        let (v, w) = sample();
        let mut t = TokenVectors::new(&v, &w);
        t.data[3] = 9.0;
        let e = TokenVectors::from_msg(&t.to_msg()).unwrap_err();
        assert!(e.message.contains("zeros"), "{e:?}");
        // Nothing at all: no dim.
        let e = TokenVectors::from_msg(&TokenVectorsResult::default()).unwrap_err();
        assert!(e.message.contains("dim"), "{e:?}");
    }

    #[test]
    fn over_the_byte_limit_names_it() {
        let msg = TokenVectorsResult {
            dim: 1,
            data: vec![0u8; MAX_TOKEN_VECTOR_BYTES as usize + 4].into(),
            ..Default::default()
        };
        let e = TokenVectors::from_msg(&msg).unwrap_err();
        assert_eq!(e.code, TURBO_E_CAPACITY);
        assert!(e.message.contains("max_token_vector_bytes"), "{e:?}");
    }
}
