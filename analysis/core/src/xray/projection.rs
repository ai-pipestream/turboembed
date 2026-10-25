use super::{NormalizationXray, XrayRunKind, invalid};
use crate::{CodePointSpan, Error, OffsetCursor, OffsetTable, Span, Utf16Span};

/// Typed coordinates for the same run order as the byte x-ray. Columns
/// own only typed offsets and run kinds, never copies of run text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XrayProjection<S> {
    parent: Vec<S>,
    child: Vec<S>,
    kind: Vec<XrayRunKind>,
}

impl<S> XrayProjection<S> {
    pub fn parent(&self) -> &[S] {
        &self.parent
    }
    pub fn child(&self) -> &[S] {
        &self.child
    }
    pub fn kinds(&self) -> &[XrayRunKind] {
        &self.kind
    }
    pub fn len(&self) -> usize {
        self.kind.len()
    }
    pub fn is_empty(&self) -> bool {
        self.kind.is_empty()
    }
}

impl NormalizationXray {
    /// Convert all boundaries using one forward OffsetCursor per side.
    /// Tables must belong to the supplied text. Byte-domain lengths and
    /// character boundaries are checked before starting either cursor.
    pub fn project_utf16(
        &self,
        from_text: &str,
        from_table: &OffsetTable,
        to_text: &str,
        to_table: &OffsetTable,
    ) -> Result<XrayProjection<Utf16Span>, Error> {
        self.project(from_text, from_table, to_text, to_table, |c, s| {
            Utf16Span::new(c.to_utf16(s.start), c.to_utf16(s.end))
        })
    }

    pub fn project_code_points(
        &self,
        from_text: &str,
        from_table: &OffsetTable,
        to_text: &str,
        to_table: &OffsetTable,
    ) -> Result<XrayProjection<CodePointSpan>, Error> {
        self.project(from_text, from_table, to_text, to_table, |c, s| {
            CodePointSpan::new(c.to_code_point(s.start), c.to_code_point(s.end))
        })
    }

    fn project<S>(
        &self,
        from: &str,
        ft: &OffsetTable,
        to: &str,
        tt: &OffsetTable,
        mut convert: impl FnMut(&mut OffsetCursor<'_>, Span) -> S,
    ) -> Result<XrayProjection<S>, Error> {
        if from.len() != self.parent_bytes as usize
            || to.len() != self.child_bytes as usize
            || ft.len() as usize != from.len()
            || tt.len() as usize != to.len()
        {
            return Err(invalid("xray projection text/table lengths do not match the byte domains"));
        }
        for r in self.iter() {
            if from.get(r.parent.range()).is_none() || to.get(r.child.range()).is_none() {
                return Err(invalid("xray projection boundary cuts a UTF-8 character"));
            }
        }
        let (mut pc, mut cc) = (OffsetCursor::new(ft, from), OffsetCursor::new(tt, to));
        let mut p = Vec::with_capacity(self.len());
        let mut c = Vec::with_capacity(self.len());
        for r in self.iter() {
            p.push(convert(&mut pc, r.parent));
            c.push(convert(&mut cc, r.child));
        }
        Ok(XrayProjection { parent: p, child: c, kind: self.kind.clone() })
    }
}
