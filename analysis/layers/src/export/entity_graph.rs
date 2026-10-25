//! The wire form of `DocumentEntities`: the view, then each entity's
//! columns and mentions, then the relations. A mention's `Entities` row is
//! written against the default `Entities` instance, named once.
//! `EntitySalience` is its row count, then each score's columns.

use turbo_analysis::ViewId;

use super::*;
use crate::{
    ConsolidatedMention, DocumentEntities, DocumentEntity, EntityRelation, EntitySalience, EntitySalienceScore,
    SyntacticRole, SyntacticRoleCounts,
};

fn put_opt_str(w: &mut Writer, s: Option<&str>) -> Result<(), ExportError> {
    match s {
        None => w.u8(0),
        Some(s) => {
            w.u8(1);
            w.str(s)?;
        }
    }
    Ok(())
}

fn get_opt_str(r: &mut Reader<'_>) -> Result<Option<Box<str>>, ExportError> {
    match r.u8()? {
        0 => Ok(None),
        1 => Ok(Some(r.string()?.into())),
        tag => Err(malformed(format!("an optional string tag of {tag}"))),
    }
}

fn put_f32(w: &mut Writer, v: f32) {
    w.u32(v.to_bits());
}

fn get_f32(r: &mut Reader<'_>) -> Result<f32, ExportError> {
    Ok(f32::from_bits(r.u32()?))
}

fn put_mention(w: &mut Writer, m: &ConsolidatedMention) {
    w.u32(m.span.start);
    w.u32(m.span.end);
    match m.entity {
        None => w.u8(0),
        Some(e) => {
            w.u8(1);
            w.u32(e.get() as u32);
        }
    }
    w.u8(u8::from(m.is_representative));
    w.u8(m.role.map_or(0, SyntacticRole::to_byte));
}

fn get_mention(
    r: &mut Reader<'_>,
    cx: &mut ImportCx,
    rows: &Option<LayerRef>,
) -> Result<ConsolidatedMention, ExportError> {
    let span = r.span()?;
    let entity = match r.u8()? {
        0 => None,
        1 => {
            let target = rows.clone().ok_or_else(|| malformed("an entity row with no entity layer named"))?;
            let index = r.u32()?;
            cx.require_node(&NodeRef { layer: target, index })?;
            Some(Ix::new(index))
        }
        tag => return Err(malformed(format!("an optional entity tag of {tag}"))),
    };
    let is_representative = match r.u8()? {
        0 => false,
        1 => true,
        tag => return Err(malformed(format!("a representative flag of {tag}"))),
    };
    let role = match r.u8()? {
        0 => None,
        b => Some(SyntacticRole::from_byte(b).ok_or_else(|| malformed(format!("a syntactic role of {b}")))?),
    };
    Ok(ConsolidatedMention { span, entity, is_representative, role })
}

impl LayerCodec for DocumentEntities {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.u16(self.view.0);
        let named = self.entities.iter().flat_map(|e| e.mentions.iter()).any(|m| m.entity.is_some());
        w.u8(u8::from(named));
        if named {
            put_ref(&mut w, &cx.resolve(&Key::<Entities>::DEFAULT))?;
        }
        w.usize(self.entities.len())?;
        for e in &self.entities {
            w.str(&e.canonical_name)?;
            w.str(&e.kind)?;
            put_opt_str(&mut w, e.kb_id.as_deref())?;
            match e.geo_coord {
                None => w.u8(0),
                Some((lat, lon)) => {
                    w.u8(1);
                    put_f32(&mut w, lat);
                    put_f32(&mut w, lon);
                }
            }
            match e.country_code {
                None => w.u8(0),
                Some([a, b]) => {
                    w.u8(1);
                    w.u8(a);
                    w.u8(b);
                }
            }
            w.usize(e.mentions.len())?;
            for m in e.mentions.iter() {
                put_mention(&mut w, m);
            }
            let c = e.role_counts;
            for v in [c.subject_count, c.object_count, c.modifier_count, c.other_count] {
                w.u16(v);
            }
            put_f32(&mut w, e.mean_sentiment);
            w.u16(e.sentiment_mentions);
            put_f32(&mut w, e.confidence);
        }
        w.usize(self.relations.len())?;
        for rel in &self.relations {
            w.u32(rel.subject);
            w.u32(rel.object);
            w.str(&rel.kind)?;
            put_f32(&mut w, rel.score);
            w.u16(rel.support);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let view = ViewId(r.u16()?);
        let rows = match r.u8()? {
            0 => None,
            1 => {
                let target = get_ref(&mut r)?;
                cx.expect_target::<Entities>(&target)?;
                Some(target)
            }
            tag => return Err(malformed(format!("an entity layer tag of {tag}"))),
        };
        let mut out = DocumentEntities::new(view);
        for _ in 0..r.usize()? {
            let canonical_name = r.string()?.into();
            let kind = r.string()?.into();
            let kb_id = get_opt_str(&mut r)?;
            let geo_coord = match r.u8()? {
                0 => None,
                1 => Some((get_f32(&mut r)?, get_f32(&mut r)?)),
                tag => return Err(malformed(format!("a coordinate tag of {tag}"))),
            };
            let country_code = match r.u8()? {
                0 => None,
                1 => Some([r.u8()?, r.u8()?]),
                tag => return Err(malformed(format!("a country code tag of {tag}"))),
            };
            let mut mentions = Vec::new();
            for _ in 0..r.usize()? {
                mentions.push(get_mention(&mut r, cx, &rows)?);
            }
            let role_counts = SyntacticRoleCounts {
                subject_count: r.u16()?,
                object_count: r.u16()?,
                modifier_count: r.u16()?,
                other_count: r.u16()?,
            };
            out.entities.push(DocumentEntity {
                canonical_name,
                kind,
                kb_id,
                geo_coord,
                country_code,
                mentions: mentions.into(),
                role_counts,
                mean_sentiment: get_f32(&mut r)?,
                sentiment_mentions: r.u16()?,
                confidence: get_f32(&mut r)?,
            });
        }
        for _ in 0..r.usize()? {
            out.relations.push(EntityRelation {
                subject: r.u32()?,
                object: r.u32()?,
                kind: r.string()?.into(),
                score: get_f32(&mut r)?,
                support: r.u16()?,
            });
        }
        done(&r, "document entities")?;
        Ok(out)
    }
}

impl LayerCodec for EntitySalience {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.scores.len())?;
        for s in &self.scores {
            w.u32(s.entity_ix);
            w.u32(s.rank);
            for v in [s.salience, s.position_score, s.syntax_score, s.frequency_score, s.centrality_score] {
                put_f32(&mut w, v);
            }
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let mut scores = Vec::new();
        for _ in 0..r.usize()? {
            let (entity_ix, rank) = (r.u32()?, r.u32()?);
            scores.push(EntitySalienceScore {
                entity_ix,
                rank,
                salience: get_f32(&mut r)?,
                position_score: get_f32(&mut r)?,
                syntax_score: get_f32(&mut r)?,
                frequency_score: get_f32(&mut r)?,
                centrality_score: get_f32(&mut r)?,
            });
        }
        done(&r, "entity salience")?;
        let out = EntitySalience { scores };
        out.check().map_err(malformed)?;
        Ok(out)
    }
}
