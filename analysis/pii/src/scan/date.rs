//! Calendar dates, `TemporalPiiExtractor`: the absolute mentions the
//! temporal scanner of turbo-numeric finds (docs/temporal.md) that name a
//! day, an ISO week or a month (`2026-07-14`, `July 14, 2026`,
//! `14th July 2026`, `July 2026`). A year alone or a quarter
//! is not a date element HIPAA counts, and a relative expression
//! (`3 weeks ago`) needs a reference date this scanner never takes, so
//! neither is a mention. The normalized form is the value in ISO 8601
//! (`2026-07-14`, a week as `2026-W29`, `2026-07`).

use std::fmt::Write;

use turbo_analysis::ViewId;
use turbo_analysis_layers::{Granularity, Origin, PiiEvidence, PiiKind, Temporals};

use crate::hits::Hits;

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let mut found = Temporals::new(ViewId::SOURCE);
    turbo_numeric::temporal::scan(text, None, &mut found);
    for (i, _) in found.spans.iter() {
        let m = found.get(i);
        let element = matches!(m.value.granularity(), Granularity::Day | Granularity::Week | Granularity::Month);
        if m.origin != Origin::Absolute || !element {
            continue;
        }
        let (start, end) = (m.span.start as usize, m.span.end as usize);
        hits.add(start, end, PiiKind::Date, PiiEvidence::Shape, |s| {
            let _ = write!(s, "{}", m.value);
        });
    }
}
