//! Amounts of money, `MoneyPiiExtractor`: the mentions the money scanner
//! of turbo-numeric finds with its default symbol table, amounts read in
//! US notation (`$1,250.00`, `EUR 40`, `40 euros`; docs/money.md). A bare number is never money. The normalized form is
//! the ISO 4217 code and the amount (`USD 1250.00`, `USD -3`), with a
//! rate's unit after a slash where the scanner reads one.

use std::fmt::Write;

use turbo_analysis::ViewId;
use turbo_analysis_layers::{Money, PiiEvidence, PiiKind};
use turbo_numeric::money::MoneyExtractor;

use crate::hits::Hits;

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let mut found = Money::new(ViewId::SOURCE);
    MoneyExtractor::default().scan(text, &mut found);
    for (i, span) in found.spans.iter() {
        let (start, end) = (span.start as usize, span.end as usize);
        hits.add(start, end, PiiKind::Currency, PiiEvidence::Shape, |s| {
            let _ = write!(s, "{} {}", found.currency[i.get()].as_str(), found.amount(i));
            if let Some(per) = found.per(i) {
                let _ = write!(s, "/{per}");
            }
        });
    }
}
