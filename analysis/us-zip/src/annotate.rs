//! `ZipLocationAnnotator`: the place of each ZIP Code the PII detector
//! found, as a `turbo:locations` layer under the key `zip`.

use turbo_analysis::{Annotator, Error, Key, KeyId, Refs, Span, View};
use turbo_analysis_layers::{Pii, PiiKind};
use turbo_geo::{GeoResolution, Locations};

use crate::{UsZipTable, zcta_entry};

/// The key of the layer it writes, so it sits beside the geocoder's
/// default `turbo:locations` rather than replacing it.
pub const ZIP_LOCATIONS: Key<Locations> = Key::named("zip");

/// Reads `turbo:pii` and writes, for each `us-zip` mention whose ZIP Code
/// has a ZCTA, that ZCTA's record ([`zcta_entry`]) over the mention's span
/// with confidence 1, into `turbo:locations` keyed [`ZIP_LOCATIONS`]. A
/// ZIP Code with no ZCTA (a PO box, single organization or military code)
/// adds nothing.
///
/// The record's name, record id and `zcta` attribute are the ZIP Code in
/// plain text, where `turbo:pii` keeps it as a sensitive value: add this
/// step only where the locations layer may disclose the ZIP Code.
pub struct ZipLocationAnnotator {
    table: &'static UsZipTable,
}

impl ZipLocationAnnotator {
    /// Over the bundled tables.
    pub fn new() -> ZipLocationAnnotator {
        ZipLocationAnnotator { table: UsZipTable::bundled() }
    }

    /// Over other tables, such as ones built from later Census files.
    pub fn with_table(table: &'static UsZipTable) -> ZipLocationAnnotator {
        ZipLocationAnnotator { table }
    }

    /// The locations of the ZIP Codes in `pii`, over the view `pii` is on.
    pub fn locations(&self, pii: &Pii) -> Locations {
        let mut found: Vec<(Span, &str)> =
            pii.mentions().filter(|m| m.kind == PiiKind::UsZip).map(|m| (m.span, m.normalized)).collect();
        found.sort_by_key(|(span, _)| (span.start, span.end));
        let mut out = Locations::new(pii.spans.view());
        for (span, zip) in found {
            if let Some(z) = self.table.zcta(zip) {
                let r = GeoResolution::new(span, zcta_entry(&z), 1.0).expect("confidence 1 is in range");
                out.push(r);
            }
        }
        out
    }
}

impl Default for ZipLocationAnnotator {
    fn default() -> Self {
        ZipLocationAnnotator::new()
    }
}

impl Annotator for ZipLocationAnnotator {
    const ID: &'static str = "turbo-us-zip:zip-locations";
    type Reads = (View, Pii);
    type Writes = (Locations,);

    fn annotate(&self, (view, pii): Refs<'_, Self>) -> Result<(Locations,), Error> {
        if pii.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "pii is in a different view".into() });
        }
        let out = self.locations(pii);
        turbo_analysis::Layer::check(&out).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((out,))
    }

    fn produces(&self) -> &[KeyId] {
        const PRODUCES: &[KeyId] = &[ZIP_LOCATIONS.id()];
        PRODUCES
    }

    fn config_hash(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        h.usize(self.table.len());
        h.finish()
    }
}
