//! `ZipLocationAnnotator`: the place of each ZIP Code the PII detector
//! found, as a `turbo:locations` layer under the key `zip`.

use turbo_analysis::{Annotator, Bound, Error, Key, Refs, Span, View};
use turbo_analysis_layers::{Pii, PiiKind};
use turbo_geo::{GeoResolution, Locations};

use turbo_geo::{AttributeValue, GazetteerEntry};

use crate::{UsZipTable, UspsLocales, Zcta, zcta_entry};

/// The attribute key of a ZIP Code's postal locale names, `|` separated.
pub const ATTRIBUTE_KEY_USPS_LOCALE: &str = "usps-locale";
/// The source of that attribute.
pub const SOURCE_USPS_LOCALES: &str = "usps-zip-locale";

/// The key of the layer it writes, so it sits beside the geocoder's
/// default `turbo:locations` rather than replacing it.
pub const ZIP_LOCATIONS: Key<Locations> = Key::named("zip");

/// Reads `turbo:pii` and writes, for each `us-zip` mention whose ZIP Code
/// has a ZCTA, that ZCTA's record ([`zcta_entry`]) over the mention's span
/// with confidence 1, into `turbo:locations` keyed [`ZIP_LOCATIONS`] when
/// added through [`ZipLocationAnnotator::bound`]. A
/// ZIP Code with no ZCTA (a PO box, single organization or military code)
/// adds nothing.
///
/// The record's name, record id and `zcta` attribute are the ZIP Code in
/// plain text, where `turbo:pii` keeps it as a sensitive value: add this
/// step only where the locations layer may disclose the ZIP Code.
pub struct ZipLocationAnnotator {
    table: &'static UsZipTable,
    usps: Option<UspsLocales>,
}

impl ZipLocationAnnotator {
    /// Over the bundled tables.
    pub fn new() -> ZipLocationAnnotator {
        ZipLocationAnnotator { table: UsZipTable::bundled(), usps: None }
    }

    /// Over other tables, such as ones built from later Census files.
    pub fn with_table(table: &'static UsZipTable) -> ZipLocationAnnotator {
        ZipLocationAnnotator { table, usps: None }
    }

    /// Adds the caller's USPS locale list: a ZIP Code's locale names
    /// become its record's alternate names and its `usps-locale`
    /// attribute. It places no ZIP Code that has no ZCTA, since the list
    /// holds no point.
    pub fn with_usps(mut self, locales: UspsLocales) -> ZipLocationAnnotator {
        self.usps = Some(locales);
        self
    }

    fn entry(&self, z: &Zcta<'_>, zip: &str) -> GazetteerEntry {
        let e = zcta_entry(z);
        let Some(names) = self.usps.as_ref().map(|u| u.locales(zip).map(|(l, _)| l.to_string()).collect::<Vec<_>>())
        else {
            return e;
        };
        if names.is_empty() {
            return e;
        }
        let mut parts = e.into_parts();
        let value =
            AttributeValue::new(names.join("|"), SOURCE_USPS_LOCALES, "").expect("locale names are never blank");
        parts.attributes.push((ATTRIBUTE_KEY_USPS_LOCALE.to_string(), value));
        parts.alternate_names = names;
        GazetteerEntry::new(parts).expect("locale names are never blank")
    }

    /// The step for an analyzer, writing its layer as [`ZIP_LOCATIONS`].
    /// Added unbound, it writes the default `turbo:locations` instead.
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(0, ZIP_LOCATIONS.name)
    }

    /// The locations of the ZIP Codes in `pii`, over the view `pii` is on.
    pub fn locations(&self, pii: &Pii) -> Locations {
        let mut found: Vec<(Span, &str)> =
            pii.mentions().filter(|m| m.kind == PiiKind::UsZip).map(|m| (m.span, m.normalized)).collect();
        found.sort_by_key(|(span, _)| (span.start, span.end));
        let mut out = Locations::new(pii.spans.view());
        for (span, zip) in found {
            if let Some(z) = self.table.zcta(zip) {
                let r = GeoResolution::new(span, self.entry(&z, zip), 1.0).expect("confidence 1 is in range");
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

    fn config_hash(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        h.u64(self.table.digest());
        match &self.usps {
            None => h.u8(0),
            Some(u) => h.u8(1).u64(u.digest()),
        };
        h.finish()
    }
}
