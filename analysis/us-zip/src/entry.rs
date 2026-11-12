//! A ZCTA as a gazetteer record, for the `turbo:locations` layer and
//! anything else that reads [`GazetteerEntry`].

use turbo_geo::{
    ATTRIBUTE_KEY_FIPS, ATTRIBUTE_KEY_GEOID, ATTRIBUTE_KEY_ZCTA, AttributeValue, EntryParts, GazetteerEntry,
};

use crate::Zcta;

/// The source of a ZCTA record and of its attributes.
pub const SOURCE_CENSUS_ZCTA: &str = "census-zcta";

/// The record of a ZCTA: its code as the name and record id, its point and
/// box, country `US`, the containment state, then the county and (when it
/// has one) the place holding most of its land in that state, outermost
/// first, and as attributes the
/// ZCTA code (`zcta`), that county's FIPS code (`fips`) and that place's
/// GEOID (`geoid`). The feature class is `POSTAL`.
pub fn zcta_entry(z: &Zcta<'_>) -> GazetteerEntry {
    let code = z.code();
    // A border ZCTA can overlap a county or place across the line; the
    // record names those of its own state only, so it never mixes two.
    let county = z.counties().map(|(c, _)| c).find(|c| c.state() == z.state());
    let place = z.places().map(|(p, _)| p).find(|p| p.state() == z.state());
    let mut containment = vec![z.state().to_string()];
    let attr = |v: String| AttributeValue::new(v, SOURCE_CENSUS_ZCTA, "").expect("a code is never blank");
    let mut attributes = vec![(ATTRIBUTE_KEY_ZCTA.to_string(), attr(code.clone()))];
    if let Some(c) = county {
        containment.push(c.name().to_string());
        attributes.push((ATTRIBUTE_KEY_FIPS.to_string(), attr(c.geoid())));
    }
    if let Some(p) = place {
        containment.push(p.name().to_string());
        attributes.push((ATTRIBUTE_KEY_GEOID.to_string(), attr(p.geoid())));
    }
    GazetteerEntry::new(EntryParts {
        source: SOURCE_CENSUS_ZCTA.to_string(),
        record_id: code.clone(),
        name: code,
        alternate_names: Vec::new(),
        location: z.point(),
        bounding_box: Some(z.bounding_box()),
        country_code: Some("US".to_string()),
        containment,
        population: 0,
        feature_class: Some("POSTAL".to_string()),
        attributes,
    })
    .expect("a ZCTA record is well formed")
}
