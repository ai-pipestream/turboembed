//! Document-level region ballot annotator (`opennlp.tools.geo.DocumentRegionAnnotator`).
//!
//! Reads `(View, Entities, Locations)` and writes `(Regions,)`, combining:
//! 1. The first [`Locations`] resolution per location-typed entity mention,
//!    weighted by its resolution confidence when its entry carries a country
//!    code.
//! 2. English country display-name matching (`0.95` weight) for location-typed
//!    entity mentions that the geocoder left unresolved or resolved to a
//!    country-less entry.
//! 3. Country flag emoji pairs (`U+1F1E6..=U+1F1FF` decoding to an assigned
//!    ISO 3166-1 alpha-2 country code, scanned left-to-right non-overlapping)
//!    across the document text (`0.95` weight).

use std::collections::{HashMap, HashSet};

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{Entities, MarginError, RegionCode, RegionVote, Regions};
use turbo_unicode::GeneralCategory;
use turbo_unicode::category::general_category;
use turbo_unicode::java::to_lowercase;
use turbo_unicode::normalize::{Form, normalize};

use crate::geocode::{DEFAULT_LOCATION_TYPE, GeocodeError, Locations, normalize_location_types};

const COUNTRY_NAME_CONFIDENCE: f64 = 0.95;
const FLAG_EMOJI_CONFIDENCE: f64 = 0.95;
const REGIONAL_INDICATOR_A: u32 = 0x1F1E6;
const REGIONAL_INDICATOR_Z: u32 = 0x1F1FF;

/// All 249 assigned ISO 3166-1 alpha-2 country codes in JDK 21
/// (`Locale.getISOCountries()`), sorted for binary search.
const ISO_COUNTRIES: [[u8; 2]; 249] = [
    *b"AD", *b"AE", *b"AF", *b"AG", *b"AI", *b"AL", *b"AM", *b"AO", *b"AQ", *b"AR", *b"AS", *b"AT", *b"AU", *b"AW",
    *b"AX", *b"AZ", *b"BA", *b"BB", *b"BD", *b"BE", *b"BF", *b"BG", *b"BH", *b"BI", *b"BJ", *b"BL", *b"BM", *b"BN",
    *b"BO", *b"BQ", *b"BR", *b"BS", *b"BT", *b"BV", *b"BW", *b"BY", *b"BZ", *b"CA", *b"CC", *b"CD", *b"CF", *b"CG",
    *b"CH", *b"CI", *b"CK", *b"CL", *b"CM", *b"CN", *b"CO", *b"CR", *b"CU", *b"CV", *b"CW", *b"CX", *b"CY", *b"CZ",
    *b"DE", *b"DJ", *b"DK", *b"DM", *b"DO", *b"DZ", *b"EC", *b"EE", *b"EG", *b"EH", *b"ER", *b"ES", *b"ET", *b"FI",
    *b"FJ", *b"FK", *b"FM", *b"FO", *b"FR", *b"GA", *b"GB", *b"GD", *b"GE", *b"GF", *b"GG", *b"GH", *b"GI", *b"GL",
    *b"GM", *b"GN", *b"GP", *b"GQ", *b"GR", *b"GS", *b"GT", *b"GU", *b"GW", *b"GY", *b"HK", *b"HM", *b"HN", *b"HR",
    *b"HT", *b"HU", *b"ID", *b"IE", *b"IL", *b"IM", *b"IN", *b"IO", *b"IQ", *b"IR", *b"IS", *b"IT", *b"JE", *b"JM",
    *b"JO", *b"JP", *b"KE", *b"KG", *b"KH", *b"KI", *b"KM", *b"KN", *b"KP", *b"KR", *b"KW", *b"KY", *b"KZ", *b"LA",
    *b"LB", *b"LC", *b"LI", *b"LK", *b"LR", *b"LS", *b"LT", *b"LU", *b"LV", *b"LY", *b"MA", *b"MC", *b"MD", *b"ME",
    *b"MF", *b"MG", *b"MH", *b"MK", *b"ML", *b"MM", *b"MN", *b"MO", *b"MP", *b"MQ", *b"MR", *b"MS", *b"MT", *b"MU",
    *b"MV", *b"MW", *b"MX", *b"MY", *b"MZ", *b"NA", *b"NC", *b"NE", *b"NF", *b"NG", *b"NI", *b"NL", *b"NO", *b"NP",
    *b"NR", *b"NU", *b"NZ", *b"OM", *b"PA", *b"PE", *b"PF", *b"PG", *b"PH", *b"PK", *b"PL", *b"PM", *b"PN", *b"PR",
    *b"PS", *b"PT", *b"PW", *b"PY", *b"QA", *b"RE", *b"RO", *b"RS", *b"RU", *b"RW", *b"SA", *b"SB", *b"SC", *b"SD",
    *b"SE", *b"SG", *b"SH", *b"SI", *b"SJ", *b"SK", *b"SL", *b"SM", *b"SN", *b"SO", *b"SR", *b"SS", *b"ST", *b"SV",
    *b"SX", *b"SY", *b"SZ", *b"TC", *b"TD", *b"TF", *b"TG", *b"TH", *b"TJ", *b"TK", *b"TL", *b"TM", *b"TN", *b"TO",
    *b"TR", *b"TT", *b"TV", *b"TW", *b"TZ", *b"UA", *b"UG", *b"UM", *b"US", *b"UY", *b"UZ", *b"VA", *b"VC", *b"VE",
    *b"VG", *b"VI", *b"VN", *b"VU", *b"WF", *b"WS", *b"YE", *b"YT", *b"ZA", *b"ZM", *b"ZW",
];

/// Normalized English country display names (and their accent-stripped forms)
/// from JDK 21's `Locale.of("", code).getDisplayCountry(Locale.ENGLISH)`,
/// sorted by name for binary search.
const COUNTRY_BY_ENGLISH_NAME: [(&str, [u8; 2]); 256] = [
    ("afghanistan", *b"AF"),
    ("aland islands", *b"AX"),
    ("albania", *b"AL"),
    ("algeria", *b"DZ"),
    ("american samoa", *b"AS"),
    ("andorra", *b"AD"),
    ("angola", *b"AO"),
    ("anguilla", *b"AI"),
    ("antarctica", *b"AQ"),
    ("antigua & barbuda", *b"AG"),
    ("argentina", *b"AR"),
    ("armenia", *b"AM"),
    ("aruba", *b"AW"),
    ("australia", *b"AU"),
    ("austria", *b"AT"),
    ("azerbaijan", *b"AZ"),
    ("bahamas", *b"BS"),
    ("bahrain", *b"BH"),
    ("bangladesh", *b"BD"),
    ("barbados", *b"BB"),
    ("belarus", *b"BY"),
    ("belgium", *b"BE"),
    ("belize", *b"BZ"),
    ("benin", *b"BJ"),
    ("bermuda", *b"BM"),
    ("bhutan", *b"BT"),
    ("bolivia", *b"BO"),
    ("bosnia & herzegovina", *b"BA"),
    ("botswana", *b"BW"),
    ("bouvet island", *b"BV"),
    ("brazil", *b"BR"),
    ("british indian ocean territory", *b"IO"),
    ("british virgin islands", *b"VG"),
    ("brunei", *b"BN"),
    ("bulgaria", *b"BG"),
    ("burkina faso", *b"BF"),
    ("burundi", *b"BI"),
    ("cambodia", *b"KH"),
    ("cameroon", *b"CM"),
    ("canada", *b"CA"),
    ("cape verde", *b"CV"),
    ("caribbean netherlands", *b"BQ"),
    ("cayman islands", *b"KY"),
    ("central african republic", *b"CF"),
    ("chad", *b"TD"),
    ("chile", *b"CL"),
    ("china", *b"CN"),
    ("christmas island", *b"CX"),
    ("cocos (keeling) islands", *b"CC"),
    ("colombia", *b"CO"),
    ("comoros", *b"KM"),
    ("congo - brazzaville", *b"CG"),
    ("congo - kinshasa", *b"CD"),
    ("cook islands", *b"CK"),
    ("costa rica", *b"CR"),
    ("cote d'ivoire", *b"CI"),
    ("croatia", *b"HR"),
    ("cuba", *b"CU"),
    ("curacao", *b"CW"),
    ("curaçao", *b"CW"),
    ("cyprus", *b"CY"),
    ("czechia", *b"CZ"),
    ("côte d'ivoire", *b"CI"),
    ("denmark", *b"DK"),
    ("djibouti", *b"DJ"),
    ("dominica", *b"DM"),
    ("dominican republic", *b"DO"),
    ("ecuador", *b"EC"),
    ("egypt", *b"EG"),
    ("el salvador", *b"SV"),
    ("equatorial guinea", *b"GQ"),
    ("eritrea", *b"ER"),
    ("estonia", *b"EE"),
    ("eswatini", *b"SZ"),
    ("ethiopia", *b"ET"),
    ("falkland islands", *b"FK"),
    ("faroe islands", *b"FO"),
    ("fiji", *b"FJ"),
    ("finland", *b"FI"),
    ("france", *b"FR"),
    ("french guiana", *b"GF"),
    ("french polynesia", *b"PF"),
    ("french southern territories", *b"TF"),
    ("gabon", *b"GA"),
    ("gambia", *b"GM"),
    ("georgia", *b"GE"),
    ("germany", *b"DE"),
    ("ghana", *b"GH"),
    ("gibraltar", *b"GI"),
    ("greece", *b"GR"),
    ("greenland", *b"GL"),
    ("grenada", *b"GD"),
    ("guadeloupe", *b"GP"),
    ("guam", *b"GU"),
    ("guatemala", *b"GT"),
    ("guernsey", *b"GG"),
    ("guinea", *b"GN"),
    ("guinea-bissau", *b"GW"),
    ("guyana", *b"GY"),
    ("haiti", *b"HT"),
    ("heard & mcdonald islands", *b"HM"),
    ("honduras", *b"HN"),
    ("hong kong sar china", *b"HK"),
    ("hungary", *b"HU"),
    ("iceland", *b"IS"),
    ("india", *b"IN"),
    ("indonesia", *b"ID"),
    ("iran", *b"IR"),
    ("iraq", *b"IQ"),
    ("ireland", *b"IE"),
    ("isle of man", *b"IM"),
    ("israel", *b"IL"),
    ("italy", *b"IT"),
    ("jamaica", *b"JM"),
    ("japan", *b"JP"),
    ("jersey", *b"JE"),
    ("jordan", *b"JO"),
    ("kazakhstan", *b"KZ"),
    ("kenya", *b"KE"),
    ("kiribati", *b"KI"),
    ("kuwait", *b"KW"),
    ("kyrgyzstan", *b"KG"),
    ("laos", *b"LA"),
    ("latvia", *b"LV"),
    ("lebanon", *b"LB"),
    ("lesotho", *b"LS"),
    ("liberia", *b"LR"),
    ("libya", *b"LY"),
    ("liechtenstein", *b"LI"),
    ("lithuania", *b"LT"),
    ("luxembourg", *b"LU"),
    ("macao sar china", *b"MO"),
    ("madagascar", *b"MG"),
    ("malawi", *b"MW"),
    ("malaysia", *b"MY"),
    ("maldives", *b"MV"),
    ("mali", *b"ML"),
    ("malta", *b"MT"),
    ("marshall islands", *b"MH"),
    ("martinique", *b"MQ"),
    ("mauritania", *b"MR"),
    ("mauritius", *b"MU"),
    ("mayotte", *b"YT"),
    ("mexico", *b"MX"),
    ("micronesia", *b"FM"),
    ("moldova", *b"MD"),
    ("monaco", *b"MC"),
    ("mongolia", *b"MN"),
    ("montenegro", *b"ME"),
    ("montserrat", *b"MS"),
    ("morocco", *b"MA"),
    ("mozambique", *b"MZ"),
    ("myanmar (burma)", *b"MM"),
    ("namibia", *b"NA"),
    ("nauru", *b"NR"),
    ("nepal", *b"NP"),
    ("netherlands", *b"NL"),
    ("new caledonia", *b"NC"),
    ("new zealand", *b"NZ"),
    ("nicaragua", *b"NI"),
    ("niger", *b"NE"),
    ("nigeria", *b"NG"),
    ("niue", *b"NU"),
    ("norfolk island", *b"NF"),
    ("north korea", *b"KP"),
    ("north macedonia", *b"MK"),
    ("northern mariana islands", *b"MP"),
    ("norway", *b"NO"),
    ("oman", *b"OM"),
    ("pakistan", *b"PK"),
    ("palau", *b"PW"),
    ("palestinian territories", *b"PS"),
    ("panama", *b"PA"),
    ("papua new guinea", *b"PG"),
    ("paraguay", *b"PY"),
    ("peru", *b"PE"),
    ("philippines", *b"PH"),
    ("pitcairn islands", *b"PN"),
    ("poland", *b"PL"),
    ("portugal", *b"PT"),
    ("puerto rico", *b"PR"),
    ("qatar", *b"QA"),
    ("reunion", *b"RE"),
    ("romania", *b"RO"),
    ("russia", *b"RU"),
    ("rwanda", *b"RW"),
    ("réunion", *b"RE"),
    ("samoa", *b"WS"),
    ("san marino", *b"SM"),
    ("sao tome & principe", *b"ST"),
    ("saudi arabia", *b"SA"),
    ("senegal", *b"SN"),
    ("serbia", *b"RS"),
    ("seychelles", *b"SC"),
    ("sierra leone", *b"SL"),
    ("singapore", *b"SG"),
    ("sint maarten", *b"SX"),
    ("slovakia", *b"SK"),
    ("slovenia", *b"SI"),
    ("solomon islands", *b"SB"),
    ("somalia", *b"SO"),
    ("south africa", *b"ZA"),
    ("south georgia & south sandwich islands", *b"GS"),
    ("south korea", *b"KR"),
    ("south sudan", *b"SS"),
    ("spain", *b"ES"),
    ("sri lanka", *b"LK"),
    ("st. barthelemy", *b"BL"),
    ("st. barthélemy", *b"BL"),
    ("st. helena", *b"SH"),
    ("st. kitts & nevis", *b"KN"),
    ("st. lucia", *b"LC"),
    ("st. martin", *b"MF"),
    ("st. pierre & miquelon", *b"PM"),
    ("st. vincent & grenadines", *b"VC"),
    ("sudan", *b"SD"),
    ("suriname", *b"SR"),
    ("svalbard & jan mayen", *b"SJ"),
    ("sweden", *b"SE"),
    ("switzerland", *b"CH"),
    ("syria", *b"SY"),
    ("são tomé & príncipe", *b"ST"),
    ("taiwan", *b"TW"),
    ("tajikistan", *b"TJ"),
    ("tanzania", *b"TZ"),
    ("thailand", *b"TH"),
    ("timor-leste", *b"TL"),
    ("togo", *b"TG"),
    ("tokelau", *b"TK"),
    ("tonga", *b"TO"),
    ("trinidad & tobago", *b"TT"),
    ("tunisia", *b"TN"),
    ("turkiye", *b"TR"),
    ("turkmenistan", *b"TM"),
    ("turks & caicos islands", *b"TC"),
    ("tuvalu", *b"TV"),
    ("türkiye", *b"TR"),
    ("u.s. outlying islands", *b"UM"),
    ("u.s. virgin islands", *b"VI"),
    ("uganda", *b"UG"),
    ("ukraine", *b"UA"),
    ("united arab emirates", *b"AE"),
    ("united kingdom", *b"GB"),
    ("united states", *b"US"),
    ("uruguay", *b"UY"),
    ("uzbekistan", *b"UZ"),
    ("vanuatu", *b"VU"),
    ("vatican city", *b"VA"),
    ("venezuela", *b"VE"),
    ("vietnam", *b"VN"),
    ("wallis & futuna", *b"WF"),
    ("western sahara", *b"EH"),
    ("yemen", *b"YE"),
    ("zambia", *b"ZM"),
    ("zimbabwe", *b"ZW"),
    ("åland islands", *b"AX"),
];

fn normalize_country_name(text: &str) -> String {
    let mut nfkc = String::with_capacity(text.len());
    normalize(Form::Nfkc, text, &mut nfkc);
    let mut replaced = String::with_capacity(nfkc.len());
    for c in nfkc.chars() {
        replaced.push(if c == '\u{2019}' { '\'' } else { c });
    }
    let mut lower = String::with_capacity(replaced.len());
    to_lowercase(&replaced, &mut lower);
    let trimmed = lower.trim_matches(|c: char| c <= '\u{0020}');
    trimmed.to_string()
}

fn strip_accents(text: &str) -> String {
    let mut decomposed = String::with_capacity(text.len());
    normalize(Form::Nfd, text, &mut decomposed);
    decomposed.chars().filter(|&c| general_category(c) != GeneralCategory::Mn).collect()
}

fn lookup_english_country(name: &str) -> Option<RegionCode> {
    COUNTRY_BY_ENGLISH_NAME
        .binary_search_by_key(&name, |&(k, _)| k)
        .ok()
        .and_then(|idx| RegionCode::new(COUNTRY_BY_ENGLISH_NAME[idx].1))
}

fn country_code_for_name(mention: &str) -> Option<RegionCode> {
    let normalized = normalize_country_name(mention);
    if let Some(code) = lookup_english_country(&normalized) {
        return Some(code);
    }
    let stripped = strip_accents(&normalized);
    lookup_english_country(&stripped)
}

fn is_regional_indicator(c: char) -> bool {
    let cp = c as u32;
    (REGIONAL_INDICATOR_A..=REGIONAL_INDICATOR_Z).contains(&cp)
}

fn add_vote(weights: &mut HashMap<RegionCode, f64>, country: RegionCode, weight: f64) {
    if weight > 0.0 {
        *weights.entry(country).or_insert(0.0) += weight;
    }
}

fn add_flag_emoji_votes(text: &str, weights: &mut HashMap<RegionCode, f64>) {
    let mut chars = text.chars().peekable();
    while let Some(first) = chars.next() {
        if !is_regional_indicator(first) {
            continue;
        }
        let Some(&second) = chars.peek() else {
            break;
        };
        if !is_regional_indicator(second) {
            continue;
        }
        let _ = chars.next();
        let bytes =
            [b'A' + (first as u32 - REGIONAL_INDICATOR_A) as u8, b'A' + (second as u32 - REGIONAL_INDICATOR_A) as u8];
        if ISO_COUNTRIES.binary_search(&bytes).is_ok()
            && let Some(code) = RegionCode::new(bytes)
        {
            add_vote(weights, code, FLAG_EMOJI_CONFIDENCE);
        }
    }
}

/// Aggregates resolved location mentions, English country-name mentions and
/// country flag emojis into the document-scoped `turbo:regions` ballot
/// (`opennlp.tools.geo.DocumentRegionAnnotator`).
pub struct DocumentRegionAnnotator {
    location_types: HashSet<String>,
}

impl Default for DocumentRegionAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentRegionAnnotator {
    /// Creates a region annotator that considers entities typed `"location"`
    /// (case-insensitively).
    pub fn new() -> DocumentRegionAnnotator {
        let mut location_types = HashSet::new();
        location_types.insert(DEFAULT_LOCATION_TYPE.to_string());
        DocumentRegionAnnotator { location_types }
    }

    /// Creates a region annotator that considers entities whose type label
    /// matches any element of `location_types` case-insensitively (`Locale.ROOT`).
    pub fn with_location_types<'a>(
        location_types: impl IntoIterator<Item = &'a str>,
    ) -> Result<DocumentRegionAnnotator, GeocodeError> {
        Ok(DocumentRegionAnnotator { location_types: normalize_location_types(location_types)? })
    }

    /// Returns the winning vote when the top row of `ballot` leads the runner-up
    /// (or `0.0` when `ballot` has one row) by at least `min_margin`
    /// (`DocumentRegionAnnotator.winner`).
    pub fn winner(ballot: &Regions, min_margin: f64) -> Result<Option<RegionVote>, MarginError> {
        ballot.winner(min_margin)
    }
}

impl Annotator for DocumentRegionAnnotator {
    const ID: &'static str = "turbo-geo:document-region";
    type Reads = (View, Entities, Locations);
    type Writes = (Regions,);

    fn annotate(&self, (view, entities, locations): Refs<'_, Self>) -> Result<(Regions,), Error> {
        if entities.spans.view() != view.id() || locations.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "entities or locations are in a different view".into(),
            });
        }

        let mut first_by_span: HashMap<Span, usize> = HashMap::new();
        for (ix, span) in locations.spans.iter() {
            first_by_span.entry(span).or_insert(ix.get());
        }

        let mut weights: HashMap<RegionCode, f64> = HashMap::new();
        let mut lower_buf = String::new();
        let text = view.text();

        for (ix, span) in entities.spans.iter() {
            lower_buf.clear();
            to_lowercase(entities.kinds.label(entities.kind[ix.get()]), &mut lower_buf);
            if !self.location_types.contains(&lower_buf) {
                continue;
            }
            if let Some(&loc_idx) = first_by_span.get(&span)
                && let Some(country) = locations.entry[loc_idx].region_code()
            {
                add_vote(&mut weights, country, locations.confidence[loc_idx]);
                continue;
            }
            let surface = &text[span.start as usize..span.end as usize];
            if let Some(country) = country_code_for_name(surface) {
                add_vote(&mut weights, country, COUNTRY_NAME_CONFIDENCE);
            }
        }

        add_flag_emoji_votes(text, &mut weights);

        let total: f64 = weights.values().sum();
        let mut ballot = Regions::new();
        if total <= 0.0 {
            return Ok((ballot,));
        }

        let mut rows: Vec<RegionVote> =
            weights.into_iter().filter_map(|(country, w)| RegionVote::new(country, w / total).ok()).collect();
        rows.sort_by(|a, b| b.share.total_cmp(&a.share).then_with(|| a.country_code.cmp(&b.country_code)));
        for row in rows {
            ballot.push(row);
        }
        Ok((ballot,))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::{EntryParts, FEATURE_CLASS_CITY, GazetteerEntry, GeoPoint};
    use crate::geocode::{GeoResolution, GeocodeAnnotator, GeocodeError, Geocoder};
    use turbo_analysis::{Analyzer, Document, IxRange, Key, Layer, Provenance, Spans, Tag, TagSet, ViewId};
    use turbo_analysis_layers::Tokens;

    const SHARE_DELTA: f64 = 1e-9;
    const TABLE_CONFIDENCE: f64 = 0.8;

    fn make_entry(id: &str, name: &str, country: Option<&str>) -> GazetteerEntry {
        GazetteerEntry::new(EntryParts {
            source: "test".into(),
            record_id: id.into(),
            name: name.into(),
            alternate_names: Vec::new(),
            location: GeoPoint::new(0.0, 0.0).unwrap(),
            bounding_box: None,
            country_code: country.map(Into::into),
            containment: Vec::new(),
            population: 1000,
            feature_class: Some(FEATURE_CLASS_CITY.into()),
            attributes: Vec::new(),
        })
        .unwrap()
    }

    struct TableGeocoder {
        entries: HashMap<String, (Option<String>, f64)>,
    }

    impl TableGeocoder {
        fn uniform(pairs: &[(&str, &str)], confidence: f64) -> TableGeocoder {
            let entries = pairs.iter().map(|&(k, v)| (k.to_string(), (Some(v.to_string()), confidence))).collect();
            TableGeocoder { entries }
        }

        fn scored(pairs: &[(&str, Option<&str>, f64)]) -> TableGeocoder {
            let entries = pairs.iter().map(|&(k, v, c)| (k.to_string(), (v.map(Into::into), c))).collect();
            TableGeocoder { entries }
        }
    }

    impl Geocoder for TableGeocoder {
        fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
            let mut out = Vec::new();
            for &m in location_mentions {
                let surface = &text[m.start as usize..m.end as usize];
                if let Some((country, conf)) = self.entries.get(surface) {
                    out.push(GeoResolution::new(m, make_entry(surface, surface, country.as_deref()), *conf)?);
                }
            }
            Ok(out)
        }
    }

    fn doc_with_location_entities(text: &str, names: &[&str]) -> Document {
        let mut doc = Document::new(text);
        doc.add(&Key::DEFAULT, Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None }, Provenance::by("test"))
            .unwrap();
        let kinds = TagSet::new("ner", &["location"]);
        let mut spans = Spans::new(ViewId::SOURCE);
        let mut tokens = Vec::new();
        let mut kind = Vec::new();
        let mut score = Vec::new();
        for &name in names {
            let start = text.find(name).unwrap();
            spans.push(Span::new(start as u32, (start + name.len()) as u32));
            tokens.push(IxRange::new(0, 0));
            kind.push(Tag(0));
            score.push(1.0);
        }
        doc.add(&Key::DEFAULT, Entities { spans, tokens, kind, kinds, score }, Provenance::by("test")).unwrap();
        doc
    }

    fn run_pipeline<G: Geocoder + Send + Sync + 'static>(geocoder: G, mut doc: Document) -> Document {
        let a = Analyzer::builder()
            .add(GeocodeAnnotator::new(geocoder))
            .add(DocumentRegionAnnotator::new())
            .given(Key::<Entities>::DEFAULT)
            .build()
            .unwrap();
        a.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
        doc
    }

    #[test]
    fn country_tables_are_sorted() {
        assert!(ISO_COUNTRIES.windows(2).all(|w| w[0] < w[1]));
        assert!(COUNTRY_BY_ENGLISH_NAME.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn geocoded_mentions_vote_with_confidence() {
        let geocoder =
            TableGeocoder::uniform(&[("Sydney", "AU"), ("Melbourne", "AU"), ("Auckland", "NZ")], TABLE_CONFIDENCE);
        let doc = run_pipeline(
            geocoder,
            doc_with_location_entities(
                "flights from Sydney and Melbourne to Auckland",
                &["Sydney", "Melbourne", "Auckland"],
            ),
        );
        let ballot = doc.get::<Regions>().unwrap();
        assert_eq!(ballot.len(), 2);
        assert_eq!(ballot.get(0).unwrap().country_code(), "AU");
        assert!((ballot.get(0).unwrap().share() - 2.0 / 3.0).abs() < SHARE_DELTA);
        assert_eq!(ballot.get(1).unwrap().country_code(), "NZ");
    }

    #[test]
    fn country_names_vote_when_unresolved_and_resolved_mentions_override_name_table() {
        let doc = run_pipeline(
            TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
            doc_with_location_entities("mining exports from Australia rose", &["Australia"]),
        );
        let ballot = doc.get::<Regions>().unwrap();
        assert_eq!(ballot.len(), 1);
        assert_eq!(ballot.get(0).unwrap().country_code(), "AU");
        assert!((ballot.get(0).unwrap().share() - 1.0).abs() < SHARE_DELTA);

        // Resolved Georgia -> US overrides country name GE.
        let doc_ga = run_pipeline(
            TableGeocoder::uniform(&[("Georgia", "US"), ("Atlanta", "US")], TABLE_CONFIDENCE),
            doc_with_location_entities(
                "Georgia's secretary of state confirmed the recount in Atlanta",
                &["Georgia", "Atlanta"],
            ),
        );
        let ballot_ga = doc_ga.get::<Regions>().unwrap();
        assert_eq!(ballot_ga.len(), 1);
        assert_eq!(ballot_ga.get(0).unwrap().country_code(), "US");

        // Resolved Australia casts one vote at its resolution confidence, tying Auckland.
        let doc_tie = run_pipeline(
            TableGeocoder::uniform(&[("Australia", "AU"), ("Auckland", "NZ")], TABLE_CONFIDENCE),
            doc_with_location_entities("exports from Australia to Auckland rose", &["Australia", "Auckland"]),
        );
        let ballot_tie = doc_tie.get::<Regions>().unwrap();
        assert_eq!(ballot_tie.len(), 2);
        assert_eq!(ballot_tie.get(0).unwrap().country_code(), "AU");
        assert!((ballot_tie.get(0).unwrap().share() - 0.5).abs() < SHARE_DELTA);
        assert_eq!(ballot_tie.get(1).unwrap().country_code(), "NZ");
        assert!((ballot_tie.get(1).unwrap().share() - 0.5).abs() < SHARE_DELTA);

        // Unresolved country name (0.95) outweighs geocoded city (0.80).
        let doc_out = run_pipeline(
            TableGeocoder::uniform(&[("Sydney", "AU")], TABLE_CONFIDENCE),
            doc_with_location_entities(
                "the Sydney office reports to New Zealand headquarters",
                &["Sydney", "New Zealand"],
            ),
        );
        let ballot_out = doc_out.get::<Regions>().unwrap();
        assert_eq!(ballot_out.len(), 2);
        assert_eq!(ballot_out.get(0).unwrap().country_code(), "NZ");
        assert!((ballot_out.get(0).unwrap().share() - 0.95 / (0.95 + TABLE_CONFIDENCE)).abs() < SHARE_DELTA);
        assert_eq!(ballot_out.get(1).unwrap().country_code(), "AU");
        assert_eq!(DocumentRegionAnnotator::winner(ballot_out, 0.05).unwrap().unwrap().country_code(), "NZ");
        assert_eq!(DocumentRegionAnnotator::winner(ballot_out, 0.1).unwrap(), None);
    }

    #[test]
    fn country_name_spelling_variants_match() {
        let cases = [
            ("Côte d\u{2019}Ivoire", "CI"),
            ("Côte d'Ivoire", "CI"),
            ("Cote d'Ivoire", "CI"),
            ("México", "MX"),
            ("AUSTRALIA", "AU"),
            ("australia", "AU"),
        ];
        for (mention, expected) in cases {
            let text = format!("new offices in {mention} opened");
            let doc = run_pipeline(
                TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
                doc_with_location_entities(&text, &[mention]),
            );
            let ballot = doc.get::<Regions>().unwrap();
            assert_eq!(ballot.len(), 1, "{mention}");
            assert_eq!(ballot.get(0).unwrap().country_code(), expected, "{mention}");
        }
    }

    #[test]
    fn flag_emojis_vote_and_invalid_flags_are_ignored() {
        // Australia flag U+1F1E6 U+1F1FA.
        let doc = run_pipeline(
            TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
            doc_with_location_entities("shipping update \u{1F1E6}\u{1F1FA} soon", &[]),
        );
        let ballot = doc.get::<Regions>().unwrap();
        assert_eq!(ballot.len(), 1);
        assert_eq!(ballot.get(0).unwrap().country_code(), "AU");

        // Consecutive FR and DE flags segment left-to-right and tie by country code (DE before FR).
        let doc2 = run_pipeline(
            TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
            doc_with_location_entities("match report \u{1F1EB}\u{1F1F7}\u{1F1E9}\u{1F1EA} today", &[]),
        );
        let ballot2 = doc2.get::<Regions>().unwrap();
        assert_eq!(ballot2.len(), 2);
        assert_eq!(ballot2.get(0).unwrap().country_code(), "DE");
        assert_eq!(ballot2.get(1).unwrap().country_code(), "FR");

        // Disagreeing flags blend with geocoded evidence.
        let doc3 = run_pipeline(
            TableGeocoder::uniform(&[("Sydney", "AU")], TABLE_CONFIDENCE),
            doc_with_location_entities("Sydney update \u{1F1EB}\u{1F1F7} \u{1F1E9}\u{1F1EA}", &["Sydney"]),
        );
        let ballot3 = doc3.get::<Regions>().unwrap();
        assert_eq!(ballot3.len(), 3);
        assert_eq!(ballot3.get(0).unwrap().country_code(), "DE");
        assert!((ballot3.get(0).unwrap().share() - 0.95 / 2.7).abs() < SHARE_DELTA);
        assert_eq!(ballot3.get(1).unwrap().country_code(), "FR");
        assert_eq!(ballot3.get(2).unwrap().country_code(), "AU");
        assert!((ballot3.get(2).unwrap().share() - 0.8 / 2.7).abs() < SHARE_DELTA);

        // Lone regional indicators do not vote.
        let doc4 = run_pipeline(
            TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
            doc_with_location_entities("grade \u{1F1E6} list \u{1F1FA}", &[]),
        );
        assert!(doc4.get::<Regions>().unwrap().is_empty());

        // Subdivision tag-sequence flag (England GB-ENG) does not vote.
        let england = "match report \u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F} today";
        let doc5 =
            run_pipeline(TableGeocoder::uniform(&[], TABLE_CONFIDENCE), doc_with_location_entities(england, &[]));
        assert!(doc5.get::<Regions>().unwrap().is_empty());

        // Unassigned XX pair does not vote.
        let doc6 = run_pipeline(
            TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
            doc_with_location_entities("status \u{1F1FD}\u{1F1FD} checked", &[]),
        );
        assert!(doc6.get::<Regions>().unwrap().is_empty());
    }

    #[test]
    fn ballot_edge_cases_tie_breaking_zero_confidence_and_null_country() {
        for text in ["trains between London and Paris", "trains between Paris and London"] {
            let doc = run_pipeline(
                TableGeocoder::scored(&[("London", Some("GB"), 0.8), ("Paris", Some("FR"), 0.8)]),
                doc_with_location_entities(text, &["London", "Paris"]),
            );
            let ballot = doc.get::<Regions>().unwrap();
            assert_eq!(ballot.len(), 2);
            assert_eq!(ballot.get(0).unwrap().country_code(), "FR");
            assert_eq!(ballot.get(1).unwrap().country_code(), "GB");
        }

        // One confident mention outvotes two weak ones.
        let doc = run_pipeline(
            TableGeocoder::scored(&[
                ("Nice", Some("FR"), 0.3),
                ("Nancy", Some("FR"), 0.3),
                ("Chicago", Some("US"), 0.7),
            ]),
            doc_with_location_entities("flights from Nice and Nancy to Chicago", &["Nice", "Nancy", "Chicago"]),
        );
        let ballot = doc.get::<Regions>().unwrap();
        assert_eq!(ballot.len(), 2);
        assert_eq!(ballot.get(0).unwrap().country_code(), "US");
        assert_eq!(ballot.get(0).unwrap().share(), 0.7 / (0.7 + 0.6));
        assert_eq!(ballot.get(1).unwrap().country_code(), "FR");

        // Zero-confidence resolution casts no vote and does not dilute.
        let doc_zero = run_pipeline(
            TableGeocoder::scored(&[("Bilbao", Some("ES"), 0.0)]),
            doc_with_location_entities("a dispatch from Bilbao", &["Bilbao"]),
        );
        assert!(doc_zero.get::<Regions>().unwrap().is_empty());

        let doc_zero2 = run_pipeline(
            TableGeocoder::scored(&[("Bilbao", Some("ES"), 0.0), ("Sydney", Some("AU"), 0.8)]),
            doc_with_location_entities("flights from Bilbao to Sydney", &["Bilbao", "Sydney"]),
        );
        let ballot_zero2 = doc_zero2.get::<Regions>().unwrap();
        assert_eq!(ballot_zero2.len(), 1);
        assert_eq!(ballot_zero2.get(0).unwrap().country_code(), "AU");
        assert_eq!(ballot_zero2.get(0).unwrap().share(), 1.0);

        // Null country code resolution casts no vote, or falls back to country name if mention is a country name.
        let doc_null = run_pipeline(
            TableGeocoder::scored(&[("Pacific", None, 0.9), ("Sydney", Some("AU"), 0.8)]),
            doc_with_location_entities("across the Pacific to Sydney", &["Pacific", "Sydney"]),
        );
        let ballot_null = doc_null.get::<Regions>().unwrap();
        assert_eq!(ballot_null.len(), 1);
        assert_eq!(ballot_null.get(0).unwrap().country_code(), "AU");

        let doc_country_fallback = run_pipeline(
            TableGeocoder::scored(&[("Australia", None, 0.8)]),
            doc_with_location_entities("mining exports from Australia rose", &["Australia"]),
        );
        let ballot_cf = doc_country_fallback.get::<Regions>().unwrap();
        assert_eq!(ballot_cf.len(), 1);
        assert_eq!(ballot_cf.get(0).unwrap().country_code(), "AU");
    }
}
