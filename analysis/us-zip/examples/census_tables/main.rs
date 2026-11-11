//! Builds the bundled tables of `turbo-us-zip` from US Census Bureau files
//! (public domain):
//!
//! ```text
//! cargo run --release -p turbo-us-zip --example census_tables -- <census dir> <out dir>
//! ```
//!
//! `<census dir>` holds, unpacked, the 2024 ZCTA Gazetteer
//! (`2024_Gaz_zcta_national.txt`), the 2020 place Gazetteer
//! (`2020_Gaz_place_national.txt`), the 2020 ZCTA to county and ZCTA to
//! place relationship files (`tab20_zcta520_county20_natl.txt`,
//! `tab20_zcta520_place20_natl.txt`), the 2020 county list
//! (`national_county2020.txt`) and the 2020 cartographic boundary ZCTA
//! shapefile (`cb_2020_us_zcta520_500k.shp` and `.dbf`). It writes
//! `us-zcta.txt`, `us-places.txt`, `us-counties.txt` and `zip3-states.txt` to `<out dir>`;
//! each file's header names the inputs and their SHA-256 digests.

mod read;

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

use read::{Table, fixed, shape_boxes, text};

const ZCTA_GAZ: &str = "2024_Gaz_zcta_national.txt";
const PLACE_GAZ: &str = "2020_Gaz_place_national.txt";
const ZCTA_COUNTY: &str = "tab20_zcta520_county20_natl.txt";
const ZCTA_PLACE: &str = "tab20_zcta520_place20_natl.txt";
const COUNTIES: &str = "national_county2020.txt";
const SHP: &str = "cb_2020_us_zcta520_500k.shp";
const DBF: &str = "cb_2020_us_zcta520_500k.dbf";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [census, out] = args.as_slice() else {
        eprintln!("usage: census_tables <census dir> <out dir>");
        std::process::exit(2);
    };
    let (census, out) = (Path::new(census), Path::new(out));
    let mut sources = Vec::new();
    let mut load = |name: &str| {
        let (t, digest) = text(&census.join(name));
        sources.push(format!("# {name} sha256 {digest}"));
        t
    };

    let counties = load(COUNTIES);
    let counties = Table::parse(&counties, '|', &["STATE", "STATEFP", "COUNTYFP", "COUNTYNAME"], COUNTIES);
    let states = state_codes(&counties);
    let zcta_gaz = load(ZCTA_GAZ);
    let zcta_gaz = Table::parse(&zcta_gaz, '\t', &["GEOID", "ALAND", "INTPTLAT", "INTPTLONG"], ZCTA_GAZ);
    let need = ["GEOID_ZCTA5_20", "AREALAND_ZCTA5_20", "AREALAND_PART"];
    let zc = load(ZCTA_COUNTY);
    let zc =
        shares(&Table::parse(&zc, '|', &[&need[..], &["GEOID_COUNTY_20"]].concat(), ZCTA_COUNTY), "GEOID_COUNTY_20");
    let zp = load(ZCTA_PLACE);
    let zp = Table::parse(&zp, '|', &[&need[..], &["GEOID_PLACE_20", "NAMELSAD_PLACE_20"]].concat(), ZCTA_PLACE);
    let place_names: BTreeMap<String, String> =
        zp.rows().map(|r| (r.get("GEOID_PLACE_20").to_string(), r.get("NAMELSAD_PLACE_20").to_string())).collect();
    let zp = shares(&zp, "GEOID_PLACE_20");
    let place_gaz = load(PLACE_GAZ);
    let place_gaz = Table::parse(&place_gaz, '\t', &["USPS", "GEOID", "NAME", "INTPTLAT", "INTPTLONG"], PLACE_GAZ);
    let (boxes, shp_digest, dbf_digest) = shape_boxes(&census.join(SHP), &census.join(DBF), "GEOID20");
    sources.push(format!("# {SHP} sha256 {shp_digest}"));
    sources.push(format!("# {DBF} sha256 {dbf_digest}"));
    let head = sources.join("\n");

    let mut zcta = String::new();
    let mut zip3: BTreeMap<String, BTreeMap<&str, usize>> = BTreeMap::new();
    let mut rows: Vec<_> = zcta_gaz.rows().collect();
    rows.sort_by(|a, b| a.get("GEOID").cmp(b.get("GEOID")));
    let mut missing_box = 0;
    let mut widened = 0;
    let mut moved = 0;
    // The state holding most of each ZCTA's land, and the state holding
    // most of each prefix's ZCTAs by that count.
    let land_state = |id: &str| {
        let county = zc.get(id).map_or(&[][..], Vec::as_slice);
        let Some((top, _)) = county.first() else { panic!("ZCTA {id} has no county") };
        states[&top[..2]]
    };
    let mut by_land: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    for r in &rows {
        let id = r.get("GEOID");
        *by_land.entry(&id[..3]).or_default().entry(land_state(id)).or_default() += 1;
    }
    let majority = |prefix: &str| {
        by_land[prefix].iter().max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0))).map(|(s, _)| *s).unwrap()
    };
    for r in &rows {
        let id = r.get("GEOID");
        let county = zc.get(id).map_or(&[][..], Vec::as_slice);
        // A border ZCTA whose land lies mostly across the line keeps the
        // state of its prefix when that state holds some of its land too
        // (82930, Evanston, is Wyoming's though most of it is in Utah); a
        // ZCTA with no land in its prefix's state keeps its own (96799 is
        // American Samoa's, in prefix 967 of Hawaii).
        let mut state = land_state(id);
        let m = majority(&id[..3]);
        if m != state && county.iter().any(|(c, _)| states[&c[..2]] == m) {
            state = m;
            moved += 1;
        }
        *zip3.entry(id[..3].to_string()).or_default().entry(state).or_default() += 1;
        let (lat, lon) = (r.fixed("INTPTLAT", 4), r.fixed("INTPTLONG", 4));
        let b = boxes.get(id).map(|&b| widen(b, lat, lon, &mut widened));
        missing_box += usize::from(b.is_none());
        let bbox = b.map_or_else(|| "\t\t\t".to_string(), |b| b.map(|v| fixed(v, 3)).join("\t"));
        let land = (r.int("ALAND") + 5000) / 10000;
        let place = zp.get(id).map_or(&[][..], Vec::as_slice);
        writeln!(
            zcta,
            "{id}\t{state}\t{}\t{}\t{}\t{bbox}\t{}\t{}",
            fixed(land, 2),
            fixed(lat, 4),
            fixed(lon, 4),
            list(county),
            list(place)
        )
        .unwrap();
    }
    write(out, "us-zcta.txt", &head, ZCTA_HEADER, &zcta);

    // The Gazetteer leaves out the places of the island areas; their
    // names come from the relationship file, with no point.
    let mut prow: BTreeMap<String, String> = BTreeMap::new();
    for r in place_gaz.rows() {
        let (name, kind) = split_kind(r.get("NAME"));
        let (lat, lon) = (fixed(r.fixed("INTPTLAT", 4), 4), fixed(r.fixed("INTPTLONG", 4), 4));
        prow.insert(r.get("GEOID").to_string(), format!("{}\t{name}\t{kind}\t{lat}\t{lon}", r.get("USPS")));
    }
    let gazetteer_places = prow.len();
    for (geoid, full) in &place_names {
        if !geoid.is_empty() && !prow.contains_key(geoid) {
            let (name, kind) = split_kind(full);
            prow.insert(geoid.clone(), format!("{}\t{name}\t{kind}\t\t", states[&geoid[..2]]));
        }
    }
    let mut places = String::new();
    for (geoid, row) in &prow {
        writeln!(places, "{geoid}\t{row}").unwrap();
    }
    write(out, "us-places.txt", &head, PLACE_HEADER, &places);

    let mut z3 = String::new();
    for (p, s) in &zip3 {
        let s: Vec<String> = s.iter().map(|(st, n)| format!("{st}:{n}")).collect();
        writeln!(z3, "{p}\t{}", s.join(",")).unwrap();
    }
    write(out, "zip3-states.txt", &head, ZIP3_HEADER, &z3);

    let mut county = String::new();
    let mut crow: Vec<_> = counties.rows().collect();
    crow.sort_by(|a, b| (a.get("STATEFP"), a.get("COUNTYFP")).cmp(&(b.get("STATEFP"), b.get("COUNTYFP"))));
    for r in &crow {
        let geoid = format!("{}{}", r.get("STATEFP"), r.get("COUNTYFP"));
        writeln!(county, "{geoid}\t{}\t{}", r.get("STATE"), r.get("COUNTYNAME")).unwrap();
    }
    write(out, "us-counties.txt", &head, COUNTY_HEADER, &county);
    println!(
        "{} ZCTAs ({missing_box} without a box, {widened} boxes widened to their point, {moved} given their prefix's state), {} places ({} from the Gazetteer), {} counties, {} prefixes",
        rows.len(),
        prow.len(),
        gazetteer_places,
        crow.len(),
        zip3.len()
    );
}

const ZCTA_HEADER: &str = "# zcta\tstate\tland_km2\tlat\tlon\twest\tsouth\teast\tnorth\tcounties\tplaces\n\
# state: the USPS code of the state holding most of the ZCTA's land, or of its prefix's main state when that holds some\n\
# lat, lon: the Census internal point; west..north: the box of the 1:500,000 shape, rounded outward and widened to hold the point\n\
# counties, places: county and place GEOIDs with their share of the ZCTA's land in thousandths, largest first";
const PLACE_HEADER: &str = "# geoid\tstate\tname\tkind\tlat\tlon\n\
# kind: the Census legal or statistical area description (city, town, village, CDP, ...)\n\
# lat, lon: the Census internal point; empty for the places of the island areas, which the Gazetteer leaves out";
const COUNTY_HEADER: &str = "# geoid\tstate\tname";
const ZIP3_HEADER: &str = "# prefix\tstates\n\
# states: the USPS code of each state holding ZCTAs of the prefix, with how many";

/// A box in thousandths of a degree grown, when it must be, to hold the
/// point `lat`, `lon` in ten-thousandths: a 2024 internal point can lie
/// just outside the generalized 2020 shape of a small coastal ZCTA.
fn widen([w, s, e, n]: [i64; 4], lat: i64, lon: i64, widened: &mut usize) -> [i64; 4] {
    let (lat_lo, lat_hi) = (lat.div_euclid(10), (lat + 9).div_euclid(10));
    let (lon_lo, lon_hi) = (lon.div_euclid(10), (lon + 9).div_euclid(10));
    let out = [w.min(lon_lo), s.min(lat_lo), e.max(lon_hi), n.max(lat_hi)];
    *widened += usize::from(out != [w, s, e, n]);
    out
}

/// Each ZCTA's parts by `key`, as (GEOID, thousandths of the ZCTA's land),
/// largest first; parts with no land are left out.
fn shares(t: &Table, key: &str) -> HashMap<String, Vec<(String, i64)>> {
    let mut out: HashMap<String, Vec<(String, i64)>> = HashMap::new();
    for r in t.rows() {
        let id = r.get("GEOID_ZCTA5_20");
        let part = r.get("AREALAND_PART");
        if id.is_empty() || r.get(key).is_empty() || part.is_empty() {
            continue;
        }
        let total = r.int("AREALAND_ZCTA5_20");
        if part == "0" && total > 0 {
            continue;
        }
        let total = total.max(1);
        let share = ((r.int("AREALAND_PART") * 1000 + total / 2) / total).max(1);
        out.entry(id.to_string()).or_default().push((r.get(key).to_string(), share));
    }
    for v in out.values_mut() {
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    }
    out
}

fn list(parts: &[(String, i64)]) -> String {
    parts.iter().map(|(g, s)| format!("{g}:{s}")).collect::<Vec<_>>().join(",")
}

/// The USPS code of each state FIPS code.
fn state_codes(t: &Table) -> HashMap<String, &'static str> {
    t.rows().map(|r| (r.get("STATEFP").to_string(), &*r.get("STATE").to_string().leak())).collect()
}

/// A Census place name split from its description: the trailing words
/// that are `CDP`, in brackets or start with a lower case letter
/// (`Springfield city`, `Nashville-Davidson metropolitan government
/// (balance)`, `San Juan zona urbana`).
fn split_kind(name: &str) -> (&str, &str) {
    let mut cut = name.len();
    for (at, word) in name.rmatch_indices(' ').map(|(at, _)| (at, &name[at + 1..])) {
        let word = word.split(' ').next().unwrap_or("");
        let descriptive = word == "CDP" || word.starts_with('(') || word.chars().next().is_some_and(char::is_lowercase);
        if !descriptive {
            break;
        }
        cut = at;
    }
    if cut == 0 || cut == name.len() { (name, "") } else { (&name[..cut], &name[cut + 1..]) }
}

fn write(out: &Path, name: &str, sources: &str, header: &str, body: &str) {
    let note = "# Built by `cargo run --release -p turbo-us-zip --example census_tables` from US Census Bureau files (public domain):";
    std::fs::create_dir_all(out).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
    let all = format!("{note}\n{sources}\n{header}\n{body}");
    std::fs::write(out.join(name), all).unwrap_or_else(|e| panic!("{name}: {e}"));
}
