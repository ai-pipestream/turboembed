//! A loaded gazetteer written to bytes and read back without parsing or
//! folding anything: every column and table of the index as a
//! little-endian array, prefixed by its length, and a checksum of all the
//! bytes before it at the end. Reading compares the checksum, which
//! refuses a file damaged on disk or cut short, copies the arrays and
//! checks every offset, code and record field the lookups and accessors
//! follow, so even a file altered with a matching checksum is refused or
//! answers queries without reading out of bounds or panicking. Nothing
//! checks that such a file answers them as the gazetteer it was written
//! from did.
//!
//! The layout: the eight bytes `TURBOGEO`, the format version as a `u32`,
//! the kind of gazetteer as a `u8`, the dataset identifier (for a user
//! table), the arrays in the order [`write_body`] writes them, then the
//! checksum as a `u64` (`checksum.rs`).

use std::fmt;
use std::io::{self, Write};
use std::path::Path;

use crate::checksum::{Summing, checksum};
use crate::entry::{GeoBoundingBox, GeoPoint};
use crate::geonames::GeoNamesGazetteer;
use crate::in_memory::InMemoryGazetteer;
use crate::index::{GazetteerIndex, KeyTable, REGIONS};
use crate::overture::OvertureGazetteer;
use crate::store::{Codes, EntryStore, NONE16, NONE32};
use crate::table::HashSlots;
use crate::text::is_blank;
use crate::user::UserGazetteer;

const MAGIC: &[u8; 8] = b"TURBOGEO";
const VERSION: u32 = 1;

/// Which gazetteer a snapshot holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotKind {
    GeoNames,
    Overture,
    User,
    InMemory,
}

impl SnapshotKind {
    fn byte(self) -> u8 {
        match self {
            SnapshotKind::GeoNames => 1,
            SnapshotKind::Overture => 2,
            SnapshotKind::User => 3,
            SnapshotKind::InMemory => 4,
        }
    }

    fn from_byte(b: u8) -> Option<SnapshotKind> {
        Some(match b {
            1 => SnapshotKind::GeoNames,
            2 => SnapshotKind::Overture,
            3 => SnapshotKind::User,
            4 => SnapshotKind::InMemory,
            _ => return None,
        })
    }
}

/// Why a snapshot cannot be read.
#[derive(Debug)]
pub enum SnapshotError {
    Io(io::Error),
    /// The bytes do not start with the snapshot marker.
    NotASnapshot,
    /// A format version this crate does not read.
    Version(u32),
    /// A snapshot of another kind of gazetteer.
    Kind {
        expected: SnapshotKind,
        found: Option<SnapshotKind>,
    },
    /// The bytes end inside an array.
    Truncated,
    /// An offset, code or table that does not fit the rest; the text says
    /// which.
    Corrupt(&'static str),
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SnapshotError::Io(e) => e.fmt(f),
            SnapshotError::NotASnapshot => f.write_str("not a gazetteer snapshot"),
            SnapshotError::Version(v) => write!(f, "gazetteer snapshot version {v}, this build reads {VERSION}"),
            SnapshotError::Kind { expected, found } => {
                write!(f, "a snapshot of a {found:?} gazetteer, expected {expected:?}")
            }
            SnapshotError::Truncated => f.write_str("the gazetteer snapshot ends early"),
            SnapshotError::Corrupt(what) => write!(f, "corrupt gazetteer snapshot: {what}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<io::Error> for SnapshotError {
    fn from(e: io::Error) -> SnapshotError {
        SnapshotError::Io(e)
    }
}

fn put_len(out: &mut impl Write, n: usize) -> io::Result<()> {
    out.write_all(&(n as u64).to_le_bytes())
}

fn put_bytes(out: &mut impl Write, b: &[u8]) -> io::Result<()> {
    put_len(out, b.len())?;
    out.write_all(b)
}

/// Writes an array of fixed size little-endian values through a buffer.
fn put_array<T: Copy, const N: usize>(out: &mut impl Write, v: &[T], bytes: impl Fn(T) -> [u8; N]) -> io::Result<()> {
    put_len(out, v.len())?;
    let mut buf = Vec::with_capacity(64 * 1024);
    for chunk in v.chunks(64 * 1024 / N) {
        buf.clear();
        for &x in chunk {
            buf.extend_from_slice(&bytes(x));
        }
        out.write_all(&buf)?;
    }
    Ok(())
}

fn put_u32s(out: &mut impl Write, v: &[u32]) -> io::Result<()> {
    put_array(out, v, u32::to_le_bytes)
}

fn put_u16s(out: &mut impl Write, v: &[u16]) -> io::Result<()> {
    put_array(out, v, u16::to_le_bytes)
}

fn put_u64s(out: &mut impl Write, v: &[u64]) -> io::Result<()> {
    put_array(out, v, u64::to_le_bytes)
}

fn put_f64s(out: &mut impl Write, v: &[f64]) -> io::Result<()> {
    put_array(out, v, |x: f64| x.to_bits().to_le_bytes())
}

fn put_slots(out: &mut impl Write, s: &HashSlots) -> io::Result<()> {
    put_u32s(out, &s.values)?;
    put_bytes(out, &s.tags)?;
    put_len(out, s.len)
}

/// Writes the header, the index and the checksum.
fn write_index(out: &mut impl Write, kind: SnapshotKind, source: &str, index: &GazetteerIndex) -> io::Result<()> {
    let mut summing = Summing::new(&mut *out);
    write_body(&mut summing, kind, source, index)?;
    let (sum, out) = summing.finish();
    out.write_all(&sum.to_le_bytes())
}

/// Writes the header and the index.
fn write_body(out: &mut impl Write, kind: SnapshotKind, source: &str, index: &GazetteerIndex) -> io::Result<()> {
    out.write_all(MAGIC)?;
    out.write_all(&VERSION.to_le_bytes())?;
    out.write_all(&[kind.byte()])?;
    put_bytes(out, source.as_bytes())?;
    let s = &index.store;
    put_len(out, s.codes.len())?;
    for name in s.codes.names() {
        put_bytes(out, name.as_bytes())?;
    }
    put_bytes(out, s.text.as_bytes())?;
    put_u32s(out, &s.bounds)?;
    put_f64s(out, &s.latitude)?;
    put_f64s(out, &s.longitude)?;
    put_u64s(out, &s.population)?;
    put_u32s(out, &s.strings)?;
    put_u32s(out, &s.alternates)?;
    put_u32s(out, &s.attributes)?;
    put_u32s(out, &s.box_of)?;
    put_u16s(out, &s.source)?;
    put_u16s(out, &s.feature)?;
    put_array(out, &s.country, |c: [u8; 2]| c)?;
    put_array(out, &s.boxes, |b: [f64; 4]| {
        let mut bytes = [0u8; 32];
        for (i, edge) in b.iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&edge.to_bits().to_le_bytes());
        }
        bytes
    })?;
    put_u32s(out, &s.attribute_key)?;
    put_u32s(out, &s.attribute_value)?;
    put_u32s(out, &s.attribute_notes)?;
    put_u16s(out, &s.attribute_source)?;
    put_bytes(out, index.names.text.as_bytes())?;
    put_u32s(out, &index.names.bounds)?;
    put_slots(out, &index.names.slots)?;
    put_u32s(out, &index.postings_start)?;
    put_u32s(out, &index.postings)?;
    put_slots(out, &index.ids)?;
    put_u32s(out, &index.regions)?;
    put_u16s(out, &index.sources)
}

/// A cursor over the snapshot's bytes.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], SnapshotError> {
        if self.bytes.len() < n {
            return Err(SnapshotError::Truncated);
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    fn len(&mut self) -> Result<usize, SnapshotError> {
        let n = u64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes"));
        usize::try_from(n).map_err(|_| SnapshotError::Truncated)
    }

    fn bytes(&mut self) -> Result<&'a [u8], SnapshotError> {
        let n = self.len()?;
        self.take(n)
    }

    fn string(&mut self) -> Result<String, SnapshotError> {
        let b = self.bytes()?;
        String::from_utf8(b.to_vec()).map_err(|_| SnapshotError::Corrupt("text that is not UTF-8"))
    }

    fn array<T, const N: usize>(&mut self, value: impl Fn([u8; N]) -> T) -> Result<Vec<T>, SnapshotError> {
        let n = self.len()?;
        let bytes = self.take(n.checked_mul(N).ok_or(SnapshotError::Truncated)?)?;
        let (chunks, _) = bytes.as_chunks::<N>();
        Ok(chunks.iter().map(|&c| value(c)).collect())
    }

    fn u32s(&mut self) -> Result<Vec<u32>, SnapshotError> {
        self.array(u32::from_le_bytes)
    }

    fn u16s(&mut self) -> Result<Vec<u16>, SnapshotError> {
        self.array(u16::from_le_bytes)
    }

    fn u64s(&mut self) -> Result<Vec<u64>, SnapshotError> {
        self.array(u64::from_le_bytes)
    }

    fn f64s(&mut self) -> Result<Vec<f64>, SnapshotError> {
        self.array(|b| f64::from_bits(u64::from_le_bytes(b)))
    }

    fn slots(&mut self) -> Result<HashSlots, SnapshotError> {
        Ok(HashSlots { values: self.u32s()?, tags: self.bytes()?.to_vec(), len: self.len()? })
    }
}

/// Reads the header and the index, checking every array against the
/// others.
fn read_index(bytes: &[u8], expected: SnapshotKind) -> Result<(String, GazetteerIndex), SnapshotError> {
    let mut r = Reader { bytes };
    if r.take(8).ok() != Some(MAGIC.as_slice()) {
        return Err(SnapshotError::NotASnapshot);
    }
    let version = u32::from_le_bytes(r.take(4)?.try_into().expect("four bytes"));
    if version != VERSION {
        return Err(SnapshotError::Version(version));
    }
    let found = SnapshotKind::from_byte(r.take(1)?[0]);
    if found != Some(expected) {
        return Err(SnapshotError::Kind { expected, found });
    }
    let Some(body_len) = bytes.len().checked_sub(8) else { return Err(SnapshotError::Truncated) };
    let (body, sum) = bytes.split_at(body_len);
    if body_len < MAGIC.len() + 5 {
        return Err(SnapshotError::Truncated);
    }
    if checksum(body) != u64::from_le_bytes(sum.try_into().expect("eight bytes")) {
        return Err(SnapshotError::Corrupt("the checksum does not match"));
    }
    r.bytes = &r.bytes[..r.bytes.len() - 8];
    let source = r.string()?;
    let code_count = r.len()?;
    let mut names = Vec::with_capacity(code_count.min(1 << 16));
    for _ in 0..code_count {
        names.push(r.string()?.into_boxed_str());
    }
    let store = EntryStore {
        codes: Codes::from_names(names),
        text: r.string()?,
        bounds: r.u32s()?,
        latitude: r.f64s()?,
        longitude: r.f64s()?,
        population: r.u64s()?,
        strings: r.u32s()?,
        alternates: r.u32s()?,
        attributes: r.u32s()?,
        box_of: r.u32s()?,
        source: r.u16s()?,
        feature: r.u16s()?,
        country: r.array(|b: [u8; 2]| b)?,
        boxes: r.array(|b: [u8; 32]| {
            let (words, _) = b.as_chunks::<8>();
            [0, 1, 2, 3].map(|i| f64::from_bits(u64::from_le_bytes(words[i])))
        })?,
        attribute_key: r.u32s()?,
        attribute_value: r.u32s()?,
        attribute_notes: r.u32s()?,
        attribute_source: r.u16s()?,
    };
    let names = KeyTable { text: r.string()?, bounds: r.u32s()?, slots: r.slots()? };
    let index = GazetteerIndex {
        store,
        names,
        postings_start: r.u32s()?,
        postings: r.u32s()?,
        ids: r.slots()?,
        regions: r.u32s()?,
        sources: r.u16s()?,
    };
    if !r.bytes.is_empty() {
        return Err(SnapshotError::Corrupt("bytes after the last array"));
    }
    check(&index)?;
    Ok((source, index))
}

fn ensure(ok: bool, what: &'static str) -> Result<(), SnapshotError> {
    if ok { Ok(()) } else { Err(SnapshotError::Corrupt(what)) }
}

/// Whether `bounds` cut `text` into strings: from 0, never decreasing, to
/// the end, each on a character boundary.
fn cuts(text: &str, bounds: &[u32]) -> bool {
    bounds.first() == Some(&0)
        && bounds.windows(2).all(|w| w[0] <= w[1])
        && bounds.last().is_some_and(|&b| b as usize == text.len())
        && bounds.iter().all(|&b| text.is_char_boundary(b as usize))
}

/// Whether `starts` is a list of ranges into `0..total` from 0, never
/// decreasing, ending at `total`.
fn ranges(starts: &[u32], total: usize) -> bool {
    starts.first() == Some(&0)
        && starts.windows(2).all(|w| w[0] <= w[1])
        && starts.last().is_some_and(|&s| s as usize == total)
}

/// Checks everything an accessor or lookup follows without its own check.
fn check(index: &GazetteerIndex) -> Result<(), SnapshotError> {
    let s = &index.store;
    let n = s.latitude.len();
    let strings = s.bounds.len().saturating_sub(1);
    let codes = s.codes.len();
    let attributes = s.attribute_key.len();
    ensure(cuts(&s.text, &s.bounds), "string offsets")?;
    ensure(
        [s.longitude.len(), s.population.len(), s.alternates.len(), s.box_of.len(), s.source.len()]
            .into_iter()
            .chain([s.feature.len(), s.country.len()])
            .all(|l| l == n)
            && s.strings.len() == n + 1
            && s.attributes.len() == n + 1,
        "column lengths",
    )?;
    ensure(ranges(&s.strings, strings), "record strings")?;
    ensure(ranges(&s.attributes, attributes), "record attributes")?;
    ensure(
        (0..n).all(|i| {
            let own = u64::from(s.strings[i + 1] - s.strings[i]);
            own >= 2 + u64::from(s.alternates[i]) + 3 * u64::from(s.attributes[i + 1] - s.attributes[i])
        }),
        "string counts",
    )?;
    ensure(
        [s.attribute_value.len(), s.attribute_notes.len(), s.attribute_source.len()].iter().all(|&l| l == attributes),
        "attribute lengths",
    )?;
    ensure(
        s.attribute_key.iter().chain(&s.attribute_value).chain(&s.attribute_notes).all(|&i| (i as usize) < strings),
        "attribute strings",
    )?;
    ensure(s.attribute_source.iter().all(|&c| usize::from(c) < codes), "attribute sources")?;
    ensure(s.codes.names().iter().all(|name| !is_blank(name)), "blank sources or feature classes")?;
    ensure(
        s.attribute_key.iter().chain(&s.attribute_value).all(|&i| !is_blank(s.string(i))),
        "blank attribute keys or values",
    )?;
    ensure((0..n).all(|i| record_strings_hold(s, i)), "blank record strings or repeated attribute keys")?;
    ensure(s.source.iter().all(|&c| usize::from(c) < codes), "sources")?;
    ensure(s.feature.iter().all(|&c| c == NONE16 || usize::from(c) < codes), "feature classes")?;
    ensure(s.country.iter().all(|c| *c == [0, 0] || c.iter().all(u8::is_ascii_uppercase)), "country codes")?;
    ensure(s.box_of.iter().all(|&b| b == NONE32 || (b as usize) < s.boxes.len()), "box indices")?;
    ensure(s.population.iter().all(|&p| p <= i64::MAX as u64), "populations")?;
    ensure((0..n).all(|i| GeoPoint::new(s.latitude[i], s.longitude[i]).is_ok()), "locations")?;
    ensure(s.boxes.iter().all(|&[w, so, e, no]| GeoBoundingBox::new(w, so, e, no).is_ok()), "boxes")?;
    let keys = index.names.bounds.len().saturating_sub(1);
    ensure(cuts(&index.names.text, &index.names.bounds), "key offsets")?;
    ensure(index.names.slots.is_well_formed() && index.names.slots.len == keys, "key table")?;
    ensure(slot_values_below(&index.names.slots, keys), "key table values")?;
    ensure(index.postings_start.len() == keys + 1 && ranges(&index.postings_start, index.postings.len()), "postings")?;
    ensure(index.postings.iter().all(|&r| (r as usize) < n), "posted records")?;
    ensure(index.ids.is_well_formed() && index.ids.len == n, "id table")?;
    ensure(slot_values_below(&index.ids, n), "id table values")?;
    ensure(
        index.regions.len() == REGIONS && index.regions.iter().all(|&r| r == NONE32 || (r as usize) < n),
        "regions",
    )?;
    ensure(index.sources.iter().all(|&c| usize::from(c) < codes), "source list")
}

/// Whether record `i` has what `GazetteerEntry` requires of the strings it
/// stores: a record id, name, alternate names and containment that are not
/// blank, and no attribute key twice. The offsets are checked already.
fn record_strings_hold(s: &EntryStore, i: usize) -> bool {
    let attributes = s.attributes[i] as usize..s.attributes[i + 1] as usize;
    let end = s.strings[i + 1] - 3 * attributes.len() as u32;
    if !(s.strings[i]..end).all(|k| !is_blank(s.string(k))) {
        return false;
    }
    let keys = &s.attribute_key[attributes];
    if keys.len() < 2 {
        return true;
    }
    let mut seen = std::collections::HashSet::with_capacity(keys.len());
    keys.iter().all(|&k| seen.insert(s.string(k)))
}

fn slot_values_below(slots: &HashSlots, n: usize) -> bool {
    slots.values.iter().zip(&slots.tags).all(|(&v, &t)| t == 0 || (v as usize) < n)
}

macro_rules! snapshot_methods {
    ($ty:ty, $kind:expr, |$this:ident| $source:expr, |$src:ident, $index:ident| $make:expr) => {
        impl $ty {
            /// Writes the loaded gazetteer as a snapshot, which
            /// `from_snapshot` reads back without parsing the table again.
            pub fn write_snapshot(&self, out: &mut impl Write) -> io::Result<()> {
                let $this = self;
                write_index(out, $kind, $source, self.index())
            }

            /// Reads a snapshot `write_snapshot` wrote.
            pub fn from_snapshot(bytes: &[u8]) -> Result<Self, SnapshotError> {
                let ($src, $index) = read_index(bytes, $kind)?;
                Ok($make)
            }

            /// Reads the snapshot in the file at `path`.
            pub fn load_snapshot(path: impl AsRef<Path>) -> Result<Self, SnapshotError> {
                Self::from_snapshot(&std::fs::read(path)?)
            }
        }
    };
}

snapshot_methods!(GeoNamesGazetteer, SnapshotKind::GeoNames, |_g| "", |_s, index| GeoNamesGazetteer::from_index(index));
snapshot_methods!(OvertureGazetteer, SnapshotKind::Overture, |_g| "", |_s, index| OvertureGazetteer::from_index(index));
snapshot_methods!(InMemoryGazetteer, SnapshotKind::InMemory, |_g| "", |_s, index| InMemoryGazetteer::from_index(index));
snapshot_methods!(UserGazetteer, SnapshotKind::User, |g| g.source(), |source, index| {
    if crate::text::is_blank(&source) {
        return Err(SnapshotError::Corrupt("a user table's source"));
    }
    UserGazetteer::from_index(source, index)
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::RegionCode;
    use crate::gazetteer::Gazetteer;

    const TABLE: &str = "q1\tQ\tAlt One|Alt Two\t1\t2\tDE\tCITY\t10\t\tBY\tka=1\tkb=2\n\
                         q2\tBerlin\t\t52.5\t13.4\tDE\tCITY\t3000000\n";

    fn snapshot() -> Vec<u8> {
        let g = UserGazetteer::from_bytes(TABLE.as_bytes(), "sites").unwrap();
        let mut bytes = Vec::new();
        g.write_snapshot(&mut bytes).unwrap();
        bytes
    }

    /// `bytes` with its checksum written again, as an altered file whose
    /// author fixed the checksum would be.
    fn resum(mut bytes: Vec<u8>) -> Vec<u8> {
        let body = bytes.len() - 8;
        let sum = checksum(&bytes[..body]);
        bytes[body..].copy_from_slice(&sum.to_le_bytes());
        bytes
    }

    /// Replaces the only occurrence of `from` in `bytes` by `to`.
    fn replace(bytes: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
        let at = bytes.windows(from.len()).position(|w| w == from).unwrap();
        assert!(bytes[at + 1..].windows(from.len()).all(|w| w != from), "{from:?} occurs once");
        let mut out = bytes.to_vec();
        out[at..at + from.len()].copy_from_slice(to);
        out
    }

    #[test]
    fn a_damaged_byte_fails_the_checksum() {
        let bytes = snapshot();
        assert!(UserGazetteer::from_snapshot(&bytes).is_ok());
        let mut damaged = bytes.clone();
        damaged[bytes.len() / 2] ^= 1;
        assert!(matches!(UserGazetteer::from_snapshot(&damaged), Err(SnapshotError::Corrupt(_))));
        assert!(matches!(UserGazetteer::from_snapshot(&bytes[..20]), Err(SnapshotError::Truncated)));
    }

    #[test]
    fn records_a_gazetteer_entry_could_not_hold_are_refused() {
        let bytes = snapshot();
        let blank_name = resum(replace(&bytes, b"q1Q", b"q1 "));
        assert!(matches!(
            UserGazetteer::from_snapshot(&blank_name),
            Err(SnapshotError::Corrupt("blank record strings or repeated attribute keys"))
        ));
        let blank_alternate = resum(replace(&bytes, b"Alt Two", b"       "));
        assert!(UserGazetteer::from_snapshot(&blank_alternate).is_err());
        let repeated_key = resum(replace(&bytes, b"kb", b"ka"));
        assert!(matches!(
            UserGazetteer::from_snapshot(&repeated_key),
            Err(SnapshotError::Corrupt("blank record strings or repeated attribute keys"))
        ));
        let blank_value = resum(replace(&bytes, b"ka1", b"ka "));
        assert!(matches!(
            UserGazetteer::from_snapshot(&blank_value),
            Err(SnapshotError::Corrupt("blank attribute keys or values"))
        ));
        let blank_class = resum(replace(&bytes, b"CITY", b"    "));
        assert!(matches!(
            UserGazetteer::from_snapshot(&blank_class),
            Err(SnapshotError::Corrupt("blank sources or feature classes"))
        ));
    }

    /// Every single byte change with a matching checksum is refused or
    /// gives a gazetteer whose queries and records read without panicking.
    #[test]
    fn altered_bytes_with_a_matching_checksum_never_panic() {
        let bytes = snapshot();
        for i in 13..bytes.len() - 8 {
            for flip in [0x01, 0x5A, 0xFF] {
                let mut altered = bytes.clone();
                altered[i] ^= flip;
                let Ok(g) = UserGazetteer::from_snapshot(&resum(altered)) else { continue };
                for q in ["Q", "alt one", "Berlin", "nowhere"] {
                    for e in g.lookup(q) {
                        let _ = e.to_entry();
                    }
                }
                for id in ["q1", "q2"] {
                    let _ = g.by_id("sites", id).map(|e| e.to_entry());
                }
                let _ = g.by_region(RegionCode::parse("DE").unwrap()).map(|e| e.to_entry());
            }
        }
    }
}
