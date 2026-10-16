//! A trained model as an OpenNLP `.bin` file: `BaseModel.serialize`, a
//! zip archive with a `manifest.properties` entry and the model's
//! artifacts, which `turbo-maxent` opens and OpenNLP's `BaseModel`
//! constructors load.
//!
//! The manifest is what `BaseModel`'s constructor builds and
//! `PropertiesSerializer` writes with `Properties.store(out, "")`: the
//! line `#`, the date, then the entries sorted by key with the escapes of
//! `saveConvert`. The base entries are `Manifest-Version`, `Language`,
//! `OpenNLP-Version`, `Timestamp` and `Component-Name`; a trainer adds
//! `Training-Eventhash` and `TrainerType` ([`crate::gis::TrainingReport`]),
//! and a tool factory adds `factory`, its class name, and its own entries.
//! The version is this port's by default; a caller reproducing a model
//! another OpenNLP wrote sets that version, and the time zone that JVM's
//! clock printed the date in.
//!
//! The JVM writes the entries deflated, in the order of a `HashMap`. The
//! archive written here stores them uncompressed, the manifest first
//! unless the caller places it elsewhere ([`ModelPackage::place_manifest`],
//! for a caller reproducing the JVM's order): the readers on both sides
//! take either method and any order, and a stored entry's bytes are the
//! artifact's bytes, which is what the parity tests compare. Timestamps in the manifest and the zip entries are taken from
//! one instant the caller supplies, in UTC, where the JVM reads its clock
//! and time zone.

use crate::gis::TrainingReport;
use crate::params::save_convert;

/// `Version.currentVersion()` of the OpenNLP this crate is a port of.
pub const OPENNLP_VERSION: &str = "3.0.0-SNAPSHOT";
/// `BaseModel.MANIFEST_ENTRY`.
pub const MANIFEST_ENTRY: &str = "manifest.properties";

/// A fixed offset from UTC with the abbreviation `Date.toString` prints
/// for it: the JVM's default zone at the instant a model was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeZone {
    /// Seconds east of UTC.
    pub offset_seconds: i32,
    /// The short name, `UTC` or `CEST`.
    pub name: &'static str,
}

impl TimeZone {
    pub const UTC: TimeZone = TimeZone { offset_seconds: 0, name: "UTC" };
    /// Central European Summer Time, the zone the UD 1.3 models' manifests
    /// are dated in.
    pub const CEST: TimeZone = TimeZone { offset_seconds: 7200, name: "CEST" };
}

/// The manifest of a model file, typed where `BaseModel` types it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    component_name: String,
    language: String,
    timestamp_ms: i64,
    timestamp: String,
    version: String,
    manifest_version: String,
    zone: TimeZone,
    /// `factory` and the entries of the trainer and of the tool factory,
    /// in the order they were set.
    entries: Vec<(String, String)>,
}

impl Manifest {
    /// `BaseModel(componentName, languageCode, ...)` at `timestamp_ms`
    /// milliseconds since the epoch (`System.currentTimeMillis()`).
    pub fn new(component_name: &str, language: &str, timestamp_ms: i64) -> Manifest {
        Manifest {
            component_name: component_name.to_owned(),
            language: language.to_owned(),
            timestamp_ms,
            timestamp: timestamp_ms.to_string(),
            version: OPENNLP_VERSION.to_owned(),
            manifest_version: String::from("1.0"),
            zone: TimeZone::UTC,
            entries: Vec::new(),
        }
    }

    /// The same manifest with `OpenNLP-Version` set to `version`, for a
    /// model another OpenNLP wrote.
    pub fn with_version(mut self, version: &str) -> Manifest {
        self.version = version.to_owned();
        self
    }

    /// The same manifest dated in `zone`, in its header line and in the
    /// zip entries' times.
    pub fn with_zone(mut self, zone: TimeZone) -> Manifest {
        self.zone = zone;
        self
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn zone(&self) -> TimeZone {
        self.zone
    }

    pub fn component_name(&self) -> &str {
        &self.component_name
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn timestamp_ms(&self) -> i64 {
        self.timestamp_ms
    }

    /// `setManifestProperty`: sets `key`, replacing an earlier value.
    pub fn set(&mut self, key: &str, value: &str) {
        match key {
            "Component-Name" => self.component_name = value.to_owned(),
            "Language" => self.language = value.to_owned(),
            "OpenNLP-Version" => self.version = value.to_owned(),
            "Timestamp" => self.timestamp = value.to_owned(),
            "Manifest-Version" => self.manifest_version = value.to_owned(),
            _ => match self.entries.iter_mut().find(|(k, _)| k == key) {
                Some(entry) => entry.1 = value.to_owned(),
                None => self.entries.push((key.to_owned(), value.to_owned())),
            },
        }
    }

    /// The entries a trainer's report adds (`Training-Eventhash`,
    /// `TrainerType`).
    pub fn set_training(&mut self, report: &TrainingReport) {
        for (key, value) in report.manifest_entries() {
            self.set(key, &value);
        }
    }

    /// `getManifestProperty`: reads a base or custom manifest entry.
    pub fn get(&self, key: &str) -> Option<&str> {
        match key {
            "Component-Name" => Some(&self.component_name),
            "Language" => Some(&self.language),
            "OpenNLP-Version" => Some(&self.version),
            "Timestamp" => Some(&self.timestamp),
            "Manifest-Version" => Some(&self.manifest_version),
            _ => self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str()),
        }
    }

    /// The `manifest.properties` text: `Properties.store(out, "")` of the
    /// entries, which is `#`, then `#` and the date, then the entries
    /// sorted by `String.compareTo` of their keys.
    pub fn to_properties(&self) -> String {
        let mut entries: Vec<(&str, &str)> = vec![
            ("Manifest-Version", &self.manifest_version),
            ("Language", &self.language),
            ("OpenNLP-Version", &self.version),
            ("Timestamp", &self.timestamp),
            ("Component-Name", &self.component_name),
        ];
        entries.extend(self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        entries.sort_by(|a, b| turbo_jvm::chars::compare(a.0, b.0));
        let mut out = String::from("#\n#");
        out.push_str(&java_date_string_in(self.timestamp_ms, self.zone));
        out.push('\n');
        for (k, v) in entries {
            save_convert(k, true, &mut out);
            out.push('=');
            save_convert(v, false, &mut out);
            out.push('\n');
        }
        out
    }
}

/// A model file to write: its manifest and its artifact entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPackage {
    manifest: Manifest,
    entries: Vec<(String, Vec<u8>)>,
    /// How many entries precede the manifest in the archive.
    before_manifest: usize,
}

impl ModelPackage {
    pub fn new(manifest: Manifest) -> ModelPackage {
        ModelPackage { manifest, entries: Vec::new(), before_manifest: 0 }
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn manifest_mut(&mut self) -> &mut Manifest {
        &mut self.manifest
    }

    /// Adds an artifact entry, `name` being its path in the archive
    /// (`sent.model`, `abbreviations.dictionary`).
    pub fn add_entry(&mut self, name: &str, bytes: Vec<u8>) {
        self.entries.push((name.to_owned(), bytes));
    }

    /// The artifact entries in the order they were added, which is their
    /// order in the archive; the manifest is not among them.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &[u8])> {
        self.entries.iter().map(|(n, b)| (n.as_str(), b.as_slice()))
    }

    /// Writes the manifest after the first `before` artifact entries. The
    /// JVM writes `manifest.properties` where its `HashMap` iterates it,
    /// which for a model with a tag dictionary is after the dictionary.
    /// A count beyond the entries puts the manifest last.
    pub fn place_manifest(&mut self, before: usize) {
        self.before_manifest = before;
    }

    /// The names of every entry in archive order, the manifest included.
    pub fn entry_names(&self) -> Vec<&str> {
        let at = self.before_manifest.min(self.entries.len());
        let mut names: Vec<&str> = Vec::with_capacity(self.entries.len() + 1);
        names.extend(self.entries[..at].iter().map(|(n, _)| n.as_str()));
        names.push(MANIFEST_ENTRY);
        names.extend(self.entries[at..].iter().map(|(n, _)| n.as_str()));
        names
    }

    /// `BaseModel.serialize`: the bytes of the `.bin` file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut zip = ZipWriter::new(self.manifest.timestamp_ms, self.manifest.zone);
        let at = self.before_manifest.min(self.entries.len());
        for (name, bytes) in &self.entries[..at] {
            zip.add(name, bytes);
        }
        zip.add(MANIFEST_ENTRY, self.manifest.to_properties().as_bytes());
        for (name, bytes) in &self.entries[at..] {
            zip.add(name, bytes);
        }
        zip.finish()
    }
}

/// What `SentenceModel(languageCode, sentModel, manifestInfoEntries,
/// SentenceDetectorFactory)` puts in a sentence detector model file.
pub const SENTENCE_COMPONENT_NAME: &str = "SentenceDetectorME";
/// `SentenceModel.MAXENT_MODEL_ENTRY_NAME`.
pub const SENTENCE_MODEL_ENTRY: &str = "sent.model";
/// The canonical class name `BaseModel` records as `factory`.
pub const SENTENCE_DETECTOR_FACTORY: &str = "opennlp.tools.sentdetect.SentenceDetectorFactory";

/// A sentence detector model file as `SentenceDetectorME.train` makes it
/// with a `SentenceDetectorFactory` and no abbreviation dictionary: the
/// manifest entries `useTokenEnd`, `eosCharacters` (the characters as one
/// string, when given) and the trainer's, and the `sent.model` entry.
/// `use_token_end` is `SentenceDetectorFactory.isUseTokenEnd`.
pub fn sentence_model_package(
    language: &str,
    timestamp_ms: i64,
    report: &TrainingReport,
    use_token_end: bool,
    eos_characters: Option<&str>,
    model_bytes: Vec<u8>,
) -> ModelPackage {
    let mut manifest = Manifest::new(SENTENCE_COMPONENT_NAME, language, timestamp_ms);
    manifest.set_training(report);
    manifest.set("factory", SENTENCE_DETECTOR_FACTORY);
    manifest.set("useTokenEnd", if use_token_end { "true" } else { "false" });
    if let Some(eos) = eos_characters {
        manifest.set("eosCharacters", eos);
    }
    let mut package = ModelPackage::new(manifest);
    package.add_entry(SENTENCE_MODEL_ENTRY, model_bytes);
    package
}

/// A calendar date and time in UTC.
struct Civil {
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    weekday: usize,
}

/// The UTC civil time of `ms` milliseconds since the epoch.
fn civil(ms: i64) -> Civil {
    let secs = ms.div_euclid(1000);
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400) as u32;
    // Howard Hinnant's days-to-civil algorithm.
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if month <= 2 { y + 1 } else { y };
    // 1970-01-01 was a Thursday; weekdays are counted from Sunday.
    let weekday = (days + 4).rem_euclid(7) as usize;
    Civil { year, month, day, hour: rem / 3600, minute: rem % 3600 / 60, second: rem % 60, weekday }
}

/// `new Date(ms).toString()` in the UTC time zone:
/// `EEE MMM dd HH:mm:ss zzz yyyy`.
pub fn java_date_string(ms: i64) -> String {
    java_date_string_in(ms, TimeZone::UTC)
}

/// `new Date(ms).toString()` on a JVM whose default zone is `zone`.
pub fn java_date_string_in(ms: i64, zone: TimeZone) -> String {
    const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let c = civil(ms.wrapping_add(i64::from(zone.offset_seconds) * 1000));
    format!(
        "{} {} {:02} {:02}:{:02}:{:02} {} {}",
        DAYS[c.weekday],
        MONTHS[c.month as usize - 1],
        c.day,
        c.hour,
        c.minute,
        c.second,
        zone.name,
        c.year
    )
}

/// The MS-DOS time and date fields of a zip entry for `ms`, in `zone`
/// (`ZipEntry.setTime` converts the instant in the default zone).
fn dos_time_date(ms: i64, zone: TimeZone) -> (u16, u16) {
    let c = civil(ms.wrapping_add(i64::from(zone.offset_seconds) * 1000));
    let time = (c.hour << 11) | (c.minute << 5) | (c.second / 2);
    let year = c.year.clamp(1980, 2107) - 1980;
    let date = ((year as u32) << 9) | (c.month << 5) | c.day;
    (time as u16, date as u16)
}

const LOCAL_SIGNATURE: u32 = 0x0403_4b50;
const CENTRAL_SIGNATURE: u32 = 0x0201_4b50;
const END_SIGNATURE: u32 = 0x0605_4b50;
/// "Version made by" and "version needed": 2.0 as `ZipOutputStream`
/// writes, and 1.0 as a stored entry needs.
const VERSION_MADE_BY: u16 = 20;
const VERSION_NEEDED: u16 = 10;
const METHOD_STORED: u16 = 0;

/// A zip archive of stored entries, written to memory.
struct ZipWriter {
    bytes: Vec<u8>,
    central: Vec<u8>,
    count: u16,
    time: u16,
    date: u16,
}

impl ZipWriter {
    fn new(timestamp_ms: i64, zone: TimeZone) -> ZipWriter {
        let (time, date) = dos_time_date(timestamp_ms, zone);
        ZipWriter { bytes: Vec::new(), central: Vec::new(), count: 0, time, date }
    }

    fn add(&mut self, name: &str, data: &[u8]) {
        assert!(self.count < u16::MAX - 1, "a model archive holds fewer than 65535 entries");
        let offset = u32::try_from(self.bytes.len()).expect("a model file is under 4 GB");
        let size = u32::try_from(data.len()).expect("a model entry is under 4 GB");
        let crc = turbo_maxent::crc32(data);
        let name = name.as_bytes();
        let name_len = u16::try_from(name.len()).expect("an entry name is under 64 KB");
        let header = |out: &mut Vec<u8>| {
            out.extend_from_slice(&VERSION_NEEDED.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // general purpose flags
            out.extend_from_slice(&METHOD_STORED.to_le_bytes());
            out.extend_from_slice(&self.time.to_le_bytes());
            out.extend_from_slice(&self.date.to_le_bytes());
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&size.to_le_bytes()); // compressed
            out.extend_from_slice(&size.to_le_bytes()); // uncompressed
            out.extend_from_slice(&name_len.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra field length
        };
        self.bytes.extend_from_slice(&LOCAL_SIGNATURE.to_le_bytes());
        header(&mut self.bytes);
        self.bytes.extend_from_slice(name);
        self.bytes.extend_from_slice(data);

        self.central.extend_from_slice(&CENTRAL_SIGNATURE.to_le_bytes());
        self.central.extend_from_slice(&VERSION_MADE_BY.to_le_bytes());
        header(&mut self.central);
        self.central.extend_from_slice(&0u16.to_le_bytes()); // comment length
        self.central.extend_from_slice(&0u16.to_le_bytes()); // disk number
        self.central.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        self.central.extend_from_slice(&0u32.to_le_bytes()); // external attributes
        self.central.extend_from_slice(&offset.to_le_bytes());
        self.central.extend_from_slice(name);
        self.count += 1;
    }

    fn finish(mut self) -> Vec<u8> {
        let central_offset = u32::try_from(self.bytes.len()).expect("a model file is under 4 GB");
        let central_len = u32::try_from(self.central.len()).expect("a central directory is under 4 GB");
        self.bytes.append(&mut self.central);
        self.bytes.extend_from_slice(&END_SIGNATURE.to_le_bytes());
        self.bytes.extend_from_slice(&0u16.to_le_bytes()); // this disk
        self.bytes.extend_from_slice(&0u16.to_le_bytes()); // central directory disk
        self.bytes.extend_from_slice(&self.count.to_le_bytes());
        self.bytes.extend_from_slice(&self.count.to_le_bytes());
        self.bytes.extend_from_slice(&central_len.to_le_bytes());
        self.bytes.extend_from_slice(&central_offset.to_le_bytes());
        self.bytes.extend_from_slice(&0u16.to_le_bytes()); // comment length
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dos_timestamps_encode_utc_and_cest_instants() {
        let (time, date) = dos_time_date(1791786313615, TimeZone::UTC);
        assert_eq!(time, (6 << 11) | (25 << 5) | 6);
        assert_eq!(date, (46 << 9) | (10 << 5) | 12);
        let (time, date) = dos_time_date(1749698375454, TimeZone::CEST);
        assert_eq!(time, (5 << 11) | (19 << 5) | 17);
        assert_eq!(date, (45 << 9) | (6 << 5) | 12);
    }
}
