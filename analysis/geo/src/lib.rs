//! OpenNLP's gazetteer API and its gazetteers (the `geo` module of the
//! OpenNLP addons): place records looked up by name, by source and record
//! id, and by country.
//!
//! - [`Gazetteer`] is the interface, [`Entry`] the view of one record and
//!   [`GazetteerEntry`] an owned record.
//! - [`BundledGazetteer`] is Natural Earth's 7342 populated places, which
//!   ship in this crate (public domain).
//! - [`GeoNamesGazetteer`] loads a GeoNames main table file and
//!   [`OvertureGazetteer`] a division table derived from Overture Maps;
//!   both are the caller's files (CC BY 4.0 and ODbL), never bundled.
//! - [`UserGazetteer`] loads a hand written table and
//!   [`InMemoryGazetteer`] indexes records built in code.
//! - [`OverlayGazetteer`] lays additions and [`Suppression`] rules over a
//!   base gazetteer.
//!
//! Every gazetteer matches by one rule. A name is cut into its UAX #29
//! word tokens, each token is put in NFC, lowercased with
//! `toLowerCase(Locale.ROOT)` and stripped of the accents of Latin, Greek
//! and Cyrillic letters, and the tokens are joined by one space: the
//! match key ([`Folder`]). A lookup folds the query and returns the
//! records with a name of the same key, ranked by population descending,
//! then `CITY` before `ADMIN` before `POI` before anything else, then
//! source and record id.
//!
//! # The index
//!
//! The records are stored by column: all their strings in one text buffer
//! addressed by 32-bit offsets, the numbers in a vector each, and sources
//! and feature classes as 16-bit codes, so a record costs its text and
//! about fifty bytes. Each distinct match key is stored once; an open
//! addressing table with a tag byte per slot finds it, and a posting list
//! gives its records, ranked when the index is built. A second table
//! finds a record by source and record id, and a 676 entry array holds
//! each country's representative. A table larger than a few megabytes is
//! parsed and folded on every core at once, and the posting lists are
//! ranked in parallel.
//!
//! It follows the `geo` module at commit
//! `abaada5e21c3177d911233149f5c7095a6266fb4` of the OpenNLP addons fork
//! (the module is the same at `33e95a4956f1535cdc074056b29246fe88b3c81a`,
//! the addons commit the goldens are built from), run against the runtime
//! of commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the OpenNLP fork
//! on JDK 21; `README.md` describes the JVM golden files the tests compare
//! with. A table that does not load gives a [`LoadError`] whose message is
//! OpenNLP's, line number included.
//!
//! # What OpenNLP does that may surprise
//!
//! - **Matching is by whole name, word by word.** Punctuation and spacing
//!   vanish (`Saint-Louis`, `Saint Louis` and `SAINT  LOUIS` are one
//!   key), but a query is never matched by a prefix or a token subset:
//!   `New York City` does not find `New York`.
//! - **Each Han ideograph is a word.** UAX #29 puts a break between
//!   ideographs, so `東京` folds to `東 京`; a query in the same script
//!   folds the same way and still matches.
//! - **Marks go only from Latin, Greek and Cyrillic letters.** Devanagari
//!   vowel signs, Arabic and Hebrew points stay. A fixed list of nineteen
//!   letters that have no decomposition is mapped to ASCII as well: `ß`
//!   becomes `ss`, `ø` becomes `o`, `æ` becomes `ae`, the dotless `ı`
//!   becomes `i`.
//! - **An empty key is never a match.** A query with no word token finds
//!   nothing, a record whose canonical name has none is refused, and an
//!   alternate name with none is refused in hand written tables and in
//!   records built in code but silently left unindexed in GeoNames and
//!   Overture tables.
//! - **`by_region` takes either case; record country codes do not.** A
//!   query code is two ASCII letters of either case, but a record's code
//!   must be two capital letters or the record is refused.
//! - **An overlay's region representative ignores the feature class.** It
//!   compares the two sides' representatives by population alone, the
//!   additions winning a tie, and when a rule hides the base's
//!   representative the base offers no other.
//! - **Unicode version.** The folding uses `turbo-unicode`'s tables, newer
//!   than JDK 21's Unicode 15.0; a name with characters assigned after
//!   15.0 can fold differently.

mod bundled;
mod entry;
mod error;
mod fold;
mod gazetteer;
mod geonames;
mod in_memory;
mod index;
mod load;
mod overlay;
mod overture;
mod ranking;
mod snapshot;
mod store;
mod suppression;
mod table;
mod text;
mod user;

pub use bundled::{BUNDLED_TABLE, BUNDLED_TABLE_NAME, BundledGazetteer};
pub use entry::{
    ATTRIBUTE_KEY_FIPS, ATTRIBUTE_KEY_GEOID, ATTRIBUTE_KEY_GEONAMES, ATTRIBUTE_KEY_WHOSONFIRST, ATTRIBUTE_KEY_WIKIDATA,
    ATTRIBUTE_KEY_ZCTA, AttributeValue, EntryError, EntryParts, FEATURE_CLASS_ADMIN, FEATURE_CLASS_CITY,
    FEATURE_CLASS_POI, GazetteerEntry, GeoBoundingBox, GeoPoint, InvalidRegionCode, RegionCode, SOURCE_UNSPECIFIED,
};
pub use error::{ColumnCount, IndexError, LoadError, NumberFormat, RowProblem, TableKind};
pub use fold::Folder;
pub use gazetteer::Gazetteer;
pub use geonames::{GEONAMES_SOURCE, GeoNamesGazetteer};
pub use in_memory::InMemoryGazetteer;
pub use overlay::{EmptyOverlay, OverlayGazetteer};
pub use overture::{OVERTURE_SOURCE, OvertureGazetteer};
pub use snapshot::{SnapshotError, SnapshotKind};
pub use store::{Attribute, CapacityError, Entry};
pub use suppression::{Suppression, SuppressionError};
pub use user::UserGazetteer;
