//! Numbers in text, ported from the OpenNLP numeric add-on onto the
//! analysis API.
//!
//! - [`scan`] reads one number at a position, as OpenNLP's `NumberScan`
//!   does: ASCII digits, strict grouping in a [`NumberNotation`], an
//!   optional fraction and an optional scale marker, with the exact
//!   decimal value.
//! - [`NumberScanner`] walks a whole text and writes every number it
//!   finds as the `turbo:numbers` layer ([`Numbers`]).
//! - [`money`] finds amounts of money (`$1.2M`, `100 USD`) as the
//!   `turbo:money` layer and converts them with ECB reference rates.
//! - [`QuantityExtractor`] finds measured quantities and percentages
//!   (`80 kg`, `2.5km`, `45%`) as the `turbo:quantities` layer
//!   ([`Quantities`]), leaving out what is money.
//!
//! Every offset is a UTF-8 byte offset into the text, on a character
//! boundary. docs/numbers.md, docs/money.md and docs/quantities.md
//! describe the rules.

pub mod annotator;
pub mod money;
pub mod quantity;
pub mod scan;

pub use annotator::NumberScanner;
pub use money::{
    CONVERTED_MONEY, Currency, Date, Decimal, EcbRates, FxRates, Money, MoneyConverter, MoneyExtractor, ScaleRangeError,
};
pub use quantity::{DEFAULT_UNITS, QuantityExtractor, UnitError};
pub use scan::{ScanResult, parse};
pub use turbo_analysis_layers::{
    DecimalRef, Multiplier, NumberNotation, Numbers, PERCENT, Quantities, QuantityMention, ScaleMarker,
};
