//! Language by span (code switching): the language and script of each
//! run of a text, for documents that change language part way.

pub mod annotate;
mod calibrated;
pub mod decode;
pub mod detector;
pub mod gold;
mod script_only;
pub mod scripts;
pub mod units;

pub use annotate::summary;
pub use calibrated::{CalibratedSpanDetector, calibration_identity};
pub use detector::{NgramDetector, Run, Settings, SpanDetector, SpanScratch};
pub use script_only::ScriptDetector;
