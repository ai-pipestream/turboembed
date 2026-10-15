//! Time expressions: what a text states ([`spec`]), durations
//! ([`duration`]), calendar arithmetic ([`calendar`]), resolution against
//! a reference date-time ([`resolve`]) and TIMEX3 values ([`timex`]).

pub mod calendar;
pub mod duration;
pub mod resolve;
pub mod spec;
pub mod timex;

pub use duration::{Duration, Unit};
pub use resolve::{Alternative, Instant, Resolution, Resolved, depends_on_reference, resolve};
pub use spec::TimeValue;
pub use timex::Timex;
