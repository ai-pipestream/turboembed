//! How the forwarder's counts are told (issue #2513): as Prometheus
//! series on every scrape of a server with a sink, and as one line when
//! the server stops. Before, they were reachable only through
//! `Server::sink_stats`, which the binary never read: a document dropped
//! at a full queue or lost with a failed call left no trace anywhere.

use std::fmt::{self, Write as _};

use super::SinkStats;

impl SinkStats {
    /// Offered or built documents that no sink is known to hold: dropped
    /// at a full queue, refused before it, left in it at a stop, sent on
    /// a call that never acknowledged them, or rejected by the sink.
    pub fn undelivered(&self) -> u64 {
        [self.refused, self.abandoned, self.unconfirmed, self.rejected]
            .into_iter()
            .fold(self.dropped, u64::saturating_add)
    }
}

/// The counts on one line, as the stop prints them.
impl fmt::Display for SinkStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "offered {}, forwarded {}, accepted {}, rejected {}, dropped {}, refused {}, unconfirmed {}, \
             abandoned {}, queued {}, failed calls {}",
            self.offered,
            self.forwarded,
            self.accepted,
            self.rejected,
            self.dropped,
            self.refused,
            self.unconfirmed,
            self.abandoned,
            self.queued,
            self.failed_calls
        )?;
        match &self.last_error {
            Some(e) => write!(f, ", last error: {e}"),
            None => Ok(()),
        }
    }
}

/// Appends the forwarder's series in Prometheus text exposition format.
pub(crate) fn render(out: &mut String, s: &SinkStats) {
    let counters = [
        ("offered", s.offered, "Completed documents offered to the sink queue."),
        ("forwarded", s.forwarded, "Documents sent on a sink call that is open or acknowledged them as received."),
        ("accepted", s.accepted, "Documents the sink's summaries counted accepted."),
        ("rejected", s.rejected, "Documents the sink's summaries counted rejected."),
        ("dropped", s.dropped, "Documents dropped because the sink queue was full."),
        ("refused", s.refused, "Document graphs refused before the sink queue."),
        ("unconfirmed", s.unconfirmed, "Documents sent on a sink call that ended without acknowledging them."),
        ("abandoned", s.abandoned, "Documents left in the sink queue when a stop ended the drain."),
        ("failed_calls", s.failed_calls, "Sink calls that ended in an error."),
    ];
    for (name, value, help) in counters {
        let _ = writeln!(out, "# HELP turbo_kserve_sink_{name}_total {help}");
        let _ = writeln!(out, "# TYPE turbo_kserve_sink_{name}_total counter");
        let _ = writeln!(out, "turbo_kserve_sink_{name}_total {value}");
    }
    let gauges = [
        ("queued_items", s.queued, "Documents waiting in the sink queue."),
        ("queued_bytes", s.queued_bytes, "Encoded bytes waiting in the sink queue."),
    ];
    for (name, value, help) in gauges {
        let _ = writeln!(out, "# HELP turbo_kserve_sink_{name} {help}");
        let _ = writeln!(out, "# TYPE turbo_kserve_sink_{name} gauge");
        let _ = writeln!(out, "turbo_kserve_sink_{name} {value}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stop_line_names_every_count_and_the_last_error() {
        let stats = SinkStats {
            offered: 9,
            forwarded: 2,
            accepted: 1,
            rejected: 1,
            dropped: 3,
            refused: 1,
            unconfirmed: 2,
            abandoned: 2,
            failed_calls: 4,
            last_error: Some("transport error".into()),
            ..Default::default()
        };
        assert_eq!(stats.undelivered(), 9);
        assert_eq!(
            stats.to_string(),
            "offered 9, forwarded 2, accepted 1, rejected 1, dropped 3, refused 1, unconfirmed 2, abandoned 2, \
             queued 0, failed calls 4, last error: transport error"
        );
        assert_eq!(SinkStats::default().undelivered(), 0);
        assert!(!SinkStats::default().to_string().contains("last error"));
    }

    #[test]
    fn the_undelivered_sum_saturates() {
        let stats = SinkStats { rejected: u64::MAX, unconfirmed: 1, dropped: 1, ..Default::default() };
        assert_eq!(stats.undelivered(), u64::MAX);
    }
}
