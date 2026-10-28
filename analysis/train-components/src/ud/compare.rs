/// Whether a trained value agrees with the published one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Agreement<T> {
    Same(T),
    Differ { published: T, trained: T },
}

impl<T> Agreement<T> {
    pub fn is_same(&self) -> bool {
        matches!(self, Agreement::Same(_))
    }
}

/// A model entry's size and SHA-256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryDigest {
    pub len: usize,
    pub sha256: String,
}

/// How far the parameters of two models of the same predicates and
/// patterns are apart.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterDistance {
    pub total: usize,
    /// How many parameters are not bit for bit the published ones.
    pub differing: usize,
    /// The widest gap in units in the last place.
    pub max_ulps: u64,
    /// The widest gap as a number.
    pub max_abs: f64,
}

/// A trained model against the published one: the event checksum, the
/// model entry, the manifest text and the zip entries. The zip container
/// itself is not compared: the JVM deflates its entries and this port
/// stores them.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub model: String,
    pub events: usize,
    pub event_hash: Agreement<u32>,
    pub model_entry: Agreement<EntryDigest>,
    /// The first byte at which the model entries differ, when they do.
    pub first_difference: Option<usize>,
    /// When the entries differ: whether the published model has the same
    /// predicates with the same outcome patterns, and if so how far the
    /// parameters are apart. `None` when the entries agree.
    pub parameters: Option<std::result::Result<ParameterDistance, String>>,
    pub manifest: Agreement<String>,
    pub entry_names: Agreement<Vec<String>>,
}

impl Comparison {
    /// Whether every compared part agrees, the model entry byte for byte.
    pub fn reproduced(&self) -> bool {
        self.event_hash.is_same() && self.model_entry.is_same() && self.manifest.is_same() && self.entry_names.is_same()
    }

    /// Whether everything but the parameters' last bits agrees: the same
    /// events, manifest and zip entries, the same predicates and patterns,
    /// and no parameter further than `max_abs` from the published one. The
    /// reproduction gate uses exact [`reproduced`](Self::reproduced)
    /// agreement. This method remains useful for diagnosing a run with
    /// different settings, such as its number of training partitions.
    pub fn agrees_within(&self, max_abs: f64) -> bool {
        let parameters = match &self.parameters {
            None => self.model_entry.is_same() && max_abs >= 0.0,
            // A NaN distance (a NaN parameter on either side) is no
            // agreement; the comparison is false for it.
            Some(Ok(d)) => d.max_abs <= max_abs,
            Some(Err(_)) => false,
        };
        self.event_hash.is_same() && self.manifest.is_same() && self.entry_names.is_same() && parameters
    }

    /// A few lines for a report.
    pub fn report(&self) -> String {
        let mut s = format!("{}: {} events\n", self.model, self.events);
        match &self.event_hash {
            Agreement::Same(h) => s.push_str(&format!("  event hash {h}: same\n")),
            Agreement::Differ { published, trained } => {
                s.push_str(&format!("  event hash: published {published}, trained {trained}\n"));
            }
        }
        match &self.model_entry {
            Agreement::Same(d) => s.push_str(&format!("  model entry {} bytes {}: same\n", d.len, d.sha256)),
            Agreement::Differ { published, trained } => s.push_str(&format!(
                "  model entry: published {} bytes {}, trained {} bytes {}, first difference at byte {}\n",
                published.len,
                published.sha256,
                trained.len,
                trained.sha256,
                self.first_difference.map_or("none".to_owned(), |i| i.to_string())
            )),
        }
        match &self.parameters {
            None => {}
            Some(Ok(d)) => s.push_str(&format!(
                "  same predicates and patterns; {} of {} parameters differ, at most {} ulps, {:e} in value\n",
                d.differing, d.total, d.max_ulps, d.max_abs
            )),
            Some(Err(why)) => s.push_str(&format!("  parameters not comparable: {why}\n")),
        }
        match &self.manifest {
            Agreement::Same(_) => s.push_str("  manifest: same\n"),
            Agreement::Differ { published, trained } => {
                s.push_str("  manifest differs:\n");
                // Every line of either, so extra lines on one side show.
                let (mut p, mut t) = (published.lines(), trained.lines());
                loop {
                    match (p.next(), t.next()) {
                        (None, None) => break,
                        (a, b) if a == b => {}
                        (a, b) => s.push_str(&format!(
                            "    published: {}\n    trained:   {}\n",
                            a.unwrap_or("(no line)"),
                            b.unwrap_or("(no line)")
                        )),
                    }
                }
            }
        }
        match &self.entry_names {
            Agreement::Same(_) => s.push_str("  zip entries: same names in the same order\n"),
            Agreement::Differ { published, trained } => {
                s.push_str(&format!("  zip entries: published {published:?}, trained {trained:?}\n"));
            }
        }
        s
    }
}

#[cfg(test)]
pub(super) fn update_parameter_distance(d: &mut ParameterDistance, a: f64, b: f64) {
    d.total += 1;
    if a.to_bits() != b.to_bits() || a.is_nan() || b.is_nan() {
        d.differing += 1;
        if !a.is_nan() && !b.is_nan() {
            d.max_ulps = d.max_ulps.max(ulps_apart(a, b));
        }
        let gap = (a - b).abs();
        // Not `f64::max`, which would drop a NaN gap: a NaN wins
        // and stays, so `agrees_within` sees it.
        if gap.is_nan() || d.max_abs.is_nan() || gap > d.max_abs {
            d.max_abs = if d.max_abs.is_nan() { d.max_abs } else { gap };
        }
    }
}

/// How many representable numbers lie between `a` and `b`: the distance
/// of their bit patterns once each is mapped from sign and magnitude to a
/// number line, so that two values on either side of zero are as far
/// apart as the numbers between them and not half the bit space.
#[cfg(test)]
pub(super) fn ulps_apart(a: f64, b: f64) -> u64 {
    fn ordered(x: f64) -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    }
    ordered(a).abs_diff(ordered(b))
}
