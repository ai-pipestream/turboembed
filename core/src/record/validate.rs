use super::*;
impl Record {
    /// Parse a record read from `name`, and check that it is well formed:
    /// its name is the one its contents give, every hash and number is
    /// one a measurement could have produced, each reference was either
    /// measured or says why not, speed_ratio is what the references
    /// give, and no text in it is a path under a home directory.
    pub fn parse(name: &str, bytes: &[u8]) -> Result<Record, String> {
        let r: Record = serde_json::from_slice(bytes).map_err(|e| format!("{name}: {e}"))?;
        r.check().map_err(|e| format!("{name}: {e}"))?;
        let want = file_name(&r).map_err(|e| format!("{name}: {e}"))?;
        if name != want {
            return Err(format!("{name}: its contents name it {want}"));
        }
        Ok(r)
    }

    pub fn check(&self) -> Result<(), String> {
        if self.record_version != RECORD_VERSION {
            return Err(format!("record_version {} is not {RECORD_VERSION}", self.record_version));
        }
        let t = self.recorded_at.as_bytes();
        let shape = "dddd-dd-ddTdd:dd:ddZ".as_bytes();
        if t.len() != shape.len()
            || !t.iter().zip(shape).all(|(c, s)| if *s == b'd' { c.is_ascii_digit() } else { c == s })
        {
            return Err(format!("recorded_at {:?} is not YYYY-MM-DDTHH:MM:SSZ", self.recorded_at));
        }
        if self.machine.arch.is_empty() || self.device.backend.is_empty() {
            return Err("machine.arch and device.backend must not be empty".into());
        }
        if !["DEVICE_CPU", "DEVICE_GPU", "DEVICE_IGPU", "DEVICE_NPU"].contains(&self.device.kind.as_str()) {
            return Err(format!("device.kind {:?} is not a DEVICE_* value", self.device.kind));
        }
        if !is_hex(&self.library.commit, 40) {
            return Err(format!("library.commit {:?} is not 40 lowercase hex", self.library.commit));
        }
        if self.library.pushed_to.is_empty() {
            return Err("library.pushed_to is empty: the commit was on no branch of origin".into());
        }
        if self.library.version.is_empty() || !self.library.build.starts_with(&self.library.version) {
            return Err(format!(
                "library.build {:?} does not start with {:?}",
                self.library.build, self.library.version
            ));
        }
        let vars: Vec<&str> = LIBRARY_VARS.iter().filter(|(b, _)| *b == self.device.backend).map(|&(_, v)| v).collect();
        let mut at = 0;
        for s in &self.library.settings {
            let name = s.split_once('=').map(|(n, _)| n);
            match name.and_then(|n| vars[at..].iter().position(|v| *v == n)) {
                Some(i) => at += i + 1,
                None => {
                    return Err(format!(
                        "library.settings {s:?} is not NAME=value, NAME one of {vars:?} for {}, each once and in \
                         that order",
                        self.device.backend
                    ));
                }
            }
        }
        if task_name(TURBO_TASK_EMBED) != Some(self.task.as_str()) {
            return Err(format!("task {:?} is not TASK_EMBED", self.task));
        }
        if ![TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT]
            .iter()
            .any(|&p| precision_name(p) == Some(self.precision.as_str()))
        {
            return Err(format!("precision {:?} is not a PRECISION_* value", self.precision));
        }
        if dtype_value(&self.compute_dtype).is_none() {
            return Err(format!("compute_dtype {:?} is not a DTYPE_* value", self.compute_dtype));
        }
        let b = &self.bundle;
        for (what, h) in
            [("manifest", &b.manifest_sha256), ("artifact", &b.artifact_sha256), ("tokenizer", &b.tokenizer_sha256)]
        {
            if !is_hex(h, 64) {
                return Err(format!("bundle.{what}_sha256 {h:?} is not 64 lowercase hex"));
            }
        }
        if b.model_id.is_empty() {
            return Err("bundle.model_id is empty".into());
        }
        let rows = &self.rows;
        let n = pass_rows(rows)?;
        if rows.live_tokens < n || rows.live_tokens > n * u64::from(rows.seq) {
            return Err(format!("rows.live_tokens {} does not fit {n} x {}", rows.live_tokens, rows.seq));
        }
        match rows.kind.as_str() {
            ROWS_MIXED => {}
            ROWS_DENSE if rows.live_tokens == rows.batch as u64 * rows.seq as u64 => {}
            ROWS_DENSE => {
                return Err(format!(
                    "rows: dense, yet {} of {} tokens are live",
                    rows.live_tokens,
                    rows.batch as u64 * rows.seq as u64
                ));
            }
            k => return Err(format!("rows.kind {k:?} is not {ROWS_MIXED} or {ROWS_DENSE}")),
        }
        // One frame is batch x seq. A batch-1 cycle is one [1, seq]
        // frame per case, and the pass computes every one of them.
        let positions = n * u64::from(rows.seq);
        let slots = rows.live_tokens..=positions;
        if let Some(got) = self.timing.computed_tokens
            && !slots.contains(&got)
        {
            return Err(format!(
                "timing.computed_tokens {got} is not between the live tokens and the padded pass, {slots:?}"
            ));
        }
        // The npu and Hailo graphs execute every position of each compiled frame,
        // including padding. The count is that frame times the frames in
        // the pass: batch x seq, or one [1, seq] frame per case.
        if matches!(self.device.backend.as_str(), "npu" | "hailo") && self.timing.computed_tokens != Some(positions) {
            return Err(
                "timing.computed_tokens for npu or hailo is every position of each compiled frame in the pass: batch x seq \
                 for one frame, and one [1, seq] frame per case when the frame batch is 1"
                    .into(),
            );
        }
        npu_record_settings(self)?;
        let t = &self.timing;
        if t.iterations == 0
            || ![t.p50_ms, t.p99_ms, t.mean_ms, t.min_ms, t.max_ms, t.rows_per_second].into_iter().all(positive)
            || !(t.min_ms <= t.p50_ms && t.p50_ms <= t.p99_ms && t.p99_ms <= t.max_ms)
            || !(t.min_ms..=t.max_ms).contains(&t.mean_ms)
        {
            return Err("timing: iterations above 0, every figure finite and above 0, min <= p50 <= p99 <= max and min <= mean <= max".into());
        }
        let c = &self.conformance;
        if c.rows == 0
            || !(c.min_cosine.is_finite() && (-1.0..=1.0 + 1e-6).contains(&c.min_cosine))
            || !(c.max_abs_diff.is_finite() && c.max_abs_diff >= 0.0)
        {
            return Err("conformance: rows above 0, min_cosine in [-1, 1], max_abs_diff finite and not negative".into());
        }
        let mut reference_names = std::collections::HashSet::new();
        for r in &self.references {
            if !reference_names.insert(&r.name) {
                return Err(format!("reference {:?} appears more than once", r.name));
            }
            if !REFERENCES.contains(&(r.name.as_str(), r.role.as_str())) {
                return Err(format!("reference {:?} with role {:?} is not one of {REFERENCES:?}", r.name, r.role));
            }
            if let Some(m) = &r.measured
                && let Some(n) = m.computed_tokens
                && !slots.contains(&n)
            {
                return Err(format!(
                    "reference {}: computed_tokens {n} is not between the live tokens and the padded pass, {slots:?}",
                    r.name
                ));
            }
            if let Some(m) = &r.measured
                && let Some(c) = m.min_cosine
                && !(c.is_finite() && (-1.0..=1.0 + 1e-6).contains(&c))
            {
                return Err(format!("reference {}: min_cosine {c} is not in [-1, 1]", r.name));
            }
            match (&r.measured, &r.not_run) {
                (Some(m), None) => {
                    if m.iterations == 0
                        || ![m.p50_ms, m.p99_ms, m.rows_per_second].into_iter().all(positive)
                        || m.p99_ms < m.p50_ms
                        || r.version.is_empty()
                        || r.commands.is_empty()
                    {
                        return Err(format!(
                            "reference {}: a measurement has iterations, a version, its commands, \
                             and figures above 0 with p50 <= p99",
                            r.name
                        ));
                    }
                }
                (None, Some(why)) if !why.is_empty() => {}
                _ => return Err(format!("reference {}: exactly one of measured and not_run", r.name)),
            }
            // A program disabled before an image was named has none.
            if (r.measured.is_some() || !r.pinned.is_empty()) && pinned(&r.pinned).is_none() {
                return Err(format!("reference {}: {:?} is not name@sha256:<64 hex>", r.name, r.pinned));
            }
        }
        let text = serde_json::to_string(self).map_err(|e| e.to_string())?;
        // JSON writes a backslash as two, so a Windows home directory is
        // folded back to one before the prefixes are looked for.
        let text = text.replace("\\\\", "\\");
        if let Some(home) = HOST_PATHS.iter().find(|h| text.contains(*h)) {
            return Err(format!(
                "the record holds a host path ({home}...): a path on the machine that made it is written as a \
                 placeholder"
            ));
        }
        let (ratio, by) = speed(t.p50_ms, &self.references);
        if fastest(&self.references).is_some() && ratio.is_none() {
            return Err("speed_ratio must be finite and above zero".into());
        }
        let same = match (ratio, self.speed_ratio) {
            (Some(a), Some(b)) => positive(b) && (a - b).abs() <= 1e-12 * a.abs(),
            (None, None) => true,
            _ => false,
        };
        if !same || by != self.speed_reference {
            return Err(format!(
                "speed_ratio {:?} by {:?} is not what the references give, {ratio:?} by {by:?}",
                self.speed_ratio, self.speed_reference
            ));
        }
        if setting(self, "TURBO_NPU_GRAPH_INPUT") == Some("INPUT_EMBEDDINGS") && self.speed_ratio.is_some() {
            return Err(
                "INPUT_EMBEDDINGS has no speed_ratio: the host gather is not benchmark_app's full ONNX encoder".into(),
            );
        }
        Ok(())
    }
}

/// An npu record names the graph format the device selected and the
/// loaded artifact's graph input. Skipping either would let a later
/// NATIVE driver, or a host-gather artifact, parse as the token-id
/// NGRAPH_LITE cell.
fn npu_record_settings(r: &Record) -> Result<(), String> {
    if r.device.backend != "npu" {
        return Ok(());
    }
    match setting(r, "TURBO_NPU_GRAPH_FORMAT") {
        Some("NGRAPH_LITE" | "NATIVE") => {}
        Some(other) => return Err(format!("TURBO_NPU_GRAPH_FORMAT={other} is not NGRAPH_LITE or NATIVE")),
        None => return Err("an npu record names TURBO_NPU_GRAPH_FORMAT (NGRAPH_LITE or NATIVE)".into()),
    }
    match setting(r, "TURBO_NPU_GRAPH_INPUT") {
        Some("INPUT_TOKEN_IDS" | "INPUT_EMBEDDINGS") => Ok(()),
        Some(other) => Err(format!("TURBO_NPU_GRAPH_INPUT={other} is not INPUT_TOKEN_IDS or INPUT_EMBEDDINGS")),
        None => Err("an npu record names TURBO_NPU_GRAPH_INPUT (INPUT_TOKEN_IDS or INPUT_EMBEDDINGS)".into()),
    }
}
