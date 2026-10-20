//! Device enumeration and capability resolution for a runtime.

use super::*;

impl Runtime {
    pub(super) fn log(&self, level: u32, message: &str) {
        if let Some(f) = self.log {
            let t = turbo_text { ptr: message.as_ptr() as *const c_char, len: message.len() as u64 };
            unsafe { f(self.log_user_data as *mut c_void, level, t) };
        }
    }

    pub(super) fn device(&self, index: u32) -> Result<&Device> {
        self.devices.get(index as usize).ok_or_else(|| {
            Error::new(INVALID_ARGUMENT, format!("device {index}: the runtime has {} devices", self.devices.len()))
        })
    }

    /// Every device each linked backend lists. A backend whose probe fails
    /// lists nothing, and the log says why; a table of the wrong size fails
    /// the runtime.
    pub(super) fn enumerate(&mut self) -> Result<()> {
        for b in backend::linked() {
            backend::check_table(b)?;
            let mut n = 0u32;
            if let Err(e) = backend::check(b, "device_count", |err| unsafe { (b.device_count)(&mut n, err) }) {
                self.log(LOG_WARNING, &e.message);
                continue;
            }
            for ordinal in 0..n {
                match probe(b, ordinal) {
                    Ok(info) => self.devices.push(Device { backend: b, ordinal, info }),
                    Err(e) => self.log(LOG_WARNING, &e.message),
                }
            }
        }
        Ok(())
    }

    /// The (device, task, precision) cell. The backend says what it has
    /// built; SUPPORTED needs a benchmark record compiled into this build
    /// that backs it (record.rs, docs/benchmarks.md), and then its numbers
    /// are the record's.
    pub(super) fn capability(&self, index: u32, task: u32, precision: u32) -> Result<turbo_capability> {
        let d = self.device(index)?;
        if matches!(task, TURBO_TASK_RERANK | TURBO_TASK_EXTRACTIVE_QA) {
            let mut cap = self.capability(index, TURBO_TASK_EMBED, precision)?;
            cap.speed_ratio = 0.0;
            cap.cosine_floor = 0.0;
            cap.benchmark.fill(0);
            if d.backend.name() == "cpu" && cap.status != backend::TURBO_CAP_UNSUPPORTED {
                cap.status = backend::TURBO_CAP_EXPERIMENTAL;
                cap.options_honored = 0b111;
                write_str(&mut cap.reason, if task == TURBO_TASK_RERANK { "CPU cross-encoder; rerank speed record pending" } else { "CPU extractive QA; task benchmark pending" });
            } else {
                cap.status = backend::TURBO_CAP_UNSUPPORTED;
                cap.dtype = 0;
                cap.options_honored = 0;
                write_str(&mut cap.reason, "rerank and extractive QA require CPU");
            }
            return Ok(cap);
        }
        if task != TURBO_TASK_EMBED && task != TURBO_TASK_RERANK && task != TURBO_TASK_EXTRACTIVE_QA {
            return Err(Error::new(INVALID_ENUM, format!("task: {task} is not a TURBO_TASK_* value")));
        }
        if precision > TURBO_PRECISION_EXACT {
            return Err(Error::new(INVALID_ENUM, format!("precision: {precision} is not a TURBO_PRECISION_* value")));
        }
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        let b = d.backend;
        backend::check(b, "capability", |err| unsafe {
            (b.capability)(
                d.ordinal,
                task,
                precision,
                &mut cap.status,
                &mut cap.dtype,
                &mut cap.options_honored,
                cap.reason.as_mut_ptr(),
                cap.reason.len() as u32,
                err,
            )
        })?;
        match cap.status {
            backend::TURBO_CAP_UNSUPPORTED => {
                // A cell that does not run computes in nothing.
                cap.dtype = 0;
                cap.options_honored = 0;
            }
            backend::TURBO_CAP_EXPERIMENTAL => {
                // The options the core applies before a backend sees the rows.
                cap.options_honored |= session::CORE_HONORED;
                let graph_format = cell_graph_format(b.name(), d.ordinal);
                let cell = record::Cell {
                    arch: &cstr(&d.info.arch),
                    name: &cstr(&d.info.name),
                    cpu: d.info.kind == TURBO_DEVICE_CPU,
                    backend: b.name(),
                    task,
                    precision,
                    dtype: cap.dtype,
                    version: record::library_version(),
                    os: std::env::consts::OS,
                    graph_format,
                };
                match record::decide_embedded(&cell) {
                    record::Verdict::Supported { benchmark, cosine_floor, speed_ratio } => {
                        cap.status = backend::TURBO_CAP_SUPPORTED;
                        cap.cosine_floor = cosine_floor as f32;
                        cap.speed_ratio = speed_ratio as f32;
                        write_str(&mut cap.benchmark, &benchmark);
                        write_str(&mut cap.reason, "");
                    }
                    record::Verdict::Not(why) => write_str(&mut cap.reason, &why),
                }
            }
            s => {
                return Err(Error::new(
                    status::INTERNAL,
                    format!(
                        "{} backend reported status {s}; only EXPERIMENTAL or UNSUPPORTED are its to report",
                        b.name()
                    ),
                ));
            }
        }
        Ok(cap)
    }
}

/// The graph format an npu device loads, so a NGRAPH_LITE record and a
/// NATIVE record are different cells. None for every other backend.
pub(super) fn cell_graph_format(backend: &str, ordinal: u32) -> Option<&'static str> {
    #[cfg(feature = "npu")]
    if backend == "npu" {
        return crate::npu::load_format(ordinal);
    }
    let _ = (backend, ordinal);
    None
}

/// Ask a backend about one of its devices, now.
pub(super) fn probe(b: &'static backend::turbo_backend, ordinal: u32) -> Result<turbo_device_info> {
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    backend::check(b, "device_info", |err| unsafe { (b.device_info)(ordinal, &mut info, err) })?;
    info.struct_size = size_of::<turbo_device_info>() as u32;
    write_str(&mut info.backend, b.name());
    Ok(info)
}
