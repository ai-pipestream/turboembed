//! Validated embedding options and shared write state.

use super::*;

// ---- Writes ---------------------------------------------------------------------

/// turbo_embed_options with every 0 resolved to what the bundle says.
#[derive(Debug, Clone, Copy)]
pub(super) struct EmbedOptions {
    pub(super) truncation: Truncation,
    /// 0 when the caller gave none.
    pub(super) max_tokens: u32,
    pub(super) prompt: PromptRole,
    pub(super) pooling: u32,
    pub(super) normalize: u32,
    pub(super) output_dim: u32,
    /// The caller's truncate and prompt_role, which rows written as tokens
    /// refuse.
    pub(super) truncate_given: bool,
    pub(super) prompt_given: bool,
    /// The caller's pooling, 0 when it gave none: a range write takes 0
    /// or TURBO_POOLING_MEAN alone.
    pub(super) pooling_given: u32,
    pub(super) quantization: Quantization,
    /// The caller's [2, output_dim] int8 ranges, valid for the write call
    /// and checked; NULL unless quantization is int8.
    pub(super) int8_ranges: *const f32,
    /// Encoder layers from the end to tap on range writes (0 when none given).
    pub(super) layers: u32,
}

pub(super) unsafe fn embed_options(s: &SessionInner, opts: *const turbo_embed_options) -> Result<EmbedOptions> {
    // A caller built against the header before quantization or layers
    // gives the earlier prefix alone, which is read without a reference
    // to the whole struct; the rest are then 0.
    let size = if opts.is_null() { 0 } else { unsafe { *(opts as *const u32) } };
    let o = if opts.is_null() {
        turbo_embed_options {
            struct_size: 0,
            truncate: 0,
            max_tokens: 0,
            prompt_role: 0,
            normalize: 0,
            pooling: 0,
            output_dim: 0,
            quantization: 0,
            int8_ranges: std::ptr::null(),
            layers: 0,
            reserved: 0,
        }
    } else if size as usize == TURBO_EMBED_OPTIONS_SIZE_V1 {
        let v1 = unsafe { std::slice::from_raw_parts(opts as *const u32, 7) };
        turbo_embed_options {
            struct_size: size,
            truncate: v1[1],
            max_tokens: v1[2],
            prompt_role: v1[3],
            normalize: v1[4],
            pooling: v1[5],
            output_dim: v1[6],
            quantization: 0,
            int8_ranges: std::ptr::null(),
            layers: 0,
            reserved: 0,
        }
    } else if size as usize == TURBO_EMBED_OPTIONS_SIZE_V2 {
        let v2 = unsafe { std::slice::from_raw_parts(opts as *const u32, 8) };
        let ranges_at = std::mem::offset_of!(turbo_embed_options, int8_ranges);
        let int8_ranges = unsafe { *opts.cast::<u8>().add(ranges_at).cast::<*const f32>() };
        turbo_embed_options {
            struct_size: size,
            truncate: v2[1],
            max_tokens: v2[2],
            prompt_role: v2[3],
            normalize: v2[4],
            pooling: v2[5],
            output_dim: v2[6],
            quantization: v2[7],
            int8_ranges,
            layers: 0,
            reserved: 0,
        }
    } else {
        sized(size, size_of::<turbo_embed_options>(), "turbo_embed_options")?;
        let raw = unsafe { *opts };
        if raw.reserved != 0 {
            return Err(Error::new(INVALID_ARGUMENT, "turbo_embed_options.reserved must be 0"));
        }
        raw
    };
    let enumerated = |name: &str, v: u32, last: u32, prefix: &str| {
        if v > last {
            return Err(Error::new(INVALID_ENUM, format!("{name}: {v} is not a TURBO_{prefix}_* value")));
        }
        Ok(v)
    };
    enumerated("truncate", o.truncate, TURBO_TRUNCATE_LEFT, "TRUNCATE")?;
    let prompt = prompt_role(o.prompt_role)?;
    enumerated("normalize", o.normalize, TURBO_NORMALIZE_L2, "NORMALIZE")?;
    enumerated("pooling", o.pooling, TURBO_POOLING_LAST, "POOLING")?;
    enumerated("quantization", o.quantization, TURBO_QUANTIZE_UBINARY, "QUANTIZE")?;

    honored(&o, s.honored, s.backend.name())?;

    if o.max_tokens > s.info.max_seq {
        return Err(Error::new(
            CAPACITY,
            format!("max_tokens {} is over the session's max_seq {}", o.max_tokens, s.info.max_seq),
        ));
    }
    let mi = &s.model.info;
    let output_dim = match o.output_dim {
        0 => mi.dim,
        d if d > mi.dim => {
            return Err(Error::field(
                INVALID_ARGUMENT,
                6,
                format!("output_dim {d} is over the model's dim {}", mi.dim),
            ));
        }
        d if d == mi.dim || s.model.output_dims.contains(&d) => d,
        d => {
            return Err(Error::field(
                UNSUPPORTED_OPTION,
                6,
                format!(
                    "output_dim {d}: the model was trained to be cut to {:?} besides its dim {}",
                    s.model.output_dims, mi.dim
                ),
            ));
        }
    };
    let quantization = match o.quantization {
        TURBO_QUANTIZE_INT8 => Quantization::Int8,
        TURBO_QUANTIZE_UBINARY => Quantization::UBinary,
        _ => Quantization::F32,
    };
    unsafe { int8_ranges(quantization, o.int8_ranges, output_dim) }?;
    Ok(EmbedOptions {
        truncation: match o.truncate {
            TURBO_TRUNCATE_MODEL => s.model.tokenizer.truncation(),
            TURBO_TRUNCATE_NONE => Truncation::None,
            TURBO_TRUNCATE_RIGHT => Truncation::Right,
            _ => Truncation::Left,
        },
        max_tokens: o.max_tokens,
        prompt,
        pooling: if o.pooling == TURBO_POOLING_MODEL { mi.pooling } else { o.pooling },
        normalize: if o.normalize == TURBO_NORMALIZE_MODEL { mi.normalize } else { o.normalize },
        output_dim,
        truncate_given: o.truncate != 0,
        prompt_given: o.prompt_role != 0,
        pooling_given: o.pooling,
        quantization,
        int8_ranges: o.int8_ranges,
        layers: o.layers,
    })
}

/// Field 8 is given exactly when field 7 asks for int8, and then holds a
/// usable [2, output_dim] table: nothing is ignored.
unsafe fn int8_ranges(q: Quantization, ranges: *const f32, output_dim: u32) -> Result<()> {
    match (q, ranges.is_null()) {
        (Quantization::Int8, true) => {
            Err(Error::field(INVALID_ARGUMENT, 8, "int8_ranges is NULL: TURBO_QUANTIZE_INT8 needs the caller's ranges"))
        }
        (Quantization::Int8, false) => {
            let r = unsafe { std::slice::from_raw_parts(ranges, 2 * output_dim as usize) };
            quantize::check_int8_ranges(r).map_err(|m| Error::field(INVALID_ARGUMENT, 8, m))
        }
        (_, false) => {
            Err(Error::field(INVALID_ARGUMENT, 8, "int8_ranges is given, but quantization is not TURBO_QUANTIZE_INT8"))
        }
        (_, true) => Ok(()),
    }
}

/// A field the device does not honor may only be left to the bundle: any
/// other value is refused by its number.
pub(super) fn honored(o: &turbo_embed_options, honored: u32, backend: &str) -> Result<()> {
    let fields = [
        ("truncate", o.truncate),
        ("max_tokens", o.max_tokens),
        ("prompt_role", o.prompt_role),
        ("normalize", o.normalize),
        ("pooling", o.pooling),
        ("output_dim", o.output_dim),
        ("quantization", o.quantization),
        ("int8_ranges", !o.int8_ranges.is_null() as u32),
    ];
    for (i, (name, v)) in fields.iter().enumerate() {
        if *v != 0 && honored & (1 << i) == 0 {
            return Err(Error::field(
                UNSUPPORTED_OPTION,
                i as u32 + 1,
                format!("{name} {v}: the {backend} backend does not honor it at this precision"),
            ));
        }
    }
    Ok(())
}

/// The start of every write: the session is free, it is an embed
/// session, and nothing is left written until this write succeeds.
pub(super) fn begin_write(s: &SessionInner) -> Result<MutexGuard<'_, State>> {
    let mut state = s.lock()?;
    state.written = None;
    state.rerank = None;
    if !matches!(s.model.info.task, TURBO_TASK_EMBED | TURBO_TASK_EXTRACTIVE_QA) {
        return Err(Error::new(UNSUPPORTED_TASK, "the model's task is not embed"));
    }
    Ok(state)
}

/// Hand checked rows to the backend and record them.
pub(super) fn write_rows(
    s: &SessionInner,
    state: &mut State,
    rows: turbo_backend_embed_rows,
    o: &EmbedOptions,
    tokenized: bool,
) -> Result<()> {
    let b = s.backend;
    let write = backend::offered!(b, embed_write)?;
    backend::check(b, "embed_write", |err| unsafe { write(s.raw, &rows, err) })?;
    record(state, o, rows.batch, o.output_dim, tokenized);
    Ok(())
}

/// What a write left for its run, and the int8 ranges it gave, copied into
/// the session's own memory.
pub(super) fn record(state: &mut State, o: &EmbedOptions, batch: u32, output_dim: u32, tokenized: bool) {
    if o.quantization == Quantization::Int8 {
        let n = 2 * output_dim as usize;
        let given = unsafe { std::slice::from_raw_parts(o.int8_ranges, n) };
        state.int8_ranges[..n].copy_from_slice(given);
    }
    state.written = Some(Written { batch, output_dim, tokenized, quantization: o.quantization });
}

pub(super) fn backend_rows(o: &EmbedOptions, batch: u32, seq: u32, row_stride: u32) -> turbo_backend_embed_rows {
    turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch,
        seq,
        row_stride,
        ids: std::ptr::null(),
        mask: std::ptr::null(),
        types: std::ptr::null(),
        pooling: o.pooling,
        normalize: o.normalize,
        output_dim: o.output_dim,
        head: 0,
        taps: 0,
        reserved: 0,
    }
}

/// A run of texts' ids, one after the other in one buffer, and where each
/// text's end: a batch allocates per run, not per text.
pub(super) struct Part {
    pub(super) ids: Vec<i32>,
    pub(super) ends: Vec<usize>,
}

impl Part {
    pub(super) fn rows(&self) -> impl Iterator<Item = &[i32]> {
        self.ends.iter().scan(0, |at, &end| {
            let row = &self.ids[*at..end];
            *at = end;
            Some(row)
        })
    }
}
