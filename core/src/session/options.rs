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
}

pub(super) unsafe fn embed_options(s: &SessionInner, opts: *const turbo_embed_options) -> Result<EmbedOptions> {
    let o = match unsafe { opts.as_ref() } {
        Some(o) => {
            sized(o.struct_size, size_of::<turbo_embed_options>(), "turbo_embed_options")?;
            *o
        }
        None => turbo_embed_options {
            struct_size: 0,
            truncate: 0,
            max_tokens: 0,
            prompt_role: 0,
            normalize: 0,
            pooling: 0,
            output_dim: 0,
        },
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
    })
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
    state.written = Some(Written { batch: rows.batch, output_dim: o.output_dim, tokenized });
    Ok(())
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
