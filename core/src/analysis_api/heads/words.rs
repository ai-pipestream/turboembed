//! The bundle's encoder, read as the analysis annotators read it
//! (`turbo-tagger`'s word encoder): each word tokenized on its own, a
//! sequence in one row when its pieces fit and else in windows of whole
//! words (`head::run::plan`), and each word's final-layer states pooled
//! on the device by the range write (`turbo_embed_write_ranges`), its
//! first piece under WORD_POOLING_FIRST, the mean of its pieces under
//! WORD_POOLING_MEAN. A word the tokenizer gives no piece is read as the
//! unknown piece; a word of more pieces than a row holds keeps the first
//! row's worth. The session runs on the cpu backend.

use std::path::Path;
use std::ptr;

use crate::bundle::Bundle;
use crate::head::run::plan;
use crate::manifest::{PromptRole, Truncation, WordPooling};
use crate::status::{DEVICE_NOT_FOUND, Error, INTERNAL, INVALID_ARGUMENT, Result, UNSUPPORTED_OPTION};
use crate::tokenizer::{Encode, Tokenizer};
use crate::{
    TURBO_AUTOTUNE_RUNTIME, TURBO_DEVICE_CPU, TURBO_DTYPE_F32, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN,
    TURBO_PRECISION_MODEL, turbo_context, turbo_context_create, turbo_context_release, turbo_device_info,
    turbo_embed_options, turbo_embed_write_ranges, turbo_error, turbo_model, turbo_model_load, turbo_model_release,
    turbo_result, turbo_result_get_info, turbo_result_info, turbo_result_read, turbo_result_release, turbo_runtime,
    turbo_runtime_create, turbo_runtime_device_count, turbo_runtime_device_info, turbo_runtime_release, turbo_session,
    turbo_session_create, turbo_session_desc, turbo_session_release, turbo_session_run, turbo_text, turbo_token_batch,
    turbo_token_range,
};

const BARE: Encode =
    Encode { add_special_tokens: false, truncation: Truncation::None, max_tokens: u32::MAX, prompt: PromptRole::None };

/// The encoder, its tokenizer, and how rows are laid out.
pub(super) struct Words {
    rt: *mut turbo_runtime,
    ctx: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
    tokenizer: Tokenizer,
    cls: Vec<i32>,
    sep: Vec<i32>,
    /// Pieces per row, beside the special tokens.
    width: usize,
    max_batch: usize,
    max_ranges: usize,
    hidden: usize,
    /// The backend and device the session runs on, for provenance.
    pub(super) backend: String,
    pub(super) device: String,
}

// SAFETY: the runtime, context and model may be used from any thread, and
// the session from one owner at a time, which `&mut self` on every call
// that uses it makes so; the engine holds a `Words` behind a mutex.
unsafe impl Send for Words {}

impl Drop for Words {
    fn drop(&mut self) {
        // Each release takes NULL, so a half-made encoder drops too.
        unsafe {
            turbo_session_release(self.session);
            turbo_model_release(self.model);
            turbo_context_release(self.ctx);
            turbo_runtime_release(self.rt);
        }
    }
}

fn new_error() -> turbo_error {
    turbo_error { struct_size: size_of::<turbo_error>() as u32, code: 0, field: 0, message: [0; 496] }
}

/// A status from the C interface as the core's error, with its message.
fn check(rc: i32, err: &turbo_error) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let bytes: Vec<u8> = err.message.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    Err(Error::field(rc, err.field, String::from_utf8_lossy(&bytes).into_owned()))
}

/// Rows of token ids for one run, `stride` apart, padded with `fill`
/// under mask 0.
struct Rows {
    ids: Vec<i32>,
    mask: Vec<i32>,
    stride: usize,
    fill: i32,
    batch: u32,
    seq: u32,
}

impl Rows {
    fn push(&mut self, parts: &[&[i32]]) -> u32 {
        let at = self.ids.len();
        self.ids.resize(at + self.stride, self.fill);
        self.mask.resize(at + self.stride, 0);
        let mut p = at;
        for part in parts {
            self.ids[p..p + part.len()].copy_from_slice(part);
            p += part.len();
        }
        self.mask[at..p].fill(1);
        self.seq = self.seq.max((p - at) as u32);
        self.batch += 1;
        self.batch - 1
    }

    fn clear(&mut self) {
        self.ids.clear();
        self.mask.clear();
        self.batch = 0;
        self.seq = 0;
    }

    fn batch(&self) -> turbo_token_batch {
        turbo_token_batch {
            struct_size: size_of::<turbo_token_batch>() as u32,
            batch: self.batch,
            seq: self.seq,
            row_stride: self.stride as u32,
            ids: self.ids.as_ptr(),
            mask: self.mask.as_ptr(),
            types: ptr::null(),
        }
    }
}

/// Where a range's states go: pooling, sequence, word. A range of one
/// piece of `pieces` (`piece` counted from 0) feeds every pooling, which
/// is pooled from the pieces here.
struct Owner {
    pooling: Option<usize>,
    seq: usize,
    word: usize,
    piece: usize,
    pieces: usize,
}

impl Words {
    /// The encoder of `bundle`, read from `dir`, on the first CPU device
    /// the runtime lists, in rows of the bundle's max_seq, max_batch a run,
    /// with the ranges the Rust annotators hold by default.
    pub(super) fn open(dir: &Path, bundle: &Bundle) -> Result<Words> {
        let path =
            dir.to_str().ok_or_else(|| Error::new(INVALID_ARGUMENT, format!("{}: not a UTF-8 path", dir.display())))?;
        let tokenizer = Tokenizer::load(bundle)?;
        let embed = bundle.manifest.embed();
        let (cls, sep) = tokenizer.template_ends();
        let overhead = cls.len() + sep.len();
        if embed.max_seq as usize <= overhead {
            return Err(Error::new(
                INVALID_ARGUMENT,
                format!("max_seq {} leaves no room beside {overhead} special tokens", embed.max_seq),
            ));
        }
        let mut w = Words {
            rt: ptr::null_mut(),
            ctx: ptr::null_mut(),
            model: ptr::null_mut(),
            session: ptr::null_mut(),
            width: embed.max_seq as usize - overhead,
            max_batch: embed.max_batch as usize,
            max_ranges: default_ranges(embed.max_batch as usize, embed.max_seq as usize),
            hidden: embed.dim as usize,
            tokenizer,
            cls,
            sep,
            backend: String::new(),
            device: String::new(),
        };
        let mut err = new_error();
        unsafe {
            check(turbo_runtime_create(ptr::null(), &mut w.rt, &mut err), &err)?;
            let (index, info) = w.cpu_device()?;
            check(turbo_context_create(w.rt, index, &mut w.ctx, &mut err), &err)?;
            let text = turbo_text { ptr: path.as_ptr().cast(), len: path.len() as u64 };
            check(turbo_model_load(w.ctx, text, &mut w.model, &mut err), &err)?;
            let desc = turbo_session_desc {
                struct_size: size_of::<turbo_session_desc>() as u32,
                max_batch: embed.max_batch,
                max_seq: embed.max_seq,
                precision: TURBO_PRECISION_MODEL,
                tuning: TURBO_AUTOTUNE_RUNTIME,
                tuning_budget_ms: 0,
                cpu_threads: 0,
                max_ranges: w.max_ranges as u32,
            };
            check(turbo_session_create(w.model, &desc, &mut w.session, &mut err), &err)?;
            w.backend = c_field(&info.backend);
            w.device = c_field(&info.name);
        }
        Ok(w)
    }

    /// The first CPU device the runtime lists.
    unsafe fn cpu_device(&self) -> Result<(u32, turbo_device_info)> {
        let (mut n, mut err) = (0, new_error());
        unsafe { check(turbo_runtime_device_count(self.rt, &mut n, &mut err), &err)? };
        for i in 0..n {
            let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
            info.struct_size = size_of::<turbo_device_info>() as u32;
            unsafe { check(turbo_runtime_device_info(self.rt, i, &mut info, &mut err), &err)? };
            if info.kind == TURBO_DEVICE_CPU {
                return Ok((i, info));
            }
        }
        Err(Error::new(DEVICE_NOT_FOUND, format!("no cpu device is listed among {n}")))
    }

    /// Values per word.
    pub(super) fn hidden(&self) -> usize {
        self.hidden
    }

    /// A word's pieces as a row holds them: the unknown piece for none,
    /// the first row's worth of many.
    fn pieces(&self, word: &str) -> Result<Vec<i32>> {
        let (mut ids, mut spans) = (Vec::new(), Vec::new());
        self.tokenizer.encode_spans(word, BARE, &mut ids, &mut spans)?;
        if ids.is_empty() {
            ids.push(self.tokenizer.unk_id);
        }
        ids.truncate(self.width);
        Ok(ids)
    }

    /// Each sequence's word features `[words * hidden]` under each pooling
    /// of `poolings`, `out[p][s]` being `poolings[p]`'s for `sequences[s]`,
    /// from one read of each window whatever the poolings.
    pub(super) fn features(&mut self, sequences: &[Vec<&str>], poolings: &[WordPooling]) -> Result<Vec<Vec<Vec<f32>>>> {
        let pieces: Vec<Vec<Vec<i32>>> =
            sequences.iter().map(|words| words.iter().map(|w| self.pieces(w)).collect()).collect::<Result<_>>()?;
        let mut out: Vec<Vec<Vec<f32>>> =
            poolings.iter().map(|_| pieces.iter().map(|s| vec![0.0; s.len() * self.hidden]).collect()).collect();
        let stride = self.width + self.cls.len() + self.sep.len();
        let mut rows =
            Rows { ids: Vec::new(), mask: Vec::new(), stride, fill: self.tokenizer.fill_id(), batch: 0, seq: 0 };
        let (mut ranges, mut owners, mut states) = (Vec::new(), Vec::<Owner>::new(), Vec::new());
        let mut row_ids = Vec::new();
        for (s, words) in pieces.iter().enumerate() {
            let lens: Vec<usize> = words.iter().map(Vec::len).collect();
            let (spans, owner) = plan(&lens, self.width);
            for (i, &(a, b)) in spans.iter().enumerate() {
                let kept: Vec<usize> = (a..b).filter(|&w| owner[w] == i).collect();
                let kept_pieces: usize = kept.iter().map(|&w| lens[w]).sum();
                // A range a word and pooling, unless those outgrow a run
                // (one row a batch): then a range a piece, which every
                // pooling reads, so the row is still read once.
                let per_piece = kept.len() * poolings.len() > self.max_ranges;
                let need = if per_piece { kept_pieces } else { kept.len() * poolings.len() };
                if rows.batch as usize == self.max_batch || ranges.len() + need > self.max_ranges {
                    self.flush(&mut rows, &mut ranges, &mut owners, &mut states, &mut out, poolings)?;
                }
                row_ids.clear();
                for p in &words[a..b] {
                    row_ids.extend_from_slice(p);
                }
                let row = rows.push(&[&self.cls, &row_ids, &self.sep]);
                let mut at = self.cls.len();
                for w in a..b {
                    let n = lens[w];
                    if owner[w] == i {
                        let start = at as u32;
                        if per_piece {
                            for piece in 0..n {
                                let p = start + piece as u32;
                                ranges.push(turbo_token_range { row, start: p, end: p + 1 });
                                owners.push(Owner { pooling: None, seq: s, word: w, piece, pieces: n });
                            }
                        } else {
                            for (p, pooling) in poolings.iter().enumerate() {
                                let len = match pooling {
                                    WordPooling::First => 1,
                                    WordPooling::Mean => n as u32,
                                };
                                ranges.push(turbo_token_range { row, start, end: start + len });
                                owners.push(Owner { pooling: Some(p), seq: s, word: w, piece: 0, pieces: n });
                            }
                        }
                    }
                    at += n;
                }
            }
        }
        self.flush(&mut rows, &mut ranges, &mut owners, &mut states, &mut out, poolings)?;
        Ok(out)
    }

    fn flush(
        &mut self,
        rows: &mut Rows,
        ranges: &mut Vec<turbo_token_range>,
        owners: &mut Vec<Owner>,
        states: &mut Vec<f32>,
        out: &mut [Vec<Vec<f32>>],
        poolings: &[WordPooling],
    ) -> Result<()> {
        if !ranges.is_empty() {
            self.run_ranges(rows, ranges, states)?;
            let h = self.hidden;
            for (r, o) in owners.iter().enumerate() {
                let state = &states[r * h..][..h];
                let Some(p) = o.pooling else {
                    // A word's pieces come in order in one run; the mean
                    // is the cpu's: the sum in order, times 1 / pieces.
                    for (p, pooling) in poolings.iter().enumerate() {
                        let dst = &mut out[p][o.seq][o.word * h..][..h];
                        match pooling {
                            WordPooling::First if o.piece == 0 => dst.copy_from_slice(state),
                            WordPooling::First => {}
                            WordPooling::Mean => {
                                if o.piece == 0 {
                                    dst.fill(0.0);
                                }
                                dst.iter_mut().zip(state).for_each(|(d, &x)| *d += x);
                                if o.piece + 1 == o.pieces {
                                    let inv = 1.0 / o.pieces as f32;
                                    dst.iter_mut().for_each(|d| *d *= inv);
                                }
                            }
                        }
                    }
                    continue;
                };
                out[p][o.seq][o.word * h..][..h].copy_from_slice(state);
            }
        }
        rows.clear();
        ranges.clear();
        owners.clear();
        Ok(())
    }

    /// `rows` encoded once and, into `out`, the mean of the encoder's
    /// final states over each range's tokens: `[ranges.len(), hidden]`.
    fn run_ranges(&mut self, rows: &Rows, ranges: &[turbo_token_range], out: &mut Vec<f32>) -> Result<()> {
        struct Owned(*mut turbo_result);
        impl Drop for Owned {
            fn drop(&mut self) {
                unsafe { turbo_result_release(self.0) };
            }
        }
        let o = turbo_embed_options {
            struct_size: size_of::<turbo_embed_options>() as u32,
            truncate: 0,
            max_tokens: 0,
            prompt_role: 0,
            normalize: TURBO_NORMALIZE_NONE,
            pooling: TURBO_POOLING_MEAN,
            output_dim: 0,
            quantization: 0,
            int8_ranges: ptr::null(),
        };
        let mut err = new_error();
        let n = ranges.len() as u32;
        let mut r = Owned(ptr::null_mut());
        let mut info: turbo_result_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_result_info>() as u32;
        unsafe {
            check(turbo_embed_write_ranges(self.session, &rows.batch(), ranges.as_ptr(), n, &o, &mut err), &err)?;
            check(turbo_session_run(self.session, &mut r.0, &mut err), &err)?;
            check(turbo_result_get_info(r.0, &mut info, &mut err), &err)?;
        }
        if info.dtype != TURBO_DTYPE_F32 {
            return Err(Error::new(UNSUPPORTED_OPTION, format!("vectors of dtype {}, not f32", info.dtype)));
        }
        if info.batch != n || info.dim as usize != self.hidden {
            return Err(Error::new(
                INTERNAL,
                format!("a result of {} vectors of {}, for {n} vectors of {}", info.batch, info.dim, self.hidden),
            ));
        }
        out.resize(n as usize * self.hidden, 0.0);
        let bytes = size_of_val(out.as_slice()) as u64;
        let mut written = 0u64;
        unsafe { check(turbo_result_read(r.0, out.as_mut_ptr().cast(), bytes, &mut written, &mut err), &err)? };
        if written != bytes {
            return Err(Error::new(INTERNAL, format!("the result read {written} bytes of {bytes}")));
        }
        Ok(())
    }
}

/// A fixed-size C string field as a String.
fn c_field(b: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The ranges a session holds by default: 16 a row, at most one a token,
/// as `turbo-analysis-embed` chooses when asked for none in particular.
fn default_ranges(max_batch: usize, max_seq: usize) -> usize {
    max_batch.saturating_mul(16).min(max_batch.saturating_mul(max_seq))
}
