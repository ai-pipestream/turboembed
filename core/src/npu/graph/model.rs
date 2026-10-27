//! Models: the graph the driver compiles from the IR, initialized.

use std::ffi::{CString, c_char, c_void};
use std::sync::Mutex;

use super::accept::*;
use super::args::*;
use super::blob::{self, Loader, Source};
use super::*;
use crate::backend::{
    TURBO_FORMAT_OPENVINO_IR, TURBO_INPUT_EMBEDDINGS, TURBO_INPUT_TOKEN_IDS, TURBO_OUTPUT_HIDDEN_STATES,
    turbo_backend_model,
};
use crate::npu::ir;
use crate::npu::native::{self, BlobError};
use crate::npu::ze::{self, GraphExt, Handle};
use crate::status::{INVALID_ARGUMENT, RUNTIME, UNSUPPORTED};
use crate::turbo_error;

pub(crate) struct Model {
    pub(super) ctx: *const Context,
    pub(super) graph: Handle,
    /// Argument values live on the graph: one run at a time per model.
    pub(super) run: Mutex<()>,
    pub(super) inputs: GraphInputs,
    pub(super) output: Arg,
    /// The shapes compiled in: a frame of rows, each of seq tokens.
    pub(super) frame_batch: u32,
    pub(super) seq: u32,
    pub(super) hidden: u32,
    /// TURBO_DTYPE_* a session computes in.
    pub compute_dtype: u32,
    /// `NGRAPH_LITE` or `NATIVE`: what this graph was initialized as.
    pub graph_format: &'static str,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    #[inline]
    pub(super) fn ctx(&self) -> &Context {
        unsafe { &*self.ctx }
    }
}

/// The build log's text, read and destroyed. Empty when there is none.
pub(super) fn build_log(ext: &GraphExt, log: Handle) -> String {
    if log.is_null() {
        return String::new();
    }
    let mut text = String::new();
    if let Some(get) = ext.build_log_get_string2() {
        let mut n = 0u32;
        if unsafe { get(log, &mut n, std::ptr::null_mut()) } == 0 && n > 0 {
            let mut buf = vec![0 as c_char; n as usize];
            if unsafe { get(log, &mut n, buf.as_mut_ptr()) } == 0 {
                text = crate::backend::cstr(&buf);
            }
        }
    }
    if let Some(destroy) = ext.build_log_destroy() {
        unsafe { destroy(log) };
    }
    text
}

/// model_load: the driver's compiler builds the graph from the IR's xml
/// and weights, and the graph is initialized, so a session allocates and
/// compiles nothing. A compile failure is RUNTIME with the compiler's
/// own log.
///
/// # Safety
/// As turbo_backend.h says for model_load.
pub(crate) unsafe extern "C" fn model_load(
    ctx: *mut c_void,
    desc: *const turbo_backend_model,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let ctx = &*(ctx as *const Context);
            let desc = &*desc;
            load(ctx, desc).map(|m| *out = Box::into_raw(Box::new(m)) as *mut c_void)
        })
    }
}

pub(super) fn load(ctx: &Context, desc: &turbo_backend_model) -> Res<Model> {
    if desc.format != TURBO_FORMAT_OPENVINO_IR {
        return Err(fail(UNSUPPORTED, format!("npu: model_load takes FORMAT_OPENVINO_IR, not format {}", desc.format)));
    }
    // artifact2, the IR's weights, was appended to the struct: a core
    // from before it cannot hand an IR's two files.
    let covers =
        desc.struct_size as usize >= std::mem::offset_of!(turbo_backend_model, artifact2_bytes) + size_of::<u64>();
    if !covers {
        return Err(fail(UNSUPPORTED, "npu: the core's turbo_backend_model ends before artifact2, the IR's weights"));
    }
    let embeddings = match desc.graph_input {
        TURBO_INPUT_TOKEN_IDS => None,
        TURBO_INPUT_EMBEDDINGS => Some(word_table(desc)?),
        other => {
            return Err(fail(
                UNSUPPORTED,
                format!("npu: graph_input {other} is not INPUT_TOKEN_IDS or INPUT_EMBEDDINGS"),
            ));
        }
    };
    if desc.graph_output != TURBO_OUTPUT_HIDDEN_STATES {
        return Err(fail(
            UNSUPPORTED,
            "npu: only OUTPUT_HIDDEN_STATES artifacts run; this backend pools and normalizes on the host",
        ));
    }
    if desc.artifact.is_null() || desc.artifact2.is_null() {
        return Err(fail(INVALID_ARGUMENT, "npu: an OpenVINO IR is two blocks of bytes, the xml and its weights"));
    }
    let (maj, min) = (ctx.compiler.major, ctx.compiler.minor);
    if (maj, min) < (5, 9) {
        return Err(fail(
            UNSUPPORTED,
            format!("npu: the driver's compiler is {maj}.{min}; this IR path needs 5.9 or later"),
        ));
    }
    if ctx.ext.create2().is_none() {
        return Err(fail(
            UNSUPPORTED,
            format!(
                "npu: the driver's graph extension is {}.{}; compiling an IR needs 1.5 or later",
                ctx.ext.version >> 16,
                ctx.ext.version & 0xffff
            ),
        ));
    }

    // What the device said it compiles, checked before any compile is
    // tried, so a refusal says why in words rather than a status code.
    if ctx.formats_supported & ze::GRAPH_FORMAT_NGRAPH_LITE == 0 {
        return Err(fail(
            UNSUPPORTED,
            "npu: the device's compiler does not take an OpenVINO IR (NGRAPH_LITE is not among its graph formats)",
        ));
    }
    let xml = unsafe { std::slice::from_raw_parts(desc.artifact as *const u8, desc.artifact_bytes as usize) };
    let bin = unsafe { std::slice::from_raw_parts(desc.artifact2 as *const u8, desc.artifact2_bytes as usize) };
    blob::begin();
    let io = blob::timed("ir", || ir::interface(xml)).map_err(|e| fail(RUNTIME, e))?;
    if ctx.max_opset != 0 && io.max_opset > ctx.max_opset {
        return Err(fail(
            UNSUPPORTED,
            format!(
                "npu: the IR uses opset {}, and the device's compiler supports up to opset {}; update the NPU \
                 driver or export the IR for an older opset",
                io.max_opset, ctx.max_opset
            ),
        ));
    }
    let flags = ir::build_flags(&io).map_err(|e| fail(RUNTIME, e))?;
    let cflags = CString::new(flags.clone()).map_err(|_| fail(RUNTIME, "npu: a NUL in the build flags"))?;

    // NATIVE, when the build asks for it and the extension exports the
    // compiled blob, whatever graphFormatsSupported says: the driver
    // clears bit 0x1 whenever its compiler loads and takes a NATIVE
    // create either way (blob.rs). The key is the manifest's digests; a
    // caller that passed none loads NGRAPH_LITE rather than hash here.
    let native = if blob::native_wanted() && native::exportable(&ctx.ext) {
        let key = blob::timed("key", || {
            digests(desc).map(|(x, b)| blob::Key::new(x, b, &flags, (maj, min), ctx.driver_version, ctx.arch))
        });
        if key.is_none() {
            ctx.say(LOG_DEBUG, &format!("npu device {}: no manifest digest for the IR; NGRAPH_LITE", ctx.ordinal));
        }
        key
    } else {
        None
    };
    let mut steps = OnDevice { ctx, desc, xml, bin, compiler: (maj, min), flags: &cflags, embeddings };
    let (model, source) = blob::select(&mut steps, native.as_ref(), &blob::CACHE)?;
    blob::record(source);
    ctx.say(
        LOG_DEBUG,
        &format!(
            "npu device {}: the graph is {} ({}): {}",
            ctx.ordinal,
            source.format(),
            source.name(),
            blob::phases()
        ),
    );
    Ok(model)
}

/// The manifest's SHA-256 of the IR's two files, when the caller's
/// description reaches that far and carries them.
pub(super) fn digests(desc: &turbo_backend_model) -> Option<([u8; 32], [u8; 32])> {
    let end = std::mem::offset_of!(turbo_backend_model, artifact2_sha256) + 32;
    if (desc.struct_size as usize) < end {
        return None;
    }
    let (x, b) = (desc.artifact_sha256, desc.artifact2_sha256);
    (x != [0; 32] && b != [0; 32]).then_some((x, b))
}

/// A load's steps on the device, for `blob::select`.
struct OnDevice<'a> {
    ctx: &'a Context,
    desc: &'a turbo_backend_model,
    xml: &'a [u8],
    bin: &'a [u8],
    compiler: (u16, u16),
    flags: &'a CString,
    embeddings: Option<(*const f32, u32)>,
}

impl Loader for OnDevice<'_> {
    type Graph = Handle;
    type Out = Model;

    /// The container copies the weights, so it is built here: a load
    /// from a kept blob never makes one.
    fn compile(&mut self) -> Res<Handle> {
        let ctx = self.ctx;
        let container = blob::timed("container", || ir::container(self.compiler, self.xml, self.bin));
        let g = blob::timed("compile", || {
            create_graph(
                ctx,
                ze::GRAPH_FORMAT_NGRAPH_LITE,
                &container,
                self.flags.as_ptr(),
                "the driver's compiler refused the IR",
            )
        })?;
        ctx.say(
            LOG_DEBUG,
            &format!("npu device {}: the IR is compiled ({})", ctx.ordinal, self.flags.to_string_lossy()),
        );
        Ok(g)
    }

    fn export(&mut self, lite: &Handle) -> Res<Vec<u8>> {
        blob::timed("export", || native::copy(&self.ctx.ext, *lite)).map_err(blob_fail)
    }

    /// The blob is read in place; the driver copies what it keeps.
    fn create_native(&mut self, blob: &[u8]) -> Res<Handle> {
        blob::timed("create", || {
            create_graph(self.ctx, ze::GRAPH_FORMAT_NATIVE, blob, c"".as_ptr(), "the driver refused the native blob")
        })
    }

    fn finish(&mut self, graph: Handle, source: Source) -> Res<Model> {
        let model = describe(self.ctx, self.desc, graph, self.embeddings, source.format());
        if model.is_err() {
            destroy_graph(self.ctx, graph);
        }
        model
    }

    fn destroy(&mut self, graph: Handle) {
        destroy_graph(self.ctx, graph);
    }

    fn note(&mut self, message: &str) {
        self.ctx.say(LOG_DEBUG, &format!("npu device {}: {message}", self.ctx.ordinal));
    }
}

pub(super) fn destroy_graph(ctx: &Context, graph: Handle) {
    if let Some(destroy) = ctx.ext.destroy() {
        unsafe { destroy(graph) };
    }
}

/// pfnCreate3 where the extension has it, so a refusal carries the
/// driver's log, else pfnCreate2. `refused` is the start of the
/// TURBO_E_RUNTIME message.
pub(super) fn create_graph(
    ctx: &Context,
    format: u32,
    input: &[u8],
    build_flags: *const c_char,
    refused: &str,
) -> Res<Handle> {
    let gdesc = if format == ze::GRAPH_FORMAT_NATIVE {
        native::descriptor(input, build_flags)
    } else {
        ze::GraphDesc2 {
            stype: ze::STRUCTURE_TYPE_GRAPH_DESC_2,
            p_next: std::ptr::null(),
            format,
            input_size: input.len(),
            input: input.as_ptr(),
            build_flags,
            flags: 0,
        }
    };
    let mut graph = std::ptr::null_mut();
    match ctx.ext.create3() {
        Some(create3) => {
            let mut log = std::ptr::null_mut();
            let rc = unsafe { create3(ctx.handle, ctx.device, &gdesc, &mut graph, &mut log) };
            if rc != 0 {
                let text = build_log(&ctx.ext, log);
                return Err(fail(RUNTIME, format!("npu: {refused} with 0x{rc:08x}: {text}")));
            }
            build_log(&ctx.ext, log);
        }
        None => {
            let create2 = ctx.ext.create2().ok_or_else(|| {
                fail(UNSUPPORTED, "npu: the driver's graph extension predates 1.5; pfnCreate2 is needed")
            })?;
            ze_res("pfnGraphCreate2", unsafe { create2(ctx.handle, ctx.device, &gdesc, &mut graph) })?;
        }
    }
    Ok(graph)
}

pub(super) fn blob_fail(e: BlobError) -> Fail {
    match e {
        BlobError::Unsupported(m) => fail(UNSUPPORTED, m),
        BlobError::Runtime(m) => fail(RUNTIME, m),
    }
}

/// The compiled graph's arguments and shapes, checked against the
/// manifest, and the graph initialized: the weights' move to the device.
pub(super) fn describe(
    ctx: &Context,
    desc: &turbo_backend_model,
    graph: Handle,
    embeddings: Option<(*const f32, u32)>,
    graph_format: &'static str,
) -> Res<Model> {
    let get_props = ctx.ext.get_properties2().ok_or_else(|| {
        fail(UNSUPPORTED, "npu: the driver's graph extension predates 1.8; pfnGetProperties2 is needed")
    })?;
    let get_arg = ctx.ext.get_argument_properties3().ok_or_else(|| {
        fail(UNSUPPORTED, "npu: the driver's graph extension predates 1.2; pfnGetArgumentProperties3 is needed")
    })?;
    let mut props = ze::GraphProperties2 { stype: ze::STRUCTURE_TYPE_GRAPH_PROPERTIES_2, ..Default::default() };
    ze_res("pfnGetProperties2", unsafe { get_props(graph, &mut props) })?;

    let mut inputs: Vec<Compiled> = Vec::new();
    let mut outputs: Vec<Compiled> = Vec::new();
    for i in 0..props.num_graph_args {
        let mut a = ze::GraphArgumentProperties3 {
            stype: ze::STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3,
            ..Default::default()
        };
        ze_res("pfnGetArgumentProperties3", unsafe { get_arg(graph, i, &mut a) })?;
        let name = ze::string(&a.name);
        let arg = Arg {
            index: i,
            precision: a.device_precision,
            elem: elem_bytes(&format!("argument {name:?}"), a.device_precision)?,
            name: name.clone(),
        };
        if a.dims_count == 0 {
            return Err(fail(
                RUNTIME,
                format!("npu: argument {name:?} reports dims_count 0; the rank is not guessed from the dimension list"),
            ));
        }
        let rank = a.dims_count;
        let compiled = Compiled { arg, dims: a.dims, rank, layout: a.device_layout };
        match a.kind {
            ze::GRAPH_ARGUMENT_TYPE_INPUT => inputs.push(compiled),
            _ => outputs.push(compiled),
        }
    }

    // Inputs are taken by name. A graph is never run on a guess about
    // which argument is which. One output is the hidden states whatever
    // its name; among several, only last_hidden_state is.
    let output = take_hidden(outputs)?;
    let (inputs, output, frame_batch, seq, hidden) = match desc.graph_input {
        TURBO_INPUT_TOKEN_IDS => accept_tokens(inputs, output, desc)?,
        TURBO_INPUT_EMBEDDINGS => {
            let (table, vocab) = embeddings
                .ok_or_else(|| fail(RUNTIME, "npu: INPUT_EMBEDDINGS reached the compiler with no word table"))?;
            let (rows, bias, output, frame_batch, seq, hidden) = accept_embeddings(inputs, output, desc)?;
            (GraphInputs::Embeddings { rows, bias, heads: desc.heads, table, vocab }, output, frame_batch, seq, hidden)
        }
        other => {
            return Err(fail(
                UNSUPPORTED,
                format!("npu: graph_input {other} is not INPUT_TOKEN_IDS or INPUT_EMBEDDINGS"),
            ));
        }
    };

    // Initialize now: the weights' move to the device is the load.
    blob::timed("initialize", || initialize(ctx, graph, props.init_stage_required))?;

    let compute_dtype = session_dtype(desc.compute_dtype, output.precision)?;
    Ok(Model { ctx, graph, run: Mutex::new(()), inputs, output, frame_batch, seq, hidden, compute_dtype, graph_format })
}

fn initialize(ctx: &Context, graph: Handle, stage: u32) -> Res<()> {
    match stage {
        ze::GRAPH_STAGE_INITIALIZE => {
            let init = ctx.ext.graph_initialize().ok_or_else(|| {
                fail(RUNTIME, "npu: the graph asks for pfnGraphInitialize and the driver's table has none")
            })?;
            ze_res("pfnGraphInitialize", unsafe { init(graph) })
        }
        ze::GRAPH_STAGE_COMMAND_LIST_INITIALIZE => {
            let init = ctx.ext.append_graph_initialize().ok_or_else(|| {
                fail(RUNTIME, "npu: the graph asks for pfnAppendGraphInitialize and the driver's table has none")
            })?;
            ctx.with_list(|list| {
                ze_res("pfnAppendGraphInitialize", unsafe {
                    init(list, graph, std::ptr::null_mut(), 0, std::ptr::null_mut())
                })?;
                wait_for_device(ctx.api, list, ctx.ordinal, "pfnAppendGraphInitialize")
            })
        }
        s => Err(fail(RUNTIME, format!("npu: the graph asks for init stage {s}, which this backend does not know"))),
    }
}

/// # Safety
/// `model` is a model this backend loaded, released once.
pub(crate) unsafe extern "C" fn model_release(model: *mut c_void) {
    quietly(|| {
        let m = unsafe { Box::from_raw(model as *mut Model) };
        if let Some(destroy) = m.ctx().ext.destroy() {
            unsafe { destroy(m.graph) };
        }
    });
}
