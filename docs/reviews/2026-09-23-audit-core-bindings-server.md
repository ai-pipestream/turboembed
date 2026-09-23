# Audit: core, C ABI, bindings, server, demo, docs (2026-09-23)

Read-only audit of `crates/turbo-abi`, `crates/turbo-capi`,
`crates/turbo-shared`, `crates/turbo-core`, `crates/turbo`,
`crates/turbo-conformance`, `bindings/`, `server/`, `demo/` and `docs/` on
branch `turbo-v2` at commit 7dd68aa, against the mission and the fifteen
rules in `AGENTS.md`. Delivered as written by the auditor. The twelve
findings known before the audit are not repeated (see the companion
audit of the providers for the list).

Ranked from most costly to the philosophy to least.

1. **AUTO picks the mock accelerator ahead of real hardware**
   - Evidence: `crates/turbo/src/lib.rs:20-30` registers the built-in providers (mock and static) unless `no_default_providers` is set. `crates/turbo-core/src/mock.rs:241-242,270-271` always reports two devices, and ordinal 1 is `DeviceKind::Accel`. `runtime.rs:119-124` registers built-ins before any loaded library, and `runtime.rs:246-251` makes AUTO "the first non-CPU device in registry order". Neither Java `Turbo.create` (`Turbo.java:33-45`) nor Swift `Runtime.init` sets the flag.
   - Result: `demo/java/.../Main.java:54-55`, `demo/java-web-spring/.../TurboService.java:171-172` and the Swift demo (`main.swift:46`) resolve AUTO to "Mock accelerator" even when the CUDA or OpenVINO library is loaded. `MetaApiTest.java:34` asserts this.
   - Rules: 4 (the mock serves real selection) and 2 (silent substitution).
   - Cut: remove the mock from the default provider set, so it is present only when a test registers it.

2. **The C ABI conversions are written twice, and the copies already disagree**
   - Evidence: `crates/turbo-core/src/abi_convert.rs:28-150,452-723` already holds `check_size`, `text`, `texts`, `kvs`, `put_str`, the embed, rerank, classify and generate descriptor parsers, a `flag()` helper, and the device-info, capability, tensor-info, stats and span writers. `crates/turbo-capi/src/lib.rs` rewrites every one of them: 122-190, 1002-1016, 1080-1094, 1117-1129, 1394-1455, 428-458, 523-545, 1274-1297, 1191-1215, 1368-1377. It imports only `read_sized`/`write_sized` (line 22), and the hand-written `declared`/`copy_nonoverlapping` block appears 9 times instead of calling `write_sized`.
   - Drift: capi maps a bad `stop` array to field 16 (`lib.rs:1399-1400`); abi_convert does not (`abi_convert.rs:564-565`).
   - More duplicates: `fail`/`boundary` in `plugin_export.rs:67-112` against `capi/lib.rs:43-118`, where the plugin_export version skips the struct-size check. `turbo_status_name` (`capi:275-318`) against `error::status_name` (`error.rs:220-256`).
   - Rules: 1 and 9.
   - Cut: make capi call abi_convert and delete its private copies.

3. **The embed path copies the data 4 to 5 times on the host, counted from Java**
   - Input: `Native.fillText` (`Native.java:52-58`) makes `getBytes` (copy 1), then `MemorySegment.copy` into the arena (copy 2). `arena.allocateFrom(String)` would do one. capi `texts()` builds a `Vec<&str>` (`lib.rs:148-158`). `PluginSession::views` (`plugin.rs:640-642,652-662`) then rebuilds the same `turbo_text[]` the caller passed, and `embed_options_to_abi` rebuilds the `turbo_embed_options` capi just parsed.
   - Output: `turbo_result_read` copies into the arena (copy 3). A device output whose buffer is larger than its logical size goes through a full-size scratch `Vec` and a second copy (`handles.rs:826-832`). No stats counter records either copy. `readFloats`/`readInts`/`readBytes` then copy `toArray` into the heap (copy 4, `Result.java:128-159`). The web demo splits rows with `System.arraycopy` (copy 5, `TurboService.java:333-339`).
   - Swift: `readFloats` copies again with `Array(out[0..<n])` (`PipestreamTurbo.swift:562,578`).
   - Per run, each output also allocates an `Arc<str>` name, a `to_vec` shape and an `Arc<PluginBuffer>` (`plugin.rs:769-781`).
   - Rule 3 (copies that are not reported).
   - Cut: pass the caller's `turbo_text` array and options through untouched, and read into the heap array with a critical, heap-access downcall.

4. **The server re-tokenizes every input twice and swallows the errors**
   - Evidence: `server/src/engine.rs:436-438`. `longest()` tokenizes each text with `format!("{prefix}{t}")` per text, then `tokens` tokenizes every text again, with `unwrap_or(0)` turning a tokenizer error into 0, and without the prefix, so the prompt_tokens count is wrong. The provider then tokenizes a third time.
   - Missing tokenizer: when there is none it falls back to the longest bucket and reports only via `eprintln` (`engine.rs:182-190`).
   - Rules: 2 and 3 (host stage repeated, fallback not reported).
   - Cut: report the token count from the provider's write and remove the server-side counting.

5. **The server adds semantics the core does not have**
   - Rerank ranking: the server strips `top_n`/`return_sorted` and ranks on the host (`engine.rs:524,549-553`). It still calls `validate_rerank` with the caller's `top_n` (line 504), so a provider without `TURBO_CAP_OPT_TOP_N` is refused for a feature the server then does itself.
   - Invented defaults: buckets 1, 8, max (`engine.rs:192-197`). Model name taken from the part of `model_id` after the last `/` (179-181).
   - Alias tables: `true`/`false` for truncate and normalize, `passage` for document (`engine.rs:321-365`). `param_str` turns any bool or int param into a string, so a JSON `true` means "right" (`oip.rs:330-332`).
   - Rules: 6 and 8, plus section 0 ("adds no semantics").
   - Cut: delete the aliases and the host ranking, and take buckets from the bundle.

6. **Chunking is quadratic in tokenizer calls, masks errors, and throws its tokens away**
   - Cost: `chunker.rs:422-431` re-tokenizes the growing span for every unit it adds, and units fall to single words when a sentence is over budget (363-373).
   - Errors: `Tokenizer::count_tokens` maps any error to `usize::MAX` (`tokenizer.rs:443-447`), which then surfaces as `NoProgress` and is mapped to `TURBO_E_CAPACITY` (`capi:1920-1923`).
   - Dead config: `sentence_boundaries` is hard-coded `true` (`capi:1918`) and `source` is always `""`.
   - Waste: the plan returns byte offsets only, so the text is tokenized again at `write_text`.
   - Rules: 2 and 3, and "tasks, not operations".
   - Cut: count incrementally over one encoding, and make chunk-then-embed one task that reuses the ids.

7. **Capability bits and options with no code behind them**
   - `TURBO_CAP_OPT_OUTPUT_DTYPE`, `OPT_GEN_TOOLS` and `OPT_GEN_N` are never set by any provider or the mock, yet they have ABI fields, parsers, and Java/Swift fields (`EmbedOptions.java:17`).
   - `TURBO_CAP_ASYNC`, `DMABUF` and `DEVICE_TOKENIZE` are referenced nowhere but the header and the display tables. `turbo_session_submit` does not exist (`architecture.md:299-303`).
   - `DEVICE_RESULT`, `EXTERNAL_QUEUE`, `UNIFIED_MEMORY` and `DEVICE_POSTPROCESS` are set by providers and never checked against the placement the result actually reports.
   - Rules: 2 and 9.
   - Cut: delete the unimplemented bits and fields until a provider implements them.

8. **The Java web demo is a second OIP server**
   - Evidence: `demo/java-web-spring/.../oip/OipController.java`, `Oip.java` and `Params.java` (770 lines) re-implement OIP v2 REST and its option parsing. There is also a separate `/api/v1` surface with its own DTOs. The class comment (lines 42-43) says "a separate Rust server is planned", but `server/` exists.
   - Rules: 8 and 9, plus section 0 (the front end is a projection).
   - Cut: point the demo page at Inferstream and delete the Java OIP and DTO layer.

9. **Live tests pass when skipped, and CI runs none of them**
   - Evidence: `let Some(..) = setup() else { return }` (`live_embed.rs:90` and every live case) reports a pass when `TURBO_LIVE_*` is unset. Every CI binding and conformance job runs against the mock only (`docs/bindings.md:50-57`).
   - The expected values are hard-wired: dim 384, `max_seq.min(256)`, and ORT CUDA golden vectors (`live_embed.rs:21,68-71`).
   - Rule 7.
   - Cut: mark the live cases `#[ignore]` so a skip is visible, and take dim and max_seq from the bundle.

10. **Validation is written three times**
    - The truncate, max_tokens and max_seq block is copied across `validate_embed`, `validate_rerank` and `validate_classify` (`handles.rs:375-510`), and `check_budget` (576-585) repeats the max_seq test.
    - The `0 => false, 1 => true` flag parse appears 7 times in capi (1080-1094, 1122-1126, 1422-1431, 1846-1850, 1882-1886).
    - `validate_generate` mixes named field constants with magic numbers 3, 5-8, 15, 22 and 23 (`handles.rs:334-360`).
    - The server re-runs the kind check and `validate_*` before the core runs them again (`engine.rs:420-430,489-504`).
    - `check_version` is duplicated in `grpc.rs:302` and `http.rs:203` and has drifted: gRPC accepts an empty version, REST does not.
    - Rule 9.
    - Cut: one helper per concern, called once.

11. **The server copies at each layer**
    - REST embed: provider to `Vec<u8>` to `Vec<f32>` (`engine.rs:839-845`) to per-row `to_vec` (466) to `flatten().collect()` (`oip.rs:506`).
    - gRPC clones every input tensor (`grpc.rs:94-101`, the `request_from` borrow) and every output (`grpc.rs:166-176`).
    - `String::from_utf8(b.clone())` for each input (`oip.rs:320`).
    - Rule 3.
    - Cut: take requests by value and read results straight into the response buffer.

12. **Docs and README claims the code contradicts**
    - README:14 says AUTO takes "the first accelerator that offers the task". `DeviceSelector` has no task field (`runtime.rs:60-71`); AUTO ignores the task.
    - README:11-12 says every provider is "at or above the vendor's own loop". The same README's table shows Jetson at 0.92x and cuda on the ORT EP.
    - README:131 says the Rust API has "lifetimes enforced by types". `crates/turbo/src/lib.rs:5-8` says single ownership is enforced at run time.
    - `architecture.md:285-288` says no code produces SHARED placements or MTLBuffer handles. The openvino and metal providers do.
    - README:25 quotes "297 tokens/s" with no commit, against the documentation rules.
    - Rule 15.

13. **Private names and paths in committed files**
    - `docs/history/poc/apple-ownership-validation-2026-09-14.md:10` names the owner's SSH host alias.
    - `~/opt/...` paths appear in README:37,40, `server/README.md:32-34,159`, `server/src/main.rs:12-13`, `demo/README.md:18`, `demo/search/README.md:35-42`, `docs/bindings.md:64`, `bindings/java/README.md:50`, `docs/testing.md:405`, `docs/packaging.md:98` and `tokenizer.rs:622`.
    - `/work/reference-code` appears in AGENTS.md:81 and `docs/reference-code.md:7`.
    - Rule 14.
    - Cut: redact the alias, and use `$BUNDLES` placeholders plus an environment variable for the reference root.

14. **Smaller layers**
    - `crates/turbo-shared` defines nothing (`lib.rs:1-8`); capi could carry `crate-type = ["cdylib","staticlib"]` itself.
    - Java and Swift hand-maintain every enum: Swift hard-codes the raw values and checks only their endpoints (`PipestreamTurbo.swift:127-141`), so GPU, IGPU and NPU are unchecked; Swift `stats()` returns the raw C struct, where Java maps -1 to null; Java `Result.output` rejects negative extents (`Result.java:94-99`), which is policy the ABI does not state; Java `placement()` and `outputCount()` each make their own `get_info` call, and every read makes an extra `output_info` call.
    - Rule 9.

15. **The README lists Python among the bindings**
    - README:136 lists `demo/python` in the bindings table. Only a demo is allowed to be Python, so the table overstates it.
    - Rule 13 (a presentation problem, not a code problem).

Not reached: a line-by-line read of `mock.rs`, `bundle.rs`, `bpe.rs`, `wordpiece.rs`, `buffer.rs` and `plugin_export.rs` past line 115; the OpenAI routes in `http.rs`; the Java tokenizer and generation classes; Swift generation; whether the conformance tests assert anything meaningful (`honesty_*`, `capability_*`); the Android demo and `demo/rag`; `docs/status.md`, `docs/providers.md` and `docs/testing.md` claim by claim; the PLAN body past section 0.
