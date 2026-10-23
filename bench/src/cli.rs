use std::collections::BTreeMap;

use turbo_bench::Result;

pub(crate) const USAGE: &str = "\
usage:
  turbo-bench record --bundle <dir> [options]
      measure the library on one device, run the reference programs on
      the same token rows, and write the record; a bundle under
      testdata/ is refused
  turbo-bench ranges --bundle <dir> [--device <d>] [--precision <p>]
                     [--batch <n>] [--seq <n>] [--chunk <n>] [--warmup <n>]
                     [--iterations <n>] [--repo <dir>] [--out <dir>]
      on dense rows (default 32 or the model's max_batch if smaller, and 256
      or its max_seq), time one vector per row, one per chunk of --chunk
      text tokens (default 64) by ranges, and one per chunk embedded as a
      row of its own; write the record to <repo>/benchmarks/ranges
  turbo-bench qa --bundle <dir> --questions <squad.json> [--device <d>]
                 [--precision <p>] [--batch <n>] [--max-batch <n>]
                 [--per-bin <n>] [--warmup <n>] [--iterations <n>]
                 [--repo <dir>] [--out <dir>]
      extractive QA questions per second on one device, by the context's
      length in tokens (bins from 0, 128, 256 and 384), up to --per-bin
      questions each (default 256), --batch questions per call (default
      8), --max-batch windows per run (default the model's), precision
      exact by default; write the record to <repo>/benchmarks/qa
  turbo-bench rerank --bundle <dir> --questions <squad.json>
                     (--tei-image <name@sha256:..> --tei-model <dir> | --no-tei)
                     [--device <d>] [--precision <p>] [--requests <n>]
                     [--passages <n>] [--warmup <n>] [--iterations <n>]
                     [--repo <dir>] [--out <dir>]
      cross-encoder pairs per second on one device: --requests queries
      (default 64), each with --passages passages that fit one window
      (default 16), and TEI's /rerank on the same pairs, in float32;
      write the record to <repo>/benchmarks/rerank
  turbo-bench check <record.json>...
      parse records and say whether each backs SUPPORTED for its cell

record options:
  --device <index|backend>     the runtime device (default cpu)
  --precision <p>              model, fastest or exact (default model)
  --batch <n>, --seq <n>       the rows' shape (default: 32 or the model's
                               max_batch if smaller; the longest reference
                               case that fits). npu refuses that default:
                               both must be the loaded artifact's fixed
                               shape. The token-id MiniLM seal is
                               --batch 1 --seq 128
  --rows <kind>                mixed: the reference cases that fit seq,
                               cycled and padded. A frame of batch 1 (the
                               npu cell) is one [1, seq] frame per case,
                               not case 0 alone. dense: every row a case
                               of at least seq tokens cut to seq, so all
                               batch x seq tokens are live (default mixed)
  --cpus <list>                processors to run on, as 0-15 or 0-7,16-23:
                               the tool is pinned to them with a library
                               thread per CPU, and TEI's container is given
                               them, MKL a thread per physical core and
                               rayon one per CPU (Linux; default: unpinned)
  --warmup <n>                 untimed runs first (default 20)
  --iterations <n>             timed runs (default 200)
  --repo <dir>                 the git working tree the library was built
                               from: clean, pushed, and at the commit this
                               tool was built from (default: the one this
                               tool was built in)
  --out <dir>                  where the record goes (default
                               <repo>/benchmarks/records)
  --work <dir>                 scratch for reference inputs (default: the
                               system's temporary directory)
  --tei-image <name@sha256:..> text-embeddings-inference, pinned
  --tei-bin <path>             TEI's router built natively, run in place of
                               an image (metal, which no container reaches)
  --tei-model <dir>            the model in the upstream layout
  --no-tei                     record that TEI was not run
  --tensorrt-image <name@sha256:..>
                               NVIDIA's TensorRT container, pinned (cuda)
  --tensorrt-bin <path>        trtexec installed on the machine, run in place
                               of an image (a Jetson, whose TensorRT is
                               JetPack's)
  --tensorrt-inputs <a,b[,c]>  the ONNX inputs for ids, mask and types
                               (default input_ids,attention_mask,token_type_ids)
  --tensorrt-input-dtype <t>   int64 or int32 (default int64)
  --tensorrt-warmup-ms <n>     trtexec --warmUp (default 1000)
  --trtexec <path>             trtexec in the image (default trtexec)
  --no-tensorrt                record that TensorRT was not run
  --openvino-image <name@sha256:..>
                               an OpenVINO container with benchmark_app,
                               pinned (levelzero, and npu on Linux)
  --openvino-bin <path>        benchmark_app installed on the machine, run
                               in place of an image (npu, when the container
                               cannot see the NPU driver)
  --openvino-accel <path>      the NPU device node handed to the container
                               (default /dev/accel/accel0; npu with
                               --openvino-image). Several NPUs are refused:
                               benchmark_app's -d NPU is OpenVINO's first,
                               and there is no per-device index
  --openvino-inputs <a,b[,c]>  the ONNX inputs for ids, mask and types
                               (default input_ids,attention_mask,token_type_ids)
  --openvino-input-dtype <t>   int64 or int32 (default int64)
  --benchmark-app <path>       benchmark_app in the image (default
                               benchmark_app)
  --no-openvino                record that OpenVINO was not run";

/// `--name value` pairs and bare `--flag`s.
pub(crate) struct Opts(pub(crate) BTreeMap<String, Option<String>>);

const FLAGS: [&str; 3] = ["--no-tei", "--no-tensorrt", "--no-openvino"];

impl Opts {
    pub(crate) fn parse(args: &[String]) -> Result<Opts> {
        let mut m = BTreeMap::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if !a.starts_with("--") {
                return Err(format!("{a}: not an option\n{USAGE}"));
            }
            let v = if FLAGS.contains(&a.as_str()) {
                None
            } else {
                Some(it.next().ok_or_else(|| format!("{a} needs a value"))?.clone())
            };
            if m.insert(a.clone(), v).is_some() {
                return Err(format!("{a} is given twice"));
            }
        }
        Ok(Opts(m))
    }

    pub(crate) fn take(&mut self, name: &str) -> Option<String> {
        self.0.remove(name).flatten()
    }

    pub(crate) fn flag(&mut self, name: &str) -> bool {
        self.0.remove(name).is_some()
    }

    pub(crate) fn number(&mut self, name: &str) -> Result<Option<u32>> {
        self.take(name).map(|v| v.parse().map_err(|_| format!("{name} {v}: not a number"))).transpose()
    }
}
