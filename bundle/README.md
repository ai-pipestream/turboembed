# turbo-bundle

Makes a model bundle (`docs/bundle.md`) from a recipe, and checks a
bundle the way a machine loading it does.

A recipe is the manifest without `files`, plus the upstream files to
fetch at `model.source.commit`, and `local`: files that come with the
recipe, beside it in `recipes/`, each pinned by SHA-256. `recipes/`
holds one per model.

```
docker build -t turbo-reference bundle/reference
docker image inspect --format '{{.Id}}' turbo-reference
# put turbo-reference@<that id> in the recipe's reference.produced_by.container
cargo run -p turbo-bundle -- make bundle/recipes/all-minilm-l6-v2.json upstream/ all-minilm-l6-v2/
```

`make` runs, in order:

1. **fetch**: every upstream file at the commit, checked against its
   pinned SHA-256 where the recipe gives one. Each file's hash is printed.
2. **stage**: the files the bundle carries are copied to their bundle
   paths: the fetched ones, and the recipe's `local` files, each checked
   against its pinned hash.
3. **reference**: the upstream pipeline runs in the pinned container,
   with no network, on the fetched files: fp32 on CPU, one text at a time.
   It writes the reference ids and vectors, and reports what ran. Python
   runs here and in the next step, in the pinned containers. `seal`
   (below) is the one path that adopts a report the host script already
   wrote, and it does not run Python.
4. **convert**: each artifact whose `produced_by` the recipe gives is
   made from the artifact it names, with no network, twice:
   - an F16 ONNX file (`produced_by` names only `from`), in the reference
     container, by `onnx_f16.py`, run with `--entrypoint python`. An image
     built before `onnx_f16.py` was added to it cannot; build it again and
     pin the new id.
   - raw weights as safetensors (`produced_by` names only `upstream`, a
     PyTorch checkpoint among the upstream files, fetched and not carried),
     in the reference container, by `bin_to_safetensors.py`: every tensor
     as stored, for a model whose repository ships no safetensors file.
   - a static-shape OpenVINO IR for the npu backend (`produced_by` names
     only `from`), two files, the xml then its weights, in the reference
     container, by `onnx_to_openvino_ir.py`: the export's inputs reshaped
     to the artifact's `fixed_batch` and `fixed_seq`, weights compressed
     to FP16. An image built before that script was added to it cannot;
     build it again and pin the new id (docs/npu.md).
   - a HEF for a Hailo device (`produced_by` names `from`, `container` and
     `inputs`), in the Dataflow Compiler container that `container` pins,
     by `bundle/hailo/hef_compile.py`: the export is cut at the
     word-embedding gather and the attention mask, quantized to 8 bits on
     the calibration texts `inputs` names, and compiled for `target` at a
     frame of `fixed_seq` tokens. The compiler cannot be redistributed, so
     its image is built locally; `bundle/hailo/Dockerfile` says how.
5. **seal**: `files` is filled with each named file's size and SHA-256,
   the reference's and each converted artifact's `produced_by` with what
   its container reported and the image it ran in (and for a conversion,
   whether the two runs gave the same bytes), and the manifest goes through the core's parser before
   it is written.
6. **verify**: the core opens the bundle (loader rules 1 to 5, so its
   tokenizer must give the reference's ids exactly), every listed file is
   checked by size and hash, nothing unlisted is in the directory, and
   every reference vector is finite, non-zero and, when the bundle says
   `NORMALIZE_L2`, of unit length.

`turbo-bundle verify <dir>` runs step 6 alone.

`turbo-bundle seal <recipe.json> <upstream-dir> <bundle-dir>` does
steps 2 and 5 without a container. The reference file and a report
that names its container must already be in the bundle, copied from a
bundle this pin already sealed. An OpenVINO IR the recipe converts
must already be there too, with the report the host script writes
(docs/npu.md); that manifest entry records `container` `host`. A
conversion whose files are absent is left out of the manifest.
