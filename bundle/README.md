# turbo-bundle

Makes a model bundle (`docs/bundle.md`) from a recipe, and checks a
bundle the way a machine loading it does.

A recipe is the manifest without `files`, plus the upstream files to
fetch at `model.source.commit`. `recipes/` holds one per model.

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
   paths.
3. **reference**: the upstream pipeline runs in the pinned container,
   with no network, on the fetched files: fp32 on CPU, one text at a time.
   It writes the reference ids and vectors, and reports what ran. Python
   runs here and in the next step, in this container, and nowhere else.
4. **convert**: each artifact whose `produced_by` in the recipe names
   only `from` is made from that artifact in the same pinned container,
   with no network, twice: today the F16 ONNX file, `onnx_f16.py`, run
   with `--entrypoint python`. An image built before `onnx_f16.py` was
   added to it cannot; build it again and pin the new id.
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

`turbo-bundle verify <dir>` runs step 5 alone.
