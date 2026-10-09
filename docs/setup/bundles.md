# Bundles

A bundle is a directory: a manifest, the model's weights and artifacts,
its tokenizer, and the reference vectors every backend is checked
against, each file pinned by size and SHA-256 ([../bundle.md](../bundle.md)).
The library loads a bundle and never fetches or converts anything.
Bundles go under `models/` in a checkout, which git ignores.

## Prebuilt bundles

The repository's GitHub releases page carries sealed bundles as
prereleases, each split into parts under 90 MiB:

| Release tag | Bundles | Artifacts |
|---|---|---|
| `minilm-npu-token-ids-f16-febfed530b66` | all-MiniLM-L6-v2 | `weights-f32`, `onnx-f32`, `onnx-f16`, `openvino-f16`, `openvino-embeddings-f16`; no HEF |
| `bge-npu-token-ids-f16-ab1d3d9` | bge-small-en-v1.5, bge-base-en-v1.5, bge-large-en-v1.5 | `weights-f32`, `onnx-f32`, `onnx-f16`, `openvino-f16` |

`weights-f32` runs on the CPU, CUDA, Intel GPU and Metal backends;
`openvino-f16` on the Intel NPU. Each release's notes list every part's
SHA-256 and the rejoined archive's. For the MiniLM bundle:

```
mkdir -p models/download models/bundles/all-minilm-l6-v2
cd models/download
tag=minilm-npu-token-ids-f16-febfed530b66
for part in z01 z02 zip; do
    curl -fLO "https://github.com/ai-pipestream/turboembed/releases/download/$tag/$tag.$part"
done
sha256sum $tag.*                       # compare with the release notes
zip -s 0 $tag.zip --out minilm-full.zip
unzip -q minilm-full.zip -d ../bundles/all-minilm-l6-v2
cd ../..
cargo run --release -p turbo-bundle -- verify models/bundles/all-minilm-l6-v2
```

Keep every part in one directory before `zip -s 0`. The BGE archives
are named `<model>-npu-sealed.z01` and so on, and unzip into a directory
of that name (`bge-small-en-v1.5-npu-sealed/`), which is the bundle
directory. `verify` opens the
bundle as the library does, checks each file's size and hash, refuses
any file the manifest does not list, and checks the reference vectors.
Then run conformance on the machine's device (each setup's page).

## Making a bundle

`turbo-bundle` makes a bundle from a recipe in `bundle/recipes/`: it
fetches the model's files from Hugging Face at the commit the recipe
pins, runs the upstream pipeline in a pinned container to write the
reference vectors, converts the artifacts the recipe asks for in the
same container, and seals and verifies the result. Python runs only
inside the containers; `turbo-bundle` itself is Rust.

### Prerequisites

- Docker, with the user able to reach the daemon. The tool calls the
  `docker` command (Docker Desktop on macOS and Windows works).
- Network access to huggingface.co for the fetch.
- For a HEF: an x86_64 machine and Hailo's Dataflow Compiler wheel
  ([hailo-10h.md](hailo-10h.md)).

```
scripts/setup/bundle-tool.sh                   # check
scripts/setup/bundle-tool.sh --install         # build the reference image and print the pin
scripts/setup/bundle-tool.sh --hailo           # check for the Dataflow Compiler too
```

### The reference image

The recipes pin the reference image by id, as
`turbo-reference@sha256:<id>` in `manifest.reference.produced_by.container`,
so a bundle records exactly which image wrote its reference. An image
built on another machine has another id, so build it and pin your own:

```
docker build -t turbo-reference bundle/reference
docker image inspect --format '{{.Id}}' turbo-reference
```

`bundle/reference/Dockerfile` installs Python 3.12 and the packages in
`bundle/reference/requirements.txt` (sentence-transformers,
transformers, CPU torch, onnx, OpenVINO and the rest, each at a fixed
version). Put the printed id into a copy of the recipe:

```
mkdir -p models/recipes
cp bundle/recipes/bge-small-en-v1.5.json models/recipes/
# edit models/recipes/bge-small-en-v1.5.json:
#   "container": "turbo-reference@sha256:<the id printed above>"
```

The recipe's `local` files (the MiniLM calibration texts) are found
beside the recipe, so copy `bundle/recipes/all-minilm-l6-v2.calibration.jsonl`
with the MiniLM recipe.

### Make

```
cargo run --release -p turbo-bundle -- make \
    models/recipes/bge-small-en-v1.5.json models/upstream/bge-small-en-v1.5 models/bundles/bge-small-en-v1.5
```

`make` refuses a bundle directory that already holds a manifest. Every
container runs with no network. Each conversion runs twice, and the
manifest records whether the two runs gave the same bytes.

The MiniLM recipe also lists the Hailo-10H HEF, whose container is
built from the Dataflow Compiler. Without that image `make` stops when
it reaches the HEF, after the other conversions. To make the MiniLM
bundle without a HEF, delete the `hef-hailo10h-s128` artifact from your
copy of the recipe.

### Other commands

| Command | What it does |
|---|---|
| `turbo-bundle fetch <recipe> <upstream-dir>` | Fetch only. A file already there is kept when its hash matches the pin. Writes `turbo-fetch.json`, where the files came from and their hashes. A model with terms beyond its licence needs `--accept-terms`, here and in `make`. |
| `turbo-bundle catalogue` | The models `make` and `fetch` take by name instead of a recipe path, with their licences. |
| `turbo-bundle reference <recipe> <upstream-dir> <bundle-dir>` | Everything `make` does after the fetch: the offline path, for an upstream directory filled another way. |
| `turbo-bundle seal <recipe> <upstream-dir> <bundle-dir>` | Stage and seal with no container, from a reference and conversion reports a container already wrote into the bundle directory (see [../../bundle/README.md](../../bundle/README.md)). |
| `turbo-bundle verify <bundle-dir>` | Check a bundle as the library loads it. |

`fetch` and `make` are the only commands that use the network.

## Recipes

| Recipe | Model | `max_seq` | Artifacts besides `weights-f32` and `onnx-f32` |
|---|---|---|---|
| `all-minilm-l6-v2.json` | sentence-transformers/all-MiniLM-L6-v2 | 256 | `onnx-f16`, `openvino-f16`, `openvino-embeddings-f16` (NPU), `hef-hailo10h-s128` (Hailo-10H) |
| `bge-small-en-v1.5.json` | BAAI/bge-small-en-v1.5 | 512 | `onnx-f16`, `openvino-f16` (NPU) |
| `bge-base-en-v1.5.json` | BAAI/bge-base-en-v1.5 | 512 | `onnx-f16`, `openvino-f16` (NPU) |
| `bge-large-en-v1.5.json` | BAAI/bge-large-en-v1.5 | 512 | `onnx-f16`, `openvino-f16` (NPU) |
| `bge-m3.json` | BAAI/bge-m3 | 512 | `onnx-f16`; the weights are converted from the PyTorch checkpoint |
| `bge-m3-8192.json` | BAAI/bge-m3 | 8192 | the same, at `max_batch` 8 |

The ONNX files are for the reference programs and the converters; the
library never executes them.

### Model2Vec's static models

`bundle/recipes/potion/` pins Model2Vec's nine potion models, and
`make` and `fetch` take each by name:

```
cargo run --release -p turbo-bundle -- catalogue
cargo run --release -p turbo-bundle -- make \
    minishlab/potion-base-8M models/upstream/potion-base-8M models/bundles/potion-base-8M
```

Each is fetched from its Hugging Face repository at a pinned commit, each
file checked against its SHA-256, only when a command names it. The
bundle computes what Model2Vec computes, to the bit (docs/static.md).
`potion-retrieval-32M` was fine-tuned on MS MARCO, whose terms allow
non-commercial use only, so it is fetched only with `--accept-terms`.
