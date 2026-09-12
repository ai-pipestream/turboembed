# Convenience targets for inferstream model artifact management.
# No target requires an interpreter other than the Rust toolchain + Make.
#
#   make test                               # cargo test --workspace
#   make test-fetch                         # fetch-crate + xtask unit tests
#
#   make fetch-embeddings                   # all nvidia ONNX embedding aliases
#   make fetch-embeddings ALIASES=minilm,mpnet
#   make verify-embeddings [ALIASES=...]    # offline SHA-256 check, no network
#   make list-embeddings                    # aliases, repos, pinned revisions
#   make update-embedding-manifest          # maintainers: re-pin + re-hash
#
#   make fetch-llms                         # GGUF + tokenizer.json
#   make fetch-llms ALIASES=qwen-0.5b
#   make verify-llms [ALIASES=...]
#   make list-llms
#   make update-llm-manifest
#
#   make fetch-mlx                          # Apple native-MLX safetensors
#   make fetch-mlx ALIASES=minilm,qwen-0.5b
#   make verify-mlx [ALIASES=...]
#   make list-mlx
#   make update-mlx-manifest
#
#   make verify-embeddings-intel            # offline SHA-256 of exported OVMS IR
#   make list-embeddings-intel
#
# Embeddings / LLMs / OVMS: `cargo run -p inferstream-fetch`.
# Apple MLX weights: `cargo xtask` (crates/xtask) against models/manifests/mlx.json.
# OVMS IR *export* is a one-off in contrib/offline-once/ (not invoked here).
# Make never invokes python3.

CARGO ?= cargo
FETCH := $(CARGO) run -q -p inferstream-fetch --
CARGO_XTASK ?= cargo xtask
ALIASES ?=

comma := ,
empty :=
space := $(empty) $(empty)
ALIAS_ARGS := $(if $(ALIASES),$(subst $(comma),$(space),$(ALIASES)),--all)

OVMS_DIR ?= /work/models/ovms-embedder
HF_TOK_DIR ?= $(HOME)/ovms-models
INTEL_ARGS := --out $(OVMS_DIR) --hf-out $(HF_TOK_DIR)

.PHONY: test test-fetch \
	fetch-embeddings verify-embeddings list-embeddings \
	update-embedding-manifest \
	fetch-llms verify-llms list-llms update-llm-manifest \
	fetch-mlx verify-mlx list-mlx update-mlx-manifest \
	verify-embeddings-intel list-embeddings-intel \
	update-embedding-manifest-intel \
	setup-sycl build-intel-sycl \
	apple smoke-apple sync-proto

test:
	$(CARGO) test --workspace

test-fetch:
	$(CARGO) test -p inferstream-fetch
	$(CARGO) test -p inferstream-xtask

fetch-embeddings:
	$(FETCH) $(ALIAS_ARGS)

verify-embeddings:
	$(FETCH) $(ALIAS_ARGS) --verify-only

list-embeddings:
	$(FETCH) --list

update-embedding-manifest:
	$(FETCH) $(ALIAS_ARGS) --update-manifest

fetch-llms:
	$(FETCH) --llms $(ALIAS_ARGS)

verify-llms:
	$(FETCH) --llms $(ALIAS_ARGS) --verify-only

list-llms:
	$(FETCH) --llms --list

update-llm-manifest:
	$(FETCH) --llms $(ALIAS_ARGS) --update-manifest

fetch-mlx:
	$(CARGO_XTASK) fetch --mlx $(ALIAS_ARGS)

verify-mlx:
	$(CARGO_XTASK) verify --mlx $(ALIAS_ARGS)

list-mlx:
	$(CARGO_XTASK) list --mlx

update-mlx-manifest:
	$(CARGO_XTASK) update-manifest --mlx $(ALIAS_ARGS)

# Apple serve path is the all-Swift gRPC server (no Rust façade).
sync-proto:
	./scripts/sync-proto.sh --check

apple: sync-proto
	swift build --package-path swift -c release

smoke-apple: apple
	./scripts/smoke-apple.sh

# Inject ggml-sycl sources into a local llama-cpp-sys-2 checkout so
# GGML_SYCL=ON cmake succeeds. No python3. Needed before llamacpp-sycl.
setup-sycl:
	scripts/setup-llamacpp-sycl.sh

# In-process SYCL binary. icpx drives the rustc link (device images).
# No python3.
build-intel-sycl: setup-sycl
	scripts/build-intel.sh

verify-embeddings-intel:
	$(FETCH) --ovms $(ALIAS_ARGS) --verify-only $(INTEL_ARGS)

list-embeddings-intel:
	$(FETCH) --ovms --list

update-embedding-manifest-intel:
	@echo "error: OpenVINO IR re-export is not invoked from Make (no Python)." >&2
	@echo "See contrib/offline-once/; then cargo run -p inferstream-fetch -- --ovms --verify-only." >&2
	@exit 1
