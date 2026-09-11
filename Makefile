# Convenience targets for inferstream model artifact management.
# No target requires an interpreter other than the Rust toolchain + Make.
#
#   make test                               # cargo test --workspace
#   make test-fetch                         # fetch-crate unit tests (offline)
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
#   make verify-embeddings-intel            # offline SHA-256 of exported OVMS IR
#   make list-embeddings-intel
#
# Fetch / verify are `cargo run -p inferstream-fetch`. Weights stay out of git.
# OVMS IR *export* is a one-off in contrib/offline-once/ (not invoked here).
# `make fetch-llms` / smoke do not need python3.

CARGO ?= cargo
FETCH := $(CARGO) run -q -p inferstream-fetch --
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
	verify-embeddings-intel list-embeddings-intel \
	setup-sycl build-intel-sycl

test:
	$(CARGO) test --workspace

test-fetch:
	$(CARGO) test -p inferstream-fetch

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
