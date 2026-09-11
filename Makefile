# Convenience targets for inferstream model artifact management.
#
#   make fetch-embeddings                       # all nvidia ONNX embedding aliases
#   make fetch-embeddings ALIASES=minilm,mpnet  # a subset
#   make verify-embeddings [ALIASES=...]        # offline SHA-256 check, no network
#   make list-embeddings                        # aliases, repos, pinned revisions
#   make update-embedding-manifest              # maintainers: re-pin + re-hash
#   make test-fetch                             # unit tests for the fetch tooling
#
#   make fetch-llms                             # GGUF + tokenizer.json (qwen-0.5b, qwen-7b)
#   make fetch-llms ALIASES=qwen-0.5b           # smoke-sized 0.5B only (~650 MiB)
#   make verify-llms [ALIASES=...]              # offline SHA-256 check, no network
#   make list-llms
#   make update-llm-manifest                    # maintainers: re-pin + re-hash
#
# Intel OVMS embedding export is an optional offline contrib (Python +
# torch/openvino). Not a Make default and not on the runtime/hot path.
# `make fetch-llms` / smoke do not call it.
#
#   make fetch-embeddings-intel [ALIASES=...] [OVMS_DIR=...] [HF_TOK_DIR=...]
#   make verify-embeddings-intel [ALIASES=...]
#   make list-embeddings-intel
#   make update-embedding-manifest-intel        # maintainers: re-pin + re-export
#
# Everything is driven against committed manifests (pinned HF revisions +
# SHA-256 per file). See docs/fetching-models.md. Weights stay out of git.

# Embedding ONNX fetch still uses the stdlib Python helper (nvidia hosts).
# LLM fetch / verify / list are shell + sha256sum — no python3.
PYTHON ?= python3
ALIASES ?=

comma := ,
empty :=
space := $(empty) $(empty)
ALIAS_ARGS := $(if $(ALIASES),$(subst $(comma),$(space),$(ALIASES)),--all)

OVMS_DIR ?= /work/models/ovms-embedder
HF_TOK_DIR ?= $(HOME)/ovms-models
INTEL_ARGS := --out $(OVMS_DIR) --hf-out $(HF_TOK_DIR)

.PHONY: fetch-embeddings verify-embeddings list-embeddings \
	update-embedding-manifest test-fetch \
	fetch-llms verify-llms list-llms update-llm-manifest \
	fetch-embeddings-intel verify-embeddings-intel list-embeddings-intel \
	update-embedding-manifest-intel \
	setup-sycl build-intel-sycl

fetch-embeddings:
	$(PYTHON) scripts/fetch_models.py $(ALIAS_ARGS)

verify-embeddings:
	$(PYTHON) scripts/fetch_models.py $(ALIAS_ARGS) --verify-only

list-embeddings:
	$(PYTHON) scripts/fetch_models.py --list

update-embedding-manifest:
	$(PYTHON) scripts/fetch_models.py $(ALIAS_ARGS) --update-manifest

test-fetch:
	scripts/test-fetch-llms.sh
	$(PYTHON) -m unittest discover -s scripts -p 'test_*.py' -v

fetch-llms:
	scripts/fetch-llms.sh $(ALIAS_ARGS)

verify-llms:
	scripts/fetch-llms.sh --verify-only $(ALIAS_ARGS)

list-llms:
	scripts/fetch-llms.sh --list

# Re-pin is a maintainer action. The committed manifest is the source of
# truth; this target is intentionally not a Python default path. Use the
# offline optional helper only when deliberately moving pins:
#   python3 scripts/fetch_models.py --llms --update-manifest
update-llm-manifest:
	@echo "error: make update-llm-manifest is not a runtime/default path." >&2
	@echo "Re-pin offline with: python3 scripts/fetch_models.py --llms --update-manifest" >&2
	@echo "(optional contrib; fetch-llms / smoke do not need python3)" >&2
	@exit 1

# Inject ggml-sycl sources into a local llama-cpp-sys-2 checkout so
# GGML_SYCL=ON cmake succeeds. No python3. Needed before llamacpp-sycl.
setup-sycl:
	scripts/setup-llamacpp-sycl.sh

# In-process SYCL binary. icpx drives the rustc link (device images).
# No python3.
build-intel-sycl: setup-sycl
	scripts/build-intel.sh

fetch-embeddings-intel:
	$(PYTHON) scripts/export_ovms_embeddings.py $(ALIAS_ARGS) $(INTEL_ARGS)

verify-embeddings-intel:
	$(PYTHON) scripts/export_ovms_embeddings.py $(ALIAS_ARGS) --verify-only $(INTEL_ARGS)

list-embeddings-intel:
	$(PYTHON) scripts/export_ovms_embeddings.py --list

update-embedding-manifest-intel:
	$(PYTHON) scripts/export_ovms_embeddings.py $(ALIAS_ARGS) --update-manifest $(INTEL_ARGS)
