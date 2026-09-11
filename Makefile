# Convenience targets for inferstream model artifact management.
#
#   make fetch-embeddings                       # all nvidia ONNX embedding aliases
#   make fetch-embeddings ALIASES=minilm,mpnet  # a subset
#   make verify-embeddings [ALIASES=...]        # offline SHA-256 check, no network
#   make list-embeddings                        # aliases, repos, pinned revisions
#   make update-embedding-manifest              # maintainers: re-pin + re-hash
#   make test-fetch                             # unit tests for crates/xtask
#
#   make fetch-llms                             # GGUF + tokenizer.json (qwen-0.5b, qwen-7b)
#   make fetch-llms ALIASES=qwen-0.5b           # smoke-sized 0.5B only
#   make verify-llms [ALIASES=...]
#   make list-llms
#   make update-llm-manifest
#
#   make fetch-mlx                              # Apple MLX safetensors (native engine)
#   make fetch-mlx ALIASES=minilm,qwen-0.5b
#   make verify-mlx [ALIASES=...]
#   make list-mlx
#   make update-mlx-manifest
#
# Intel OVMS IR is pre-exported (OpenVINO's own toolchain). xtask only
# verifies SHA-256 against models/manifests/ovms-embeddings.json:
#
#   make verify-embeddings-intel [ALIASES=...] [OVMS_DIR=...]
#   make list-embeddings-intel
#
# Everything is driven against committed JSON manifests (pinned HF revisions
# + SHA-256 per file). See docs/fetching-models.md. Weights stay out of git.
# Make never invokes python3.

CARGO_XTASK ?= cargo xtask
ALIASES ?=

comma := ,
empty :=
space := $(empty) $(empty)
ALIAS_ARGS := $(if $(ALIASES),$(subst $(comma),$(space),$(ALIASES)),--all)

OVMS_DIR ?= /work/models/ovms-embedder

.PHONY: fetch-embeddings verify-embeddings list-embeddings \
	update-embedding-manifest test-fetch \
	fetch-llms verify-llms list-llms update-llm-manifest \
	fetch-mlx verify-mlx list-mlx update-mlx-manifest \
	fetch-embeddings-intel verify-embeddings-intel list-embeddings-intel \
	update-embedding-manifest-intel

fetch-embeddings:
	$(CARGO_XTASK) fetch --embeddings $(ALIAS_ARGS)

verify-embeddings:
	$(CARGO_XTASK) verify --embeddings $(ALIAS_ARGS)

list-embeddings:
	$(CARGO_XTASK) list --embeddings

update-embedding-manifest:
	$(CARGO_XTASK) update-manifest --embeddings $(ALIAS_ARGS)

test-fetch:
	cargo test -p inferstream-xtask

fetch-llms:
	$(CARGO_XTASK) fetch --llms $(ALIAS_ARGS)

verify-llms:
	$(CARGO_XTASK) verify --llms $(ALIAS_ARGS)

list-llms:
	$(CARGO_XTASK) list --llms

update-llm-manifest:
	$(CARGO_XTASK) update-manifest --llms $(ALIAS_ARGS)

fetch-mlx:
	$(CARGO_XTASK) fetch --mlx $(ALIAS_ARGS)

verify-mlx:
	$(CARGO_XTASK) verify --mlx $(ALIAS_ARGS)

list-mlx:
	$(CARGO_XTASK) list --mlx

update-mlx-manifest:
	$(CARGO_XTASK) update-manifest --mlx $(ALIAS_ARGS)

fetch-embeddings-intel:
	@echo "Intel OVMS IR is not downloaded by xtask; verify pre-exported artifacts:"
	$(CARGO_XTASK) verify --ovms $(ALIAS_ARGS) --out $(OVMS_DIR)

verify-embeddings-intel:
	$(CARGO_XTASK) verify --ovms $(ALIAS_ARGS) --out $(OVMS_DIR)

list-embeddings-intel:
	$(CARGO_XTASK) list --ovms

update-embedding-manifest-intel:
	@echo "error: OpenVINO IR re-export is not invoked from Make (no Python)." >&2
	@echo "Use OpenVINO's official conversion tools, then cargo xtask verify --ovms." >&2
	@exit 1
