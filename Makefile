# Convenience targets for inferstream model artifact management.
#
#   make fetch-embeddings                       # all nvidia ONNX embedding aliases
#   make fetch-embeddings ALIASES=minilm,mpnet  # a subset
#   make verify-embeddings [ALIASES=...]        # offline SHA-256 check, no network
#   make list-embeddings                        # aliases, repos, pinned revisions
#   make update-embedding-manifest              # maintainers: re-pin + re-hash
#   make test-fetch                             # unit tests for the fetch tooling
#
# Intel (OVMS) equivalents — export OpenVINO IR + tokenizers at pinned HF
# revisions and verify SHA-256 against models/manifests/ovms-embeddings.json
# (scripts/export_ovms_embeddings.py; needs a venv with torch/transformers/
# openvino/openvino-tokenizers — see the script docstring):
#
#   make fetch-embeddings-intel [ALIASES=...] [OVMS_DIR=...] [HF_TOK_DIR=...]
#   make verify-embeddings-intel [ALIASES=...]
#   make list-embeddings-intel
#   make update-embedding-manifest-intel        # maintainers: re-pin + re-export
#
# Everything is driven against committed manifests (pinned HF revisions +
# SHA-256 per file). See docs/fetching-models.md.

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
	fetch-embeddings-intel verify-embeddings-intel list-embeddings-intel \
	update-embedding-manifest-intel

fetch-embeddings:
	$(PYTHON) scripts/fetch_models.py $(ALIAS_ARGS)

verify-embeddings:
	$(PYTHON) scripts/fetch_models.py $(ALIAS_ARGS) --verify-only

list-embeddings:
	$(PYTHON) scripts/fetch_models.py --list

update-embedding-manifest:
	$(PYTHON) scripts/fetch_models.py $(ALIAS_ARGS) --update-manifest

test-fetch:
	$(PYTHON) -m unittest discover -s scripts -p 'test_*.py' -v

fetch-embeddings-intel:
	$(PYTHON) scripts/export_ovms_embeddings.py $(ALIAS_ARGS) $(INTEL_ARGS)

verify-embeddings-intel:
	$(PYTHON) scripts/export_ovms_embeddings.py $(ALIAS_ARGS) --verify-only $(INTEL_ARGS)

list-embeddings-intel:
	$(PYTHON) scripts/export_ovms_embeddings.py --list

update-embedding-manifest-intel:
	$(PYTHON) scripts/export_ovms_embeddings.py $(ALIAS_ARGS) --update-manifest $(INTEL_ARGS)
