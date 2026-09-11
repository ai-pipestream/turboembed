# Convenience targets for inferstream model artifact management.
#
#   make fetch-embeddings                       # all nvidia ONNX embedding aliases
#   make fetch-embeddings ALIASES=minilm,mpnet  # a subset
#   make verify-embeddings [ALIASES=...]        # offline SHA-256 check, no network
#   make list-embeddings                        # aliases, repos, pinned revisions
#   make update-embedding-manifest              # maintainers: re-pin + re-hash
#   make test-fetch                             # unit tests for the fetch tooling
#
# Everything is driven by scripts/fetch_models.py against the committed
# manifest models/manifests/embeddings.json (pinned HF revisions + SHA-256
# per file). See docs/fetching-models.md.

PYTHON ?= python3
ALIASES ?=

comma := ,
empty :=
space := $(empty) $(empty)
ALIAS_ARGS := $(if $(ALIASES),$(subst $(comma),$(space),$(ALIASES)),--all)

.PHONY: fetch-embeddings verify-embeddings list-embeddings \
	update-embedding-manifest test-fetch

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
