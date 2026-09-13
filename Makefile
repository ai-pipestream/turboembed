# Convenience targets for inferstream model artifact management.
# No target requires an interpreter other than the Rust toolchain + Make.
#
#   make test                               # cargo test --workspace
#   make test-fetch                         # fetch-crate + xtask unit tests
#   make test-turboembed-nvidia             # live ORT CUDA C ABI MiniLM (ignored)
#   make e2e-nvidia / e2e-intel / e2e-apple # live harness (FETCH=1 by default)
#   make e2e-nvidia FETCH=0                 # skip auto-download (CI / already fetched)
#   make e2e-nvidia FETCH=all               # every matrix alias (includes qwen-7b)
#   make e2e-nvidia FETCH=serve             # aliases in config/nvidia.toml serve
#   make e2e-nvidia FETCH_CORPUS=1          # also pull Tiny Shakespeare + STS
#   make e2e-all                            # each arch whose *_ADDR is set
#   make fetch-e2e-nvidia                   # ensure artifacts only (no gRPC)
#   make fetch-corpus                       # SHA-pinned soak/STS text (optional)
#   make e2e-parity                         # cross-arch cosine when *_ADDR set
#   make e2e-parity-goldens TARGET=nvidia WRITE=1
#   make e2e-drift                          # popular models × arches (same floors)
#   make turboembed-stub                    # C++ ABI stub (native/turboembed)
#   make test-turboembed                    # Rust crate ABI smoke
#   make test-turboembed-intel              # --features genai; TextEmbeddingPipeline on CPU and GPU; NPU create fails loud if missing
#   make test-turboembed-apple              # Mac: Metal create lists minilm + goldens receipt
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
#   make fetch-ov-genai                     # Intel in-process GenAI OV-format dirs
#   make verify-ov-genai [ALIASES=...]
#   make list-ov-genai
#   make update-ov-genai-manifest
#
# Embeddings / LLMs / OV-GenAI: `cargo run -p inferstream-fetch`.
# Apple MLX weights: `cargo xtask` (crates/xtask) against models/manifests/mlx.json.
# Historical OpenVINO IR export (not serving) lives in contrib/offline-once/.
# Make never invokes python3. OVMS gRPC is out of scope.

CARGO ?= cargo
FETCH := $(CARGO) run -q -p inferstream-fetch --
CARGO_XTASK ?= cargo xtask
ALIASES ?=

comma := ,
empty :=
space := $(empty) $(empty)
ALIAS_ARGS := $(if $(ALIASES),$(subst $(comma),$(space),$(ALIASES)),--all)

.PHONY: test test-fetch test-turboembed-nvidia \
	fetch-embeddings verify-embeddings list-embeddings \
	update-embedding-manifest \
	fetch-llms verify-llms list-llms update-llm-manifest \
	fetch-mlx verify-mlx list-mlx update-mlx-manifest \
	fetch-ov-genai verify-ov-genai list-ov-genai update-ov-genai-manifest \
	setup-sycl build-intel-sycl \
	apple smoke-apple sync-proto \
	e2e e2e-nvidia e2e-intel e2e-apple e2e-all e2e-mock \
	fetch-e2e-nvidia fetch-e2e-intel fetch-e2e-apple fetch-e2e-mock \
	fetch-corpus verify-corpus list-corpus update-corpus-manifest \
	e2e-parity e2e-parity-goldens e2e-drift \
	turboembed-stub test-turboembed test-turboembed-intel test-turboembed-apple

test:
	$(CARGO) test --workspace

test-fetch:
	$(CARGO) test -p inferstream-fetch
	$(CARGO) test -p inferstream-xtask

# Live NVIDIA proof: C ABI embed("minilm", text) via ORT CUDA IoBinding
# and explicit CPU EP. --include-ignored also runs TensorRT MiniLM when
# libnvinfer.so.10 is on LD_LIBRARY_PATH (scripts/fetch-runtime-libs.sh
# nvidia-trt). See docs/turboembed.md.
test-turboembed-nvidia:
	LD_LIBRARY_PATH="$(CURDIR)/.libs/nvidia/lib:$(LD_LIBRARY_PATH)" \
		$(CARGO) test -p turboembed --features ort-cuda -- --include-ignored --nocapture

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
	./scripts/build-apple-metallib.sh

smoke-apple: apple
	./scripts/smoke-apple.sh

fetch-ov-genai:
	$(FETCH) --ov-genai $(ALIAS_ARGS)

verify-ov-genai:
	$(FETCH) --ov-genai $(ALIAS_ARGS) --verify-only

list-ov-genai:
	$(FETCH) --ov-genai --list

update-ov-genai-manifest:
	$(FETCH) --ov-genai $(ALIAS_ARGS) --update-manifest

# Inject ggml-sycl sources into a local llama-cpp-sys-2 checkout so
# GGML_SYCL=ON cmake succeeds. No python3. Needed before llamacpp-sycl.
setup-sycl:
	scripts/setup-llamacpp-sycl.sh

# In-process SYCL binary. icpx drives the rustc link (device images).
# No python3.
build-intel-sycl: setup-sycl
	scripts/build-intel.sh

# Unified E2E harness (crates/e2e). Talks gRPC to an already-running server;
# never starts remote GPUs. See docs/e2e.md.
#
# Lab checkout paths are local to each Machine (README chart). On-box
# default is 127.0.0.1:8461. For a remote worker set
# INFERSTREAM_E2E_{NVIDIA,INTEL,APPLE}_ADDR to that Machine's gRPC host:port.
#
# FETCH=1 (default on worker targets): download missing SHA-256-pinned
# artifacts via inferstream-fetch / cargo xtask fetch --mlx, then run.
# FETCH=0 skips download (CI mock, or hosts that already have weights).
# FETCH=all / FETCH=serve widen the alias set (see docs/e2e.md).
E2E := $(CARGO) run -q -p inferstream-e2e --
INFERSTREAM_E2E_TOKEN ?= change-me
INFERSTREAM_E2E_NVIDIA_ADDR ?=
INFERSTREAM_E2E_INTEL_ADDR ?=
INFERSTREAM_E2E_APPLE_ADDR ?=
FETCH ?= 1
# Optional soak/STS corpus. Default off — CI stays on committed micro fixtures.
FETCH_CORPUS ?= 0

# FETCH=0|false|no|off → no flag
# FETCH=1|true|yes|on  → --fetch
# FETCH=all            → --fetch --fetch-all
# FETCH=serve          → --fetch --fetch-serve
E2E_FETCH_ARGS := $(strip \
	$(if $(filter 0 false no off,$(FETCH)),,\
		$(if $(filter all,$(FETCH)),--fetch --fetch-all,\
			$(if $(filter serve,$(FETCH)),--fetch --fetch-serve,\
				--fetch))))
E2E_CORPUS_ARGS := $(strip \
	$(if $(filter 1 true yes on,$(FETCH_CORPUS)),--fetch-corpus,))

e2e-nvidia:
	$(E2E) --target nvidia --addr $(or $(INFERSTREAM_E2E_NVIDIA_ADDR),$(INFERSTREAM_E2E_ADDR),127.0.0.1:8461) --token "$(INFERSTREAM_E2E_TOKEN)" $(E2E_FETCH_ARGS) $(E2E_CORPUS_ARGS)

e2e-intel:
	$(E2E) --target intel --addr $(or $(INFERSTREAM_E2E_INTEL_ADDR),$(INFERSTREAM_E2E_ADDR),127.0.0.1:8461) --token "$(INFERSTREAM_E2E_TOKEN)" $(E2E_FETCH_ARGS) $(E2E_CORPUS_ARGS)

e2e-apple:
	$(E2E) --target apple --addr $(or $(INFERSTREAM_E2E_APPLE_ADDR),$(INFERSTREAM_E2E_ADDR),127.0.0.1:8461) --token "$(INFERSTREAM_E2E_TOKEN)" $(E2E_FETCH_ARGS) $(E2E_CORPUS_ARGS)

# Local mock under the same logical names (config/e2e-mock.toml must be up).
# Auto-fetch is off: the mock has no on-disk weights.
e2e-mock:
	$(E2E) --target mock --addr $(or $(INFERSTREAM_E2E_ADDR),127.0.0.1:8461) --token "$(INFERSTREAM_E2E_TOKEN)"

# Ensure artifacts only (bring-up / serving). Same FETCH= scope as e2e-*.
fetch-e2e-nvidia:
	$(E2E) --target nvidia --fetch-only $(if $(filter all,$(FETCH)),--fetch-all,) $(if $(filter serve,$(FETCH)),--fetch-serve,)

fetch-e2e-intel:
	$(E2E) --target intel --fetch-only $(if $(filter all,$(FETCH)),--fetch-all,) $(if $(filter serve,$(FETCH)),--fetch-serve,)

fetch-e2e-apple:
	$(E2E) --target apple --fetch-only $(if $(filter all,$(FETCH)),--fetch-all,) $(if $(filter serve,$(FETCH)),--fetch-serve,)

fetch-e2e-mock:
	@echo "e2e-mock: no on-disk models to fetch"

# Run each arch whose INFERSTREAM_E2E_<ARCH>_ADDR is set. CI cloud can call
# this safely: with no addrs it prints a skip line and exits 0.
e2e-all:
	@ran=0; \
	if [ -n "$(INFERSTREAM_E2E_NVIDIA_ADDR)" ]; then \
	  $(MAKE) e2e-nvidia INFERSTREAM_E2E_NVIDIA_ADDR=$(INFERSTREAM_E2E_NVIDIA_ADDR); ran=1; \
	fi; \
	if [ -n "$(INFERSTREAM_E2E_INTEL_ADDR)" ]; then \
	  $(MAKE) e2e-intel INFERSTREAM_E2E_INTEL_ADDR=$(INFERSTREAM_E2E_INTEL_ADDR); ran=1; \
	fi; \
	if [ -n "$(INFERSTREAM_E2E_APPLE_ADDR)" ]; then \
	  $(MAKE) e2e-apple INFERSTREAM_E2E_APPLE_ADDR=$(INFERSTREAM_E2E_APPLE_ADDR); ran=1; \
	fi; \
	if [ "$$ran" = 0 ]; then \
	  echo "e2e-all: no INFERSTREAM_E2E_{NVIDIA,INTEL,APPLE}_ADDR set; nothing to run (will not start remote GPUs)."; \
	fi

e2e: e2e-all

# SHA-pinned text corpus (Tiny Shakespeare soak + STS pairs). Not pulled by
# default CI. `sts-pairs.jsonl` is committed; shakespeare is downloaded.
fetch-corpus:
	$(FETCH) --corpus $(ALIAS_ARGS)

verify-corpus:
	$(FETCH) --corpus $(ALIAS_ARGS) --verify-only

list-corpus:
	$(FETCH) --corpus --list

update-corpus-manifest:
	$(FETCH) --corpus $(ALIAS_ARGS) --update-manifest

# Cross-arch embedding parity. Never starts GPUs.
#   make e2e-parity
#     runs --parity-cross for every INFERSTREAM_E2E_{NVIDIA,INTEL,APPLE}_ADDR
#     (and/or DUMP_NVIDIA / DUMP_INTEL / DUMP_APPLE). With none set, skip.
#   make e2e-parity-goldens TARGET=nvidia WRITE=1
#     capture testdata/e2e/goldens/nvidia/<alias>.json from a live server.
WRITE ?= 0
TARGET ?= nvidia
DUMP_NVIDIA ?=
DUMP_INTEL ?=
DUMP_APPLE ?=
ifeq ($(TARGET),intel)
  PARITY_GOLDEN_ADDR ?= $(or $(INFERSTREAM_E2E_INTEL_ADDR),$(INFERSTREAM_E2E_ADDR),127.0.0.1:8461)
else ifeq ($(TARGET),apple)
  PARITY_GOLDEN_ADDR ?= $(or $(INFERSTREAM_E2E_APPLE_ADDR),$(INFERSTREAM_E2E_ADDR),127.0.0.1:8461)
else ifeq ($(TARGET),mock)
  PARITY_GOLDEN_ADDR ?= $(or $(INFERSTREAM_E2E_ADDR),127.0.0.1:8461)
else
  PARITY_GOLDEN_ADDR ?= $(or $(INFERSTREAM_E2E_NVIDIA_ADDR),$(INFERSTREAM_E2E_ADDR),127.0.0.1:8461)
endif

e2e-parity-goldens:
	$(E2E) --parity-goldens $(if $(filter 1 true yes on,$(WRITE)),--parity-write,) \
	  --target $(TARGET) \
	  --addr $(PARITY_GOLDEN_ADDR) \
	  --token "$(INFERSTREAM_E2E_TOKEN)" $(E2E_CORPUS_ARGS)

e2e-parity:
	@peers=""; dumps=""; \
	if [ -n "$(INFERSTREAM_E2E_NVIDIA_ADDR)" ]; then peers="$$peers --peer nvidia=$(INFERSTREAM_E2E_NVIDIA_ADDR)"; fi; \
	if [ -n "$(INFERSTREAM_E2E_INTEL_ADDR)" ]; then peers="$$peers --peer intel=$(INFERSTREAM_E2E_INTEL_ADDR)"; fi; \
	if [ -n "$(INFERSTREAM_E2E_APPLE_ADDR)" ]; then peers="$$peers --peer apple=$(INFERSTREAM_E2E_APPLE_ADDR)"; fi; \
	if [ -n "$(DUMP_NVIDIA)" ]; then dumps="$$dumps --dump nvidia=$(DUMP_NVIDIA)"; fi; \
	if [ -n "$(DUMP_INTEL)" ]; then dumps="$$dumps --dump intel=$(DUMP_INTEL)"; fi; \
	if [ -n "$(DUMP_APPLE)" ]; then dumps="$$dumps --dump apple=$(DUMP_APPLE)"; fi; \
	if [ -z "$$peers" ] && [ -z "$$dumps" ]; then \
	  echo "e2e-parity: no INFERSTREAM_E2E_{NVIDIA,INTEL,APPLE}_ADDR or DUMP_* set; nothing to run (will not start remote GPUs)."; \
	  echo "Capture: make e2e-parity-goldens TARGET=nvidia WRITE=1"; \
	  echo "Three-way: set INFERSTREAM_E2E_{NVIDIA,INTEL,APPLE}_ADDR to each Machine's gRPC host:port (README lab chart); make e2e-parity"; \
	  exit 0; \
	fi; \
	$(E2E) --parity-cross $$peers $$dumps --token "$(INFERSTREAM_E2E_TOKEN)" $(E2E_CORPUS_ARGS)

# Popular-model cosine drift. Same skip-if-no-addrs rule as e2e-parity.
# Reuses pair_threshold floors. See docs/turboembed-drift.md.
e2e-drift:
	@peers=""; dumps=""; \
	if [ -n "$(INFERSTREAM_E2E_NVIDIA_ADDR)" ]; then peers="$$peers --peer nvidia=$(INFERSTREAM_E2E_NVIDIA_ADDR)"; fi; \
	if [ -n "$(INFERSTREAM_E2E_INTEL_ADDR)" ]; then peers="$$peers --peer intel=$(INFERSTREAM_E2E_INTEL_ADDR)"; fi; \
	if [ -n "$(INFERSTREAM_E2E_APPLE_ADDR)" ]; then peers="$$peers --peer apple=$(INFERSTREAM_E2E_APPLE_ADDR)"; fi; \
	if [ -n "$(DUMP_NVIDIA)" ]; then dumps="$$dumps --dump nvidia=$(DUMP_NVIDIA)"; fi; \
	if [ -n "$(DUMP_INTEL)" ]; then dumps="$$dumps --dump intel=$(DUMP_INTEL)"; fi; \
	if [ -n "$(DUMP_APPLE)" ]; then dumps="$$dumps --dump apple=$(DUMP_APPLE)"; fi; \
	if [ -z "$$peers" ] && [ -z "$$dumps" ]; then \
	  echo "e2e-drift: no INFERSTREAM_E2E_{NVIDIA,INTEL,APPLE}_ADDR or DUMP_* set; nothing to run (will not start remote GPUs)."; \
	  echo "Models: minilm minilm-l12 mpnet bge-* e5-* gte-* nomic-embed-text (docs/turboembed-drift.md)"; \
	  exit 0; \
	fi; \
	$(E2E) --drift $$peers $$dumps --token "$(INFERSTREAM_E2E_TOKEN)" $(E2E_CORPUS_ARGS)

# C++ TurboEmbed ABI stub (no Rust). Writes native/turboembed/build/libturboembed.a
CXX ?= c++
AR ?= ar
turboembed-stub:
	mkdir -p native/turboembed/build
	$(CXX) -std=c++17 -fPIC -O2 -I include \
	  -c native/turboembed/src/stub.cpp \
	  -o native/turboembed/build/stub.o
	$(AR) rcs native/turboembed/build/libturboembed.a native/turboembed/build/stub.o
	@echo "wrote native/turboembed/build/libturboembed.a"

test-turboembed:
	$(CARGO) test -p turboembed

# Live TextEmbeddingPipeline on Intel CPU and GPU. GPU/NPU create fails if
# that plugin is missing (no silent CPU). Sources the host OpenVINO toolkit; no Python.
OPENVINO_SETUPVARS ?= /work/opt/openvino_genai/setupvars.sh
test-turboembed-intel:
	@if [ -f "$(OPENVINO_SETUPVARS)" ]; then \
	  set +u; . "$(OPENVINO_SETUPVARS)"; set -u; \
	fi; \
	$(CARGO) test -p turboembed --features genai

# Real Metal MiniLM through turboembed.h. No Python.
# Runs metal_create_lists_minilm_not_only_mock (create lists 384-d minilm,
# never mock-only) and apple_minilm_metal_cosine_vs_goldens (receipt).
test-turboembed-apple:
	$(CARGO) test -p turboembed --features mlx-live -- --include-ignored --nocapture
