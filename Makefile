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
#   make fetch-rerankers                    # SHA-pin MiniLM-L6 cross-encoder
#   make turbo-buffer-tests                 # unified arena ABI (CPU + CUDA/ZE when live)
#   make turbo-buffer-intel-receipt         # Machine B ZE HOST/SHARED/DEVICE receipt
#   make wordpiece-tests                    # frozen vocab + write-through + tripwire
#   make turborerank-tests                  # C++ buffer/pack + CUDA if nvcc
#   make turborerank-tests-nocuda           # same tests, CUDA create fails loud
#   make test-turborerank                   # fetch + C++ + Rust live scores
#   make test-turborerank-nvidia            # Machine A CUDA receipt + live CE
#   make convert-rerank-ov                  # ONNX→IR (C++); ONNX from contrib/offline-once
#   make test-turborerank-intel             # Machine B OpenVINO GPU/CPU receipt + live CE
#   make bench-machine-b-ov                 # Machine B FINAL SOLIDIFY bench (OV GPU p50/p99)
#   make probe-remote-usm                   # Machine B: remote OCL/USM wrap probe (may FAIL)
#   make test-turborerank-apple             # Machine C Metal receipt + live CE
#   make metal-erf-probe                    # Machine C: MSL has no erf(); Hart GELU vs libm
#   make bench-machine-c                    # Machine C FINAL SOLIDIFY bench receipt
#   make test-turboembed-intel              # --features genai; WordPiece→USM + CompiledModel; NPU create fails loud if missing
#   make test-turboembed-apple              # Mac: Metal create lists minilm + goldens receipt
#   make bench-machine-a                    # Machine A CUDA p50/p99 + H2D/D2H + goldens
#   make bench-turbo MACHINE=A              # same as bench-machine-a
#   make bench-turbo MACHINE=B              # same as bench-machine-b-ov
#   make bench-turbo MACHINE=C              # same as bench-machine-c
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
	turboembed-stub test-turboembed test-turboembed-intel test-turboembed-apple \
	fetch-rerankers verify-rerankers list-rerankers update-rerank-manifest \
	turborerank-tests 	turborerank-tests-nocuda turborerank-tests-noov \
	turborerank-cuda-gemm-proof \
	turborerank-tests-nometal libturborerank-apple libturbo-buffer-apple \
	turbo-buffer-tests wordpiece-tests wordpiece-tripwire \
	turboembed-mock-arena-tests turboembed-genai-arena-tests \
	test-turborerank test-turborerank-nvidia turborerank-nvidia-receipt \
	convert-rerank-ov verify-rerank-ov probe-remote-usm test-turborerank-intel \
	turborerank-intel-receipt test-turborerank-apple turborerank-apple-receipt \
	metal-erf-probe bench-machine-b-ov \
	bench-turbo bench-machine-a \
	bench-machine-c bench-machine-c-rerank

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
		$(CARGO) test -p turboembed --features ort-cuda -- --include-ignored --nocapture --test-threads=1

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

# Static C++/ObjC++ ABI for the Swift server (Machine C). Header-only
# TurboRerankC in SPM links this archive — no second @_cdecl dylib.
libturborerank-apple:
	mkdir -p native/turborerank/build
	@for src in $(TURBORERANK_SRCS) $(TURBORERANK_METAL_SRC) $(TURBO_BUFFER_METAL_SRC); do \
	  obj="native/turborerank/build/$$(basename $$src).o"; \
	  $(TURBORERANK_CXX) -std=c++17 -O2 -fPIC $(TURBORERANK_INCLUDES) \
	    $(TURBORERANK_CPPFLAGS) $(TURBORERANK_METAL_FLAGS) \
	    -c $$src -o $$obj; \
	done
	$(AR) rcs native/turborerank/build/libturborerank_apple.a \
	  native/turborerank/build/alloc.cpp.o \
	  native/turborerank/build/pack.cpp.o \
	  native/turborerank/build/wordpiece.cpp.o \
	  native/turborerank/build/vocab_load.cpp.o \
	  native/turborerank/build/encode.cpp.o \
	  native/turborerank/build/safetensors.cpp.o \
	  native/turborerank/build/bert_cpu.cpp.o \
	  native/turborerank/build/cuda_api.cpp.o \
	  native/turborerank/build/ov_api.cpp.o \
	  native/turborerank/build/metal_api.cpp.o \
	  native/turborerank/build/engine.cpp.o \
	  native/turborerank/build/arena.cpp.o \
	  native/turborerank/build/cuda.cpp.o \
	  native/turborerank/build/ze.cpp.o \
	  native/turborerank/build/metal.cpp.o \
	  $(if $(TURBORERANK_METAL_SRC),native/turborerank/build/metal_api.mm.o,) \
	  $(if $(TURBO_BUFFER_METAL_SRC),native/turborerank/build/metal.mm.o,)
	@nm -g native/turborerank/build/libturborerank_apple.a | grep -q turbo_buffer_arena_rent
	@nm -g native/turborerank/build/libturborerank_apple.a | grep -q turbo_buffer_metal_owns
	@echo "wrote native/turborerank/build/libturborerank_apple.a (arena + Metal SHARED)"

# Metal SHARED arena for libTurboEmbed.dylib (SOLIDIFY item 4).
libturbo-buffer-apple:
	./scripts/build-turbo-buffer-apple.sh

apple: sync-proto libturborerank-apple libturbo-buffer-apple
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
# Clang as `c++` on Linux images lacks libstdc++ headers; g++ is the
# TurboRerank test compiler there. On Machine C (Darwin) clang++ is
# required for Objective-C++ Metal. Override with TURBORERANK_CXX=...
ifeq ($(shell uname -s),Darwin)
TURBORERANK_CXX ?= clang++
else
TURBORERANK_CXX ?= g++
endif
AR ?= ar
turboembed-stub:
	mkdir -p native/turboembed/build
	$(CXX) -std=c++17 -fPIC -O2 -I include -I native/turbo_buffer/src \
	  -c native/turboembed/src/stub.cpp \
	  -o native/turboembed/build/stub.o
	$(CXX) -std=c++17 -fPIC -O2 -I include -I native/turbo_buffer/src \
	  -c native/turbo_buffer/src/arena.cpp \
	  -o native/turboembed/build/arena.o
	$(CXX) -std=c++17 -fPIC -O2 -I include -I native/turbo_buffer/src \
	  -c native/turbo_buffer/src/cuda.cpp \
	  -o native/turboembed/build/tb_cuda.o
	$(CXX) -std=c++17 -fPIC -O2 -I include -I native/turbo_buffer/src \
	  -c native/turbo_buffer/src/ze.cpp \
	  -o native/turboembed/build/tb_ze.o
	$(CXX) -std=c++17 -fPIC -O2 -I include -I native/turbo_buffer/src \
	  -c native/turbo_buffer/src/metal.cpp \
	  -o native/turboembed/build/tb_metal.o
	$(CXX) -std=c++17 -fPIC -O2 -I include -I native/wordpiece \
	  -c native/wordpiece/vocab_load.cpp \
	  -o native/turboembed/build/vocab_load.o
	$(CXX) -std=c++17 -fPIC -O2 -I include -I native/wordpiece \
	  -c native/wordpiece/encode.cpp \
	  -o native/turboembed/build/encode.o
	$(AR) rcs native/turboembed/build/libturboembed.a \
	  native/turboembed/build/stub.o \
	  native/turboembed/build/arena.o \
	  native/turboembed/build/tb_cuda.o \
	  native/turboembed/build/tb_ze.o \
	  native/turboembed/build/tb_metal.o \
	  native/turboembed/build/vocab_load.o \
	  native/turboembed/build/encode.o
	@echo "wrote native/turboembed/build/libturboembed.a"

test-turboembed: turboembed-mock-arena-tests
	$(CARGO) test -p turboembed

# Live GenAI Tokenizer+CompiledModel on Intel CPU and GPU, ZE USM arena.
# GPU/NPU create fails if that plugin or ZE SHARED is missing (no silent CPU).
OPENVINO_SETUPVARS ?= /work/opt/openvino_genai/setupvars.sh
OPENVINO_GENAI_ROOT ?= /work/opt/openvino_genai
test-turboembed-intel: turboembed-genai-arena-tests
	@if [ -f "$(OPENVINO_SETUPVARS)" ]; then \
	  set +u; . "$(OPENVINO_SETUPVARS)"; set -u; \
	fi; \
	$(CARGO) test -p turboembed --features genai -- --test-threads=1

# C++ proof: GPU SHARED + CPU HOST token/result rent, allocs/forward==0.
turboembed-genai-arena-tests:
	@if [ ! -f "$(OPENVINO_GENAI_ROOT)/runtime/include/openvino/genai/tokenizer.hpp" ]; then \
	  echo "skip turboembed-genai-arena-tests: OpenVINO GenAI headers missing"; \
	  exit 1; \
	fi
	mkdir -p native/turboembed/build
	@if [ -f "$(OPENVINO_SETUPVARS)" ]; then set +u; . "$(OPENVINO_SETUPVARS)"; set -u; fi; \
	$(TURBORERANK_CXX) -std=c++17 -O2 -g \
	  -DTURBOEMBED_GENAI -DTURBO_BUFFER_ZE=1 \
	  -DTURBOEMBED_WORKSPACE_ROOT=\"$(CURDIR)\" \
	  -I include -I native/turboembed/src -I native/turbo_buffer/src \
	  -I native/wordpiece \
	  -I $(OPENVINO_GENAI_ROOT)/runtime/include \
	  native/turboembed/src/stub.cpp \
	  native/turboembed/src/genai.cpp \
	  $(WORDPIECE_SRCS) \
	  $(TURBO_BUFFER_SRCS) \
	  native/turboembed/tests/genai_arena_tests.cpp \
	  -L$(OPENVINO_GENAI_ROOT)/runtime/lib/intel64 \
	  -lopenvino -lopenvino_genai -lopenvino_tokenizers -lze_loader -lm \
	  -Wl,-rpath,$(OPENVINO_GENAI_ROOT)/runtime/lib/intel64 \
	  -o native/turboembed/build/genai_arena_tests
	INFERSTREAM_ROOT=$(CURDIR) native/turboembed/build/genai_arena_tests

# Real Metal MiniLM through turboembed.h. No Python.
# Runs metal_create_lists_minilm_not_only_mock (create lists 384-d minilm,
# never mock-only) and apple_minilm_metal_cosine_vs_goldens (receipt).
test-turboembed-apple: libturbo-buffer-apple
	$(CARGO) test -p turboembed --features mlx-live -- --include-ignored --nocapture

# TurboRerank: SHA-pinned MiniLM-L6 CE + C++ / Rust tests.
fetch-rerankers:
	$(FETCH) --rerankers $(ALIAS_ARGS)

verify-rerankers:
	$(FETCH) --rerankers --verify-only $(ALIAS_ARGS)

list-rerankers:
	$(FETCH) --rerankers --list

update-rerank-manifest:
	$(FETCH) --rerankers --update-manifest $(ALIAS_ARGS)

TURBO_BUFFER_SRCS := \
	native/turbo_buffer/src/arena.cpp \
	native/turbo_buffer/src/cuda.cpp \
	native/turbo_buffer/src/ze.cpp \
	native/turbo_buffer/src/metal.cpp

WORDPIECE_SRCS := \
	native/wordpiece/vocab_load.cpp \
	native/wordpiece/encode.cpp

TURBORERANK_SRCS := \
	native/turborerank/src/alloc.cpp \
	native/turborerank/src/pack.cpp \
	native/turborerank/src/wordpiece.cpp \
	$(WORDPIECE_SRCS) \
	native/turborerank/src/safetensors.cpp \
	native/turborerank/src/bert_cpu.cpp \
	native/turborerank/src/cuda_api.cpp \
	native/turborerank/src/ov_api.cpp \
	native/turborerank/src/metal_api.cpp \
	native/turborerank/src/engine.cpp \
	$(TURBO_BUFFER_SRCS)

TURBORERANK_NVCC ?= nvcc
# CUDA 12.4 rejects gcc 15 as nvcc host. g++-13 is on Machine A.
TURBORERANK_NVCC_CCBIN ?= g++-13
# 0/1. Default: compile CUDA when nvcc + cuda_runtime.h exist.
TURBORERANK_ENABLE_CUDA ?= $(shell \
	if command -v $(TURBORERANK_NVCC) >/dev/null 2>&1 && \
	   { test -f /usr/include/cuda_runtime.h || test -f /usr/local/cuda/include/cuda_runtime.h; }; \
	then echo 1; else echo 0; fi)
TURBORERANK_CUDA_ARCH ?= native

TURBORERANK_INCLUDES := -I include -I native/wordpiece -I native/turborerank/src -I native/turbo_buffer/src
TURBORERANK_CPPFLAGS := -DTURBORERANK_WORKSPACE_ROOT=\"$(CURDIR)\"
TURBORERANK_CUDA_LIBS :=
TURBORERANK_CUDA_OBJ :=
TURBORERANK_OV_LIBS :=
TURBORERANK_METAL_LIBS :=
TURBORERANK_METAL_SRC :=
TURBO_BUFFER_METAL_SRC :=
TURBORERANK_METAL_FLAGS :=

# 0/1. Default: compile Metal on Darwin unless TURBORERANK_ENABLE_METAL=0.
TURBORERANK_ENABLE_METAL ?= $(shell \
	if [ "$$(uname -s)" = Darwin ]; then echo 1; else echo 0; fi)

# 0/1. Default: compile OpenVINO when pkg-config openvino works.
TURBORERANK_ENABLE_OV ?= $(shell \
	if pkg-config --exists openvino 2>/dev/null; then echo 1; else echo 0; fi)
TURBORERANK_ENABLE_L0 ?= $(shell \
	if test -f /usr/include/level_zero/ze_api.h && \
	   { test -f /usr/lib/x86_64-linux-gnu/libze_loader.so || \
	     test -f /usr/lib/x86_64-linux-gnu/libze_loader.so.1; }; \
	then echo 1; else echo 0; fi)

ifeq ($(TURBORERANK_ENABLE_OV),1)
TURBORERANK_CPPFLAGS += -DTURBORERANK_OPENVINO=1
TURBORERANK_INCLUDES += $(shell pkg-config --cflags openvino 2>/dev/null)
TURBORERANK_OV_LIBDIR := $(shell pkg-config --variable=libdir openvino 2>/dev/null)
ifeq ($(TURBORERANK_OV_LIBDIR),)
TURBORERANK_OV_LIBDIR := $(shell pkg-config --libs-only-L openvino 2>/dev/null | sed 's/-L//')
endif
TURBORERANK_OV_LIBS := -L$(TURBORERANK_OV_LIBDIR) -lopenvino -Wl,-rpath,$(TURBORERANK_OV_LIBDIR)
ifeq ($(TURBORERANK_ENABLE_L0),1)
TURBORERANK_CPPFLAGS += -DTURBORERANK_LEVEL_ZERO=1 -DTURBO_BUFFER_ZE=1
TURBORERANK_OV_LIBS += -lze_loader
endif
endif

ifeq ($(TURBORERANK_ENABLE_CUDA),1)
TURBORERANK_CPPFLAGS += -DTURBORERANK_CUDA=1 -DTURBO_BUFFER_CUDA=1
TURBORERANK_CUDA_LIBS := -lcudart -lcublasLt
TURBORERANK_CUDA_OBJ := native/turborerank/build/bert_cuda.o
endif

TURBO_BUFFER_METAL_SRC :=
ifeq ($(TURBORERANK_ENABLE_METAL),1)
TURBORERANK_CPPFLAGS += -DTURBORERANK_METAL=1 -DTURBO_BUFFER_METAL=1
TURBORERANK_METAL_FLAGS := -fobjc-arc
TURBORERANK_METAL_SRC := native/turborerank/src/metal_api.mm
TURBO_BUFFER_METAL_SRC := native/turbo_buffer/src/metal.mm
TURBORERANK_METAL_LIBS := -framework Metal -framework Foundation
endif

native/turborerank/build/bert_cuda.o: native/turborerank/src/bert_cuda.cu \
		native/turborerank/src/cuda_api.hpp native/turborerank/src/internal.hpp \
		include/turborerank.h include/reranker.hpp include/turbo_buffer.h \
		native/turbo_buffer/src/cuda_runtime_hooks.hpp
	mkdir -p native/turborerank/build
	$(TURBORERANK_NVCC) -std=c++17 -O2 -arch=$(TURBORERANK_CUDA_ARCH) \
	  -ccbin=$(TURBORERANK_NVCC_CCBIN) \
	  $(TURBORERANK_INCLUDES) -DTURBORERANK_CUDA=1 -DTURBO_BUFFER_CUDA=1 \
	  -DTURBORERANK_WORKSPACE_ROOT=\"$(CURDIR)\" \
	  -c native/turborerank/src/bert_cuda.cu \
	  -o native/turborerank/build/bert_cuda.o

# SOLIDIFY (3): primary GEMM is cuBLASLt. The hand-rolled linear_nt_kernel
# must not exist; the linked test binary must reference cublasLtMatmul.
turborerank-cuda-gemm-proof: $(TURBORERANK_CUDA_OBJ)
ifeq ($(TURBORERANK_ENABLE_CUDA),1)
	@grep -q 'cublasLtMatmul(' native/turborerank/src/bert_cuda.cu || { \
	  echo "FAIL: bert_cuda.cu must call cublasLtMatmul"; exit 1; }
	@if grep -n 'linear_nt_kernel' native/turborerank/src/bert_cuda.cu; then \
	  echo "FAIL: linear_nt_kernel still present — not the cuBLASLt stack"; \
	  exit 1; fi
	@echo "gemm primary path: cublasLtMatmul (no linear_nt_kernel)"
endif

# SOLIDIFY (5): encode.cpp must stay heap-free. Load-time heap is vocab_load.cpp.
wordpiece-tripwire:
	@if grep -nE 'std::(vector|string|unordered_map)|new |malloc\(|posix_memalign' \
	    native/wordpiece/encode.cpp; then \
	  echo "FAIL: heap token staging reintroduced in encode.cpp"; exit 1; \
	fi
	@if grep -nE 'copy_tokens_to_i32|tokenizer\.encode' native/turboembed/src/genai.cpp; then \
	  echo "FAIL: GenAI encode→copy path reintroduced"; exit 1; \
	fi
	@echo "wordpiece tripwire: encode.cpp heap-free, GenAI has no encode→copy"

wordpiece-tests: wordpiece-tripwire turbo-buffer-tests
	mkdir -p native/wordpiece/build
	$(TURBORERANK_CXX) -std=c++17 -O2 -g -I include -I native/wordpiece \
	  -I native/turbo_buffer/src \
	  $(WORDPIECE_SRCS) native/turbo_buffer/src/arena.cpp \
	  native/turbo_buffer/src/cuda.cpp native/turbo_buffer/src/ze.cpp \
	  native/turbo_buffer/src/metal.cpp \
	  native/wordpiece/tests/wordpiece_tests.cpp \
	  -lm -o native/wordpiece/build/wordpiece_tests
	INFERSTREAM_ROOT=$(CURDIR) native/wordpiece/build/wordpiece_tests

turbo-buffer-tests:
	mkdir -p native/turbo_buffer/build
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  $(TURBORERANK_CPPFLAGS) $(TURBORERANK_METAL_FLAGS) \
	  $(TURBO_BUFFER_SRCS) $(TURBO_BUFFER_METAL_SRC) \
	  native/turbo_buffer/tests/turbo_buffer_tests.cpp \
	  -lm $(TURBORERANK_CUDA_LIBS) $(TURBORERANK_OV_LIBS) $(TURBORERANK_METAL_LIBS) \
	  -o native/turbo_buffer/build/turbo_buffer_tests
	native/turbo_buffer/build/turbo_buffer_tests

turbo-buffer-intel-receipt: turbo-buffer-tests
	mkdir -p native/turbo_buffer/build testdata/receipts/turbo_buffer
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  $(TURBORERANK_CPPFLAGS) \
	  $(TURBO_BUFFER_SRCS) \
	  native/turbo_buffer/tools/write_intel_ze_receipt.cpp \
	  -lm $(TURBORERANK_OV_LIBS) \
	  -o native/turbo_buffer/build/write_intel_ze_receipt
	INFERSTREAM_ROOT=$(CURDIR) native/turbo_buffer/build/write_intel_ze_receipt

turboembed-mock-arena-tests: turboembed-stub
	$(TURBORERANK_CXX) -std=c++17 -O2 -g -I include \
	  native/turboembed/tests/mock_arena_tests.cpp \
	  native/turboembed/build/libturboembed.a \
	  -lm -o native/turboembed/build/mock_arena_tests
	native/turboembed/build/mock_arena_tests

turborerank-tests: $(TURBORERANK_CUDA_OBJ) turborerank-cuda-gemm-proof turbo-buffer-tests wordpiece-tests
	mkdir -p native/turborerank/build
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  $(TURBORERANK_CPPFLAGS) $(TURBORERANK_METAL_FLAGS) \
	  $(TURBORERANK_SRCS) $(TURBORERANK_METAL_SRC) $(TURBO_BUFFER_METAL_SRC) \
	  $(TURBORERANK_CUDA_OBJ) \
	  native/turborerank/tests/turborerank_tests.cpp \
	  -lm $(TURBORERANK_CUDA_LIBS) $(TURBORERANK_OV_LIBS) $(TURBORERANK_METAL_LIBS) \
	  -o native/turborerank/build/turborerank_tests
ifeq ($(TURBORERANK_ENABLE_CUDA),1)
	@nm native/turborerank/build/turborerank_tests | grep -q cublasLtMatmul || { \
	  echo "FAIL: turborerank_tests does not reference cublasLtMatmul"; exit 1; }
endif
	INFERSTREAM_ROOT=$(CURDIR) native/turborerank/build/turborerank_tests

# Prove CUDA create fails loud when the binary has no CUDA.
# Strip CUDA defines even on a Machine A host — this target is the
# no-nvcc / no-cudart binary, not "CUDA flags without -lcudart".
turborerank-tests-nocuda:
	mkdir -p native/turborerank/build
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  $(filter-out -DTURBORERANK_CUDA=1 -DTURBO_BUFFER_CUDA=1,$(TURBORERANK_CPPFLAGS)) \
	  -DTURBORERANK_WORKSPACE_ROOT=\"$(CURDIR)\" \
	  $(TURBORERANK_SRCS) $(TURBORERANK_METAL_SRC) $(TURBO_BUFFER_METAL_SRC) \
	  native/turborerank/tests/turborerank_tests.cpp \
	  -lm $(TURBORERANK_OV_LIBS) $(TURBORERANK_METAL_LIBS) $(TURBORERANK_METAL_FLAGS) \
	  -o native/turborerank/build/turborerank_tests_nocuda
	INFERSTREAM_ROOT=$(CURDIR) native/turborerank/build/turborerank_tests_nocuda

# Prove OpenVINO GPU/CPU create fails loud when the binary has no OV.
turborerank-tests-noov:
	mkdir -p native/turborerank/build
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  -DTURBORERANK_WORKSPACE_ROOT=\"$(CURDIR)\" \
	  $(if $(filter 1,$(TURBORERANK_ENABLE_METAL)),-DTURBORERANK_METAL=1 -DTURBO_BUFFER_METAL=1) \
	  $(TURBORERANK_METAL_FLAGS) \
	  $(TURBORERANK_SRCS) $(TURBORERANK_METAL_SRC) $(TURBO_BUFFER_METAL_SRC) \
	  native/turborerank/tests/turborerank_tests.cpp \
	  -lm $(TURBORERANK_METAL_LIBS) \
	  -o native/turborerank/build/turborerank_tests_noov
	INFERSTREAM_ROOT=$(CURDIR) native/turborerank/build/turborerank_tests_noov

# Prove Metal create fails loud when the binary has no Metal.
turborerank-tests-nometal:
	mkdir -p native/turborerank/build
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  -DTURBORERANK_WORKSPACE_ROOT=\"$(CURDIR)\" \
	  $(TURBORERANK_SRCS) native/turborerank/tests/turborerank_tests.cpp \
	  -lm -o native/turborerank/build/turborerank_tests_nometal
	INFERSTREAM_ROOT=$(CURDIR) native/turborerank/build/turborerank_tests_nometal

turborerank-nvidia-receipt: $(TURBORERANK_CUDA_OBJ)
	mkdir -p native/turborerank/build
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  $(TURBORERANK_CPPFLAGS) \
	  $(TURBORERANK_SRCS) $(TURBORERANK_METAL_SRC) $(TURBO_BUFFER_METAL_SRC) \
	  $(TURBORERANK_CUDA_OBJ) \
	  native/turborerank/tools/write_nvidia_receipt.cpp \
	  -lm $(TURBORERANK_CUDA_LIBS) $(TURBORERANK_OV_LIBS) $(TURBORERANK_METAL_LIBS) \
	  $(TURBORERANK_METAL_FLAGS) \
	  -o native/turborerank/build/write_nvidia_receipt
	INFERSTREAM_ROOT=$(CURDIR) native/turborerank/build/write_nvidia_receipt

# C++ OpenVINO IR from the exported ONNX (no Python).
convert-rerank-ov:
	mkdir -p native/turborerank/build models/ov-rerank/ms-marco-minilm-l6
	$(TURBORERANK_CXX) -std=c++17 -O2 $(TURBORERANK_INCLUDES) \
	  native/turborerank/tools/onnx_to_ir.cpp \
	  $(TURBORERANK_OV_LIBS) -o native/turborerank/build/onnx_to_ir
	native/turborerank/build/onnx_to_ir \
	  models/ov-rerank/ms-marco-minilm-l6/model.onnx \
	  models/ov-rerank/ms-marco-minilm-l6/openvino_model.xml

verify-rerank-ov:
	@xml=models/ov-rerank/ms-marco-minilm-l6/openvino_model.xml; \
	bin=models/ov-rerank/ms-marco-minilm-l6/openvino_model.bin; \
	test -f "$$xml" && test -f "$$bin" || { \
	  echo "OpenVINO IR missing ($$xml). Export ONNX then make convert-rerank-ov."; \
	  exit 1; }; \
	got_xml=$$(sha256sum "$$xml" | awk '{print $$1}'); \
	got_bin=$$(sha256sum "$$bin" | awk '{print $$1}'); \
	exp_xml=7c14ec8664b37a6d1e74370c65202b7e9f275ca545d81af5fc99f39e08d2f525; \
	exp_bin=1ec03e9f414b6a4ccb73fc1d674f906bd7f246a04414d9ad058e61b40b270dcd; \
	test "$$got_xml" = "$$exp_xml" || { echo "xml sha256 $$got_xml != $$exp_xml"; exit 1; }; \
	test "$$got_bin" = "$$exp_bin" || { echo "bin sha256 $$got_bin != $$exp_bin"; exit 1; }; \
	echo "verified $$xml sha256=$$got_xml"; \
	echo "verified $$bin sha256=$$got_bin"

turborerank-intel-receipt: $(TURBORERANK_CUDA_OBJ)
	mkdir -p native/turborerank/build
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  $(TURBORERANK_CPPFLAGS) \
	  $(TURBORERANK_SRCS) $(TURBORERANK_METAL_SRC) $(TURBO_BUFFER_METAL_SRC) \
	  $(TURBORERANK_CUDA_OBJ) \
	  native/turborerank/tools/write_intel_receipt.cpp \
	  -lm $(TURBORERANK_CUDA_LIBS) $(TURBORERANK_OV_LIBS) $(TURBORERANK_METAL_LIBS) \
	  $(TURBORERANK_METAL_FLAGS) \
	  -o native/turborerank/build/write_intel_receipt
	INFERSTREAM_ROOT=$(CURDIR) native/turborerank/build/write_intel_receipt

# SOLIDIFY (7): live RemoteContext USM_USER_BUFFER wrap of ZE SHARED.
# Exit 1 is the documented Machine B outcome (OCL size-0). Do not
# treat a FAIL as a green wrap.
probe-remote-usm:
	mkdir -p native/turborerank/build testdata/receipts/turborerank
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  $(TURBORERANK_CPPFLAGS) \
	  $(TURBO_BUFFER_SRCS) \
	  native/turborerank/tools/probe_remote_usm.cpp \
	  -lm $(TURBORERANK_OV_LIBS) \
	  -o native/turborerank/build/probe_remote_usm
	set +e; \
	INFERSTREAM_ROOT=$(CURDIR) native/turborerank/build/probe_remote_usm \
	  > testdata/receipts/turborerank/intel-remote-usm-probe.txt 2>&1; \
	rc=$$?; \
	cat testdata/receipts/turborerank/intel-remote-usm-probe.txt; \
	if [ $$rc -eq 0 ]; then \
	  echo "probe-remote-usm: WRAP_POSSIBLE"; \
	elif grep -q WRAP_IMPOSSIBLE_ON_THIS_STACK \
	    testdata/receipts/turborerank/intel-remote-usm-probe.txt; then \
	  echo "probe-remote-usm: documented unavailable path (exit $$rc)"; \
	else \
	  echo "probe-remote-usm: unexpected exit $$rc"; \
	  exit $$rc; \
	fi

turborerank-apple-receipt:
	mkdir -p native/turborerank/build
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  $(TURBORERANK_CPPFLAGS) $(TURBORERANK_METAL_FLAGS) \
	  $(TURBORERANK_SRCS) $(TURBORERANK_METAL_SRC) $(TURBO_BUFFER_METAL_SRC) \
	  native/turborerank/tools/write_apple_receipt.cpp \
	  -lm $(TURBORERANK_METAL_LIBS) \
	  -o native/turborerank/build/write_apple_receipt
	INFERSTREAM_ROOT=$(CURDIR) native/turborerank/build/write_apple_receipt

test-turborerank: fetch-rerankers turboembed-mock-arena-tests turborerank-tests turborerank-tests-nocuda
	INFERSTREAM_ROOT=$(CURDIR) $(CARGO) test -p turborerank -- --include-ignored --nocapture
	INFERSTREAM_ROOT=$(CURDIR) $(CARGO) test -p inferstream-backend-turborerank -- --include-ignored --nocapture

test-turborerank-nvidia: test-turborerank turborerank-nvidia-receipt

metal-erf-probe:
	@if [ "$(TURBORERANK_ENABLE_METAL)" != "1" ]; then \
	  echo "metal-erf-probe requires Darwin + TURBORERANK_ENABLE_METAL=1"; \
	  exit 1; \
	fi
	mkdir -p native/turborerank/build
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  $(TURBORERANK_CPPFLAGS) $(TURBORERANK_METAL_FLAGS) \
	  native/turborerank/tools/metal_erf_probe.mm \
	  -lm $(TURBORERANK_METAL_LIBS) \
	  -o native/turborerank/build/metal_erf_probe
	INFERSTREAM_ROOT=$(CURDIR) native/turborerank/build/metal_erf_probe

# Machine C: Metal MiniLM CE vs HF Berlin golden. nometal proves fail-loud.
# libturborerank-apple is the Swift-linkable archive — same arena objects.
# metal-erf-probe proves MSL still has no erf() and Hart beats A&S on device.
test-turborerank-apple: fetch-rerankers metal-erf-probe turborerank-tests \
		turborerank-tests-nometal libturborerank-apple
	INFERSTREAM_ROOT=$(CURDIR) $(CARGO) test -p turborerank -- --include-ignored --nocapture
	$(MAKE) turborerank-apple-receipt

# FINAL SOLIDIFY bench — Machine C Metal. Live p50/p99, allocs/forward==0,
# Berlin + embed goldens in band, SHARED, no CPU fallback. Does not copy
# a prior receipt. Writes testdata/receipts/bench/machine-c-metal.json.
BENCH_RERANK_JSON ?= $(CURDIR)/native/turborerank/build/machine-c-rerank-bench.json

bench-machine-c-rerank:
	@if [ "$(TURBORERANK_ENABLE_METAL)" != "1" ]; then \
	  echo "bench-machine-c-rerank requires Darwin + TURBORERANK_ENABLE_METAL=1"; \
	  exit 1; \
	fi
	mkdir -p native/turborerank/build testdata/receipts/bench
	$(TURBORERANK_CXX) -std=c++17 -O2 -g $(TURBORERANK_INCLUDES) \
	  $(TURBORERANK_CPPFLAGS) $(TURBORERANK_METAL_FLAGS) \
	  $(TURBORERANK_SRCS) $(TURBORERANK_METAL_SRC) $(TURBO_BUFFER_METAL_SRC) \
	  native/turborerank/tools/bench_apple_metal.cpp \
	  -lm $(TURBORERANK_METAL_LIBS) \
	  -o native/turborerank/build/bench_apple_metal
	INFERSTREAM_ROOT=$(CURDIR) BENCH_WARMUP=$(BENCH_WARMUP) \
	  BENCH_ITERS=$(BENCH_ITERS) BENCH_RERANK_JSON=$(BENCH_RERANK_JSON) \
	  native/turborerank/build/bench_apple_metal

bench-machine-c: fetch-rerankers metal-erf-probe \
		turborerank-tests-nometal libturbo-buffer-apple bench-machine-c-rerank
	INFERSTREAM_ROOT=$(CURDIR) BENCH_WARMUP=$(BENCH_WARMUP) \
	  BENCH_ITERS=$(BENCH_ITERS) BENCH_RERANK_JSON=$(BENCH_RERANK_JSON) \
	  $(CARGO) test -p turboembed --features mlx-live --test apple_solidify_bench \
	  -- --ignored --nocapture --test-threads=1

# Machine B: OpenVINO GPU/CPU MiniLM CE vs HF Berlin golden + ZE arena.
test-turborerank-intel: fetch-rerankers verify-rerank-ov turborerank-tests turborerank-tests-noov
	@if [ -f "$(OPENVINO_SETUPVARS)" ]; then \
	  set +u; . "$(OPENVINO_SETUPVARS)"; set -u; \
	fi; \
	INFERSTREAM_ROOT=$(CURDIR) $(CARGO) test -p turborerank -- --include-ignored --nocapture
	$(MAKE) probe-remote-usm
	$(MAKE) turbo-buffer-intel-receipt
	$(MAKE) turborerank-intel-receipt

# FINAL SOLIDIFY bench — Machine B OpenVINO GPU. Live p50/p99, USM-byte
# honesty, allocs/forward==0, Berlin + MiniLM goldens. Writes
# testdata/receipts/bench/machine-b-ov.json. BENCH_WARMUP / BENCH_ITERS
# override the 32 / 128 defaults. No invented numbers.
bench-machine-b-ov:
	@if [ ! -f "$(OPENVINO_GENAI_ROOT)/runtime/include/openvino/genai/tokenizer.hpp" ]; then \
	  echo "bench-machine-b-ov: OpenVINO GenAI headers missing"; \
	  exit 1; \
	fi
	mkdir -p native/bench/build testdata/receipts/bench
	@if [ -f "$(OPENVINO_SETUPVARS)" ]; then set +u; . "$(OPENVINO_SETUPVARS)"; set -u; fi; \
	$(TURBORERANK_CXX) -std=c++17 -O2 -g \
	  -DTURBOEMBED_GENAI -DTURBO_BUFFER_ZE=1 \
	  -DTURBOEMBED_WORKSPACE_ROOT=\"$(CURDIR)\" \
	  $(TURBORERANK_INCLUDES) $(TURBORERANK_CPPFLAGS) \
	  -I native/turboembed/src \
	  -I $(OPENVINO_GENAI_ROOT)/runtime/include \
	  $(TURBORERANK_SRCS) \
	  native/turboembed/src/stub.cpp \
	  native/turboembed/src/genai.cpp \
	  native/bench/machine_b_ov.cpp \
	  -lm $(TURBORERANK_OV_LIBS) \
	  -L$(OPENVINO_GENAI_ROOT)/runtime/lib/intel64 \
	  -lopenvino_genai -lopenvino_tokenizers \
	  -Wl,-rpath,$(OPENVINO_GENAI_ROOT)/runtime/lib/intel64 \
	  -o native/bench/build/machine_b_ov
	@if [ -f "$(OPENVINO_SETUPVARS)" ]; then set +u; . "$(OPENVINO_SETUPVARS)"; set -u; fi; \
	INFERSTREAM_ROOT=$(CURDIR) native/bench/build/machine_b_ov

# SOLIDIFY (7) Machine A: TurboEmbed MiniLM + TurboRerank MiniLM-L6 CUDA
# bench. Release build. N≥100 after warmup. Writes
# testdata/receipts/bench/machine-a-cuda.json. Numbers are live — do not
# hand-edit the receipt.
#
#   make bench-machine-a
#   make bench-turbo MACHINE=A
#   make bench-turbo MACHINE=B
#   make bench-turbo MACHINE=C
MACHINE ?=
BENCH_WARMUP ?= 32
BENCH_ITERS ?= 200
BENCH_PARTIAL := native/bench/build
BENCH_RECEIPT_A := testdata/receipts/bench/machine-a-cuda.json

bench-machine-a:
	mkdir -p $(BENCH_PARTIAL) testdata/receipts/bench
	$(MAKE) fetch-rerankers
	INFERSTREAM_ROOT=$(CURDIR) \
	LD_LIBRARY_PATH="$(CURDIR)/.libs/nvidia/lib:$(LD_LIBRARY_PATH)" \
		$(CARGO) run -p bench-turbo --release --features embed --bin bench-turboembed -- \
		  --warmup $(BENCH_WARMUP) --iters $(BENCH_ITERS) \
		  --out $(BENCH_PARTIAL)/machine-a-embed.json
	INFERSTREAM_ROOT=$(CURDIR) \
		$(CARGO) run -p bench-turbo --release --features rerank --bin bench-turborerank -- \
		  --warmup $(BENCH_WARMUP) --iters $(BENCH_ITERS) \
		  --out $(BENCH_PARTIAL)/machine-a-rerank.json
	INFERSTREAM_ROOT=$(CURDIR) \
		$(CARGO) run -p bench-turbo --release --bin bench-turbo -- \
		  --machine A \
		  --embed $(BENCH_PARTIAL)/machine-a-embed.json \
		  --rerank $(BENCH_PARTIAL)/machine-a-rerank.json \
		  --out $(BENCH_RECEIPT_A)

bench-turbo:
ifeq ($(MACHINE),A)
	$(MAKE) bench-machine-a
else ifeq ($(MACHINE),B)
	$(MAKE) bench-machine-b-ov
else ifeq ($(MACHINE),C)
	$(MAKE) bench-machine-c
else
	@echo "bench-turbo: set MACHINE=A (CUDA), MACHINE=B (OpenVINO GPU), or MACHINE=C (Metal)."; \
	echo "  make bench-machine-a          # NVIDIA / Machine A"; \
	echo "  make bench-machine-b-ov       # Intel / Machine B"; \
	echo "  make bench-machine-c          # Apple / Machine C"; \
	echo "See docs/bench-turbo-machine-a.md, docs/solidify-bench-machine-b.md,"; \
	echo "and docs/apple-solidify-bench-machine-c.md"; \
	exit 1
endif
