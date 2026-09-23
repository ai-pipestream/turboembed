# Direct-native reference for the metal provider: the same Metal kernels
# driven by a plain Objective-C++ program, with no libturbo, no provider
# vtable and no session machinery in the timed path.
#
#   make -C reference/metal
#   build/reference-metal/reference-metal --tokens tokens.json --commit <sha> --out native.json

TURBO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
OUT ?= $(TURBO_ROOT)/build/reference-metal
CXX ?= clang++

INCLUDES := -I$(TURBO_ROOT)/include -I$(TURBO_ROOT)/native/provider_common -I$(TURBO_ROOT)/providers/metal/src -I$(TURBO_ROOT)/third_party
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 -ObjC++ -fobjc-arc -Wall -Wextra $(INCLUDES)

BIN := $(OUT)/reference-metal

all: $(BIN)

$(OUT):
	mkdir -p $(OUT)

$(BIN): main.mm $(TURBO_ROOT)/providers/metal/src/kernels.inc $(TURBO_ROOT)/providers/metal/src/safetensors.hpp | $(OUT)
	$(CXX) $(CXXFLAGS) main.mm -o $@ -framework Metal -framework Foundation

clean:
	rm -rf $(OUT)

.PHONY: all clean
