# Turbo Metal provider: built with clang++ and make only, because that is
# what a Mac with the Command Line Tools has. No cmake, no Xcode project,
# no offline shader compile: the MSL in src/kernels.inc is compiled by the
# provider at load time.
#
#   make            -> build/metal/libturbo_provider_metal.dylib
#   make test       -> build and run the vtable tests (TURBO_LIVE_BUNDLE,
#                      TURBO_LIVE_RERANK_BUNDLE select the bundles)

TURBO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
OUT ?= $(TURBO_ROOT)/build/metal
CXX ?= clang++
CC ?= clang

INCLUDES := -I$(TURBO_ROOT)/include -I$(TURBO_ROOT)/native/provider_common \
            -I$(TURBO_ROOT)/native/wordpiece -I$(TURBO_ROOT)/native/wordpiece/include \
            -I$(TURBO_ROOT)/third_party -I$(TURBO_ROOT)/third_party/utf8proc
WARN := -Wall -Wextra -Werror=return-type
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 -fvisibility=hidden -fvisibility-inlines-hidden $(WARN) $(INCLUDES)
OBJCXXFLAGS := -ObjC++ -fobjc-arc
CFLAGS ?= -O2
CFLAGS += -std=c11 -fvisibility=hidden -DUTF8PROC_STATIC -I$(TURBO_ROOT)/third_party/utf8proc

LIB := $(OUT)/libturbo_provider_metal.dylib
TEST := $(OUT)/turbo_provider_metal_test
OBJS := $(OUT)/provider.o $(OUT)/vocab_load.o $(OUT)/encode.o $(OUT)/utf8proc.o

all: $(LIB)

$(OUT):
	mkdir -p $(OUT)

$(OUT)/provider.o: src/provider.mm src/safetensors.hpp src/kernels.inc | $(OUT)
	$(CXX) $(CXXFLAGS) $(OBJCXXFLAGS) -c $< -o $@

$(OUT)/vocab_load.o: $(TURBO_ROOT)/native/wordpiece/vocab_load.cpp | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/encode.o: $(TURBO_ROOT)/native/wordpiece/encode.cpp | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/utf8proc.o: $(TURBO_ROOT)/third_party/utf8proc/utf8proc.c | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJS) exports.txt
	$(CXX) -dynamiclib -o $@ $(OBJS) -framework Metal -framework Foundation \
	    -Wl,-exported_symbols_list,exports.txt -Wl,-install_name,@rpath/libturbo_provider_metal.dylib
	@nm -gU $@ | grep -q ' T _turbo_provider_get$$' || (echo "turbo_provider_get is not exported" && exit 1)

$(TEST): tests/provider_test.cpp $(LIB)
	$(CXX) -std=c++17 -O1 $(WARN) -I$(TURBO_ROOT)/include -DTURBO_PROVIDER_LIB_PATH='"$(LIB)"' $< -o $@

test: $(TEST)
	$(TEST)

clean:
	rm -rf $(OUT)

.PHONY: all test clean
