# Turbo C demo. Builds against the repository's header and libturbo:
#   cargo build -p turbo-shared      (from the repository root; target/debug/libturbo.so)
#   make -C demo/c && make -C demo/c test
ROOT := $(abspath $(CURDIR)/../..)
LIBDIR ?= $(ROOT)/target/debug
CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -Werror -O2
BUNDLE ?= $(ROOT)/testdata/bundles/mock/embedding

GEN_BUNDLE ?= $(ROOT)/testdata/bundles/mock/generative

all: turbo-demo-c turbo-summarize

turbo-demo-c: main.c demo_args.h $(ROOT)/include/turbo/turbo.h
	$(CC) $(CFLAGS) -I$(ROOT)/include main.c -L$(LIBDIR) -lturbo -lm -Wl,-rpath,$(LIBDIR) -o $@

turbo-summarize: summarize.c demo_args.h $(ROOT)/include/turbo/turbo.h
	$(CC) $(CFLAGS) -I$(ROOT)/include summarize.c -L$(LIBDIR) -lturbo -Wl,-rpath,$(LIBDIR) -o $@

test: turbo-demo-c turbo-summarize
	./turbo-demo-c --bundle $(BUNDLE) "a brown dog runs through the grass" "a dog is running on the lawn" "the stock market closed higher"
	echo "The quick brown fox jumps over the lazy dog. It does so every day." | ./turbo-summarize --bundle $(GEN_BUNDLE) --max-new-tokens 24
	echo "The quick brown fox jumps over the lazy dog. It does so every day." | ./turbo-summarize --bundle $(GEN_BUNDLE) --max-new-tokens 64 --max-chars 20

clean:
	rm -f turbo-demo-c turbo-summarize

.PHONY: all test clean
