# Turbo C demo. Builds against the repository's header and libturbo:
#   cargo build -p turbo-shared      (from the repository root; target/debug/libturbo.so)
#   make -C demo/c && make -C demo/c test
ROOT := $(abspath $(CURDIR)/../..)
LIBDIR ?= $(ROOT)/target/debug
CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -Werror -O2
BUNDLE ?= $(ROOT)/testdata/bundles/mock/embedding

turbo-demo-c: main.c $(ROOT)/include/turbo/turbo.h
	$(CC) $(CFLAGS) -I$(ROOT)/include main.c -L$(LIBDIR) -lturbo -lm -Wl,-rpath,$(LIBDIR) -o $@

test: turbo-demo-c
	./turbo-demo-c --bundle $(BUNDLE) "a brown dog runs through the grass" "a dog is running on the lawn" "the stock market closed higher"

clean:
	rm -f turbo-demo-c

.PHONY: test clean
