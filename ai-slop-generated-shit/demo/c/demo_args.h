/* SPDX-License-Identifier: Apache-2.0
 *
 * Command-line helpers shared by the C demos. Every one of them fails with
 * a message naming the flag and exits 2; none substitutes a default for a
 * value the caller got wrong, and none accepts an argument it does not
 * understand.
 */
#ifndef TURBO_DEMO_ARGS_H
#define TURBO_DEMO_ARGS_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* The value that follows argv[*i], advancing *i; a flag at the end of the
 * command line is an error, not an empty value. */
static const char *demo_value(int argc, char **argv, int *i) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s needs a value\n", argv[*i]);
        exit(2);
    }
    *i += 1;
    return argv[*i];
}

/* strtoull with errno, the end pointer, and the range all checked. */
static unsigned long long demo_u64(const char *flag, const char *s, unsigned long long max) {
    char *end = NULL;
    unsigned long long v;
    if (s[0] == '\0' || s[0] == '-') {
        fprintf(stderr, "%s: %s is not a number in 0..%llu\n", flag, s, max);
        exit(2);
    }
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > max) {
        fprintf(stderr, "%s: %s is not a number in 0..%llu\n", flag, s, max);
        exit(2);
    }
    return v;
}

static uint32_t demo_u32(const char *flag, const char *s) {
    return (uint32_t)demo_u64(flag, s, 0xFFFFFFFFULL);
}

/* An argument that looks like a flag but is not one of ours is a typo, and
 * embedding or summarizing it would look like a plausible right answer. */
static void demo_reject_unknown_flag(const char *arg) {
    if (arg[0] == '-' && arg[1] == '-') {
        fprintf(stderr, "unknown argument %s\n", arg);
        exit(2);
    }
}

#endif /* TURBO_DEMO_ARGS_H */
