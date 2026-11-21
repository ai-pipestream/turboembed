/* SPDX-License-Identifier: Apache-2.0
 *
 * The few host compiler extensions the CUDA backend uses, spelled for
 * gcc and clang (Linux) and for MSVC (Windows), the host compilers nvcc
 * drives (#2453).
 */

#pragma once

#if defined(_MSC_VER)
#include <malloc.h>
#include <string.h>
#define TURBO_PRINTF(fmt, args)
#define TURBO_UNREACHABLE() __assume(0)
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define strtok_r strtok_s
/* MSVC has no aligned_alloc; its aligned blocks need their own free. */
#define turbo_aligned_alloc(align, n) _aligned_malloc((n), (align))
#define turbo_aligned_free(p) _aligned_free(p)
#else
#include <stdlib.h>
#include <strings.h>
#define turbo_aligned_alloc(align, n) aligned_alloc((align), (n))
#define turbo_aligned_free(p) free(p)
#define TURBO_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))
#define TURBO_UNREACHABLE() __builtin_unreachable()
#endif
