/* SPDX-License-Identifier: Apache-2.0
 *
 * The few host compiler extensions the CUDA backend uses, spelled for
 * gcc and clang (Linux) and for MSVC (Windows), the host compilers nvcc
 * drives (#2453).
 */

#pragma once

#if defined(_MSC_VER)
#include <string.h>
#define TURBO_PRINTF(fmt, args)
#define TURBO_UNREACHABLE() __assume(0)
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#else
#include <strings.h>
#define TURBO_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))
#define TURBO_UNREACHABLE() __builtin_unreachable()
#endif
