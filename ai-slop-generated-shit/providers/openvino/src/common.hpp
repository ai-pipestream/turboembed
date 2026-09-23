// SPDX-License-Identifier: Apache-2.0
//
// The OpenVINO provider's helpers live in the shared C++ provider header
// (native/provider_common); this file keeps them visible under the
// provider's own namespace.
#pragma once

#include "turbo_provider_common.hpp"

namespace turbo_ov {
using namespace turbo_pc; // NOLINT(google-build-using-namespace)
} // namespace turbo_ov
