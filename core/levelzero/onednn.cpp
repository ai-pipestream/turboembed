// oneDNN's matrix engines kernels for the levelzero backend's F16 linear
// layers (the levelzero-onednn feature, docs/levelzero.md): a C interface
// over oneDNN's SYCL runtime, on the backend's own Level Zero device,
// context and memory. The backend keeps its command list; this file keeps
// a SYCL queue on the same context, and the backend orders the two by
// host synchronisation around each run of oneDNN work.
//
// Built by core/build.rs with the oneAPI compiler into a shared library.
#include <level_zero/ze_api.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <cstdio>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <map>
#include <string>
#include <tuple>

namespace {

struct Handle {
    sycl::device device;
    sycl::context context;
    sycl::queue queue;
    dnnl::engine engine;
    dnnl::stream stream;
    // The matmul primitives by (m, k, n, gelu, residual), each with its
    // weights' layout, and the LayerNorms by (m, n).
    std::map<std::tuple<int, int, int, int, int>, dnnl::matmul> matmuls;
    std::map<std::tuple<int, int>, dnnl::memory::desc> weights;
    std::map<std::tuple<int, int>, dnnl::layer_normalization_forward> norms;
    // The events of the work queued since the last wait, kept so their
    // Level Zero handles stay valid for the backend's list to wait on.
    std::vector<sycl::event> events;
};

// The weights' layout the matmul asks for at the widest batch the backend
// runs, which every batch then uses.
constexpr int PACK_M = 8192;

void say(char *err, size_t n, const std::string &what) {
    if (err && n) {
        snprintf(err, n, "%s", what.c_str());
    }
}

template <typename F> int guarded(char *err, size_t n, F f) {
    try {
        f();
        return 0;
    } catch (const dnnl::error &e) {
        say(err, n, std::string("onednn: ") + e.what());
    } catch (const sycl::exception &e) {
        say(err, n, std::string("sycl: ") + e.what());
    } catch (const std::exception &e) {
        say(err, n, e.what());
    }
    return 1;
}

dnnl::memory::desc weights_desc(Handle &h, int k, int n) {
    auto key = std::make_tuple(k, n);
    auto it = h.weights.find(key);
    if (it != h.weights.end()) {
        return it->second;
    }
    using namespace dnnl;
    memory::desc a({PACK_M, k}, memory::data_type::f16, memory::format_tag::ab);
    memory::desc w({k, n}, memory::data_type::f16, memory::format_tag::any);
    memory::desc bias({1, n}, memory::data_type::f32, memory::format_tag::ab);
    memory::desc c({PACK_M, n}, memory::data_type::f16, memory::format_tag::ab);
    matmul::primitive_desc pd(h.engine, a, w, bias, c);
    h.weights.emplace(key, pd.weights_desc());
    return pd.weights_desc();
}

dnnl::matmul &matmul_for(Handle &h, int m, int k, int n, int gelu, int residual) {
    auto key = std::make_tuple(m, k, n, gelu, residual);
    auto it = h.matmuls.find(key);
    if (it != h.matmuls.end()) {
        return it->second;
    }
    using namespace dnnl;
    memory::desc a({m, k}, memory::data_type::f16, memory::format_tag::ab);
    memory::desc bias({1, n}, memory::data_type::f32, memory::format_tag::ab);
    memory::desc c({m, n}, memory::data_type::f16, memory::format_tag::ab);
    primitive_attr attr;
    post_ops po;
    if (residual) {
        po.append_binary(algorithm::binary_add, c);
    }
    if (gelu) {
        po.append_eltwise(algorithm::eltwise_gelu_erf, 0.f, 0.f);
    }
    attr.set_post_ops(po);
    matmul::primitive_desc pd(h.engine, a, weights_desc(h, k, n), bias, c, attr);
    return h.matmuls.emplace(key, matmul(pd)).first->second;
}

dnnl::layer_normalization_forward &norm_for(Handle &h, int m, int n, float eps) {
    auto key = std::make_tuple(m, n);
    auto it = h.norms.find(key);
    if (it != h.norms.end()) {
        return it->second;
    }
    using namespace dnnl;
    memory::desc x({m, n}, memory::data_type::f16, memory::format_tag::ab);
    layer_normalization_forward::primitive_desc pd(h.engine, prop_kind::forward_inference, x, x, eps,
                                                   normalization_flags::use_scale | normalization_flags::use_shift);
    return h.norms.emplace(key, layer_normalization_forward(pd)).first->second;
}

// The dependencies of a primitive: the backend's event, where given.
std::vector<sycl::event> deps_of(Handle &h, void *ze_wait) {
    std::vector<sycl::event> deps;
    if (ze_wait) {
        namespace lz = sycl::ext::oneapi::level_zero;
        deps.push_back(sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
            {(ze_event_handle_t)ze_wait, lz::ownership::keep}, h.context));
    }
    return deps;
}

// Keeps the primitive's event and hands out its Level Zero handle.
void *signal_of(Handle &h, sycl::event ev) {
    h.events.push_back(ev);
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(h.events.back());
}

dnnl::memory usm(Handle &h, const dnnl::memory::desc &md, const void *p) {
    return dnnl::sycl_interop::make_memory(md, h.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<void *>(p));
}

} // namespace

extern "C" {

void *turbo_dnnl_open(void *ze_device, void *ze_context, char *err, size_t n) {
    Handle *h = nullptr;
    int rc = guarded(err, n, [&] {
        namespace lz = sycl::ext::oneapi::level_zero;
        sycl::device dev = sycl::make_device<sycl::backend::ext_oneapi_level_zero>((ze_device_handle_t)ze_device);
        sycl::context ctx = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(
            {(ze_context_handle_t)ze_context, {dev}, lz::ownership::keep});
        sycl::queue q(ctx, dev, sycl::property::queue::in_order());
        dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        dnnl::stream st = dnnl::sycl_interop::make_stream(eng, q);
        h = new Handle{dev, ctx, q, eng, st, {}, {}, {}};
    });
    return rc ? nullptr : h;
}

void turbo_dnnl_close(void *h) {
    delete static_cast<Handle *>(h);
}

// Bytes the packed weights of a [k, n] F16 matrix take.
int turbo_dnnl_weights_bytes(void *hp, int k, int n, size_t *bytes, char *err, size_t n_err) {
    Handle &h = *static_cast<Handle *>(hp);
    return guarded(err, n_err, [&] { *bytes = weights_desc(h, k, n).get_size(); });
}

// src, [k, n] F16 row-major, packed into dst; waits.
int turbo_dnnl_pack(void *hp, const void *src, int k, int n, void *dst, char *err, size_t n_err) {
    Handle &h = *static_cast<Handle *>(hp);
    return guarded(err, n_err, [&] {
        using namespace dnnl;
        memory::desc s({k, n}, memory::data_type::f16, memory::format_tag::ab);
        memory from = usm(h, s, src), to = usm(h, weights_desc(h, k, n), dst);
        reorder(from, to).execute(h.stream, from, to);
        h.stream.wait();
    });
}

// c [m, n] F16 = a [m, k] F16 times the packed weights, plus bias (F32, n)
// where given, plus residual [m, n] F16 where given, then GELU where
// asked. Queued after the Level Zero event ze_wait where given; the
// event it signals is handed out in ze_signal, valid until turbo_dnnl_wait.
int turbo_dnnl_matmul(void *hp, const void *a, const void *packed, const float *bias, const void *residual, void *c,
                      int m, int k, int n, int gelu, void *ze_wait, void **ze_signal, char *err, size_t n_err) {
    Handle &h = *static_cast<Handle *>(hp);
    return guarded(err, n_err, [&] {
        using namespace dnnl;
        matmul &mm = matmul_for(h, m, k, n, gelu ? 1 : 0, residual ? 1 : 0);
        memory::desc amd({m, k}, memory::data_type::f16, memory::format_tag::ab);
        memory::desc biasmd({1, n}, memory::data_type::f32, memory::format_tag::ab);
        memory::desc cmd({m, n}, memory::data_type::f16, memory::format_tag::ab);
        if (!bias) throw std::runtime_error("a bias is required");
        std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, usm(h, amd, a)},
                                                {DNNL_ARG_WEIGHTS, usm(h, weights_desc(h, k, n), packed)},
                                                {DNNL_ARG_BIAS, usm(h, biasmd, bias)},
                                                {DNNL_ARG_DST, usm(h, cmd, c)}};
        if (residual) {
            args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, usm(h, cmd, residual));
        }
        *ze_signal = signal_of(h, sycl_interop::execute(mm, h.stream, args, deps_of(h, ze_wait)));
    });
}

// dst [m, n] F16 = LayerNorm of src over n, scaled by gamma and shifted by
// beta (F32, n). Queued.
int turbo_dnnl_layer_norm(void *hp, const void *src, const float *gamma, const float *beta, float eps, void *dst,
                          int m, int n, void *ze_wait, void **ze_signal, char *err, size_t n_err) {
    Handle &h = *static_cast<Handle *>(hp);
    return guarded(err, n_err, [&] {
        using namespace dnnl;
        layer_normalization_forward &ln = norm_for(h, m, n, eps);
        memory::desc x({m, n}, memory::data_type::f16, memory::format_tag::ab);
        memory::desc g({n}, memory::data_type::f32, memory::format_tag::a);
        std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, usm(h, x, src)},
                                                {DNNL_ARG_DST, usm(h, x, dst)},
                                                {DNNL_ARG_SCALE, usm(h, g, gamma)},
                                                {DNNL_ARG_SHIFT, usm(h, g, beta)}};
        *ze_signal = signal_of(h, sycl_interop::execute(ln, h.stream, args, deps_of(h, ze_wait)));
    });
}

// Waits for everything queued.
int turbo_dnnl_wait(void *hp, char *err, size_t n_err) {
    Handle &h = *static_cast<Handle *>(hp);
    return guarded(err, n_err, [&] {
        h.stream.wait();
        h.events.clear();
    });
}

const char *turbo_dnnl_version(void) {
    static char v[64];
    const dnnl_version_t *dv = dnnl_version();
    snprintf(v, sizeof v, "oneDNN %d.%d.%d", dv->major, dv->minor, dv->patch);
    return v;
}
}
