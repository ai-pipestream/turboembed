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

#include <cstdint>
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
    // The matmul primitives by (m, k, n, gelu, residual, f32, bias) and the
    // LayerNorms by (m, n, eps's bits): the context's models share them, and
    // models of one width may differ in eps.
    std::map<std::tuple<int, int, int, int, int, int, int>, dnnl::matmul> matmuls;
    std::map<std::tuple<int, int, uint32_t>, dnnl::layer_normalization_forward> norms;
    // The events of the work queued since the last wait, kept so their
    // Level Zero handles stay valid for the backend's list to wait on.
    std::vector<sycl::event> events;
};

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

// The operands' type: F32, or F16 for the matrix engines.
dnnl::memory::data_type type_of(int f32) {
    return f32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::f16;
}

// The weights' layout: [k, n] stored transposed, n rows of k, which is a
// linear layer's own [out, in]. Left to choose, oneDNN pads n to a
// multiple of 32 and keeps the rows of n, and its F16 kernels on a B70 run
// 2% slower from that on bge-base and bge-large; the transposed plain
// layout is what OpenVINO hands it.
dnnl::memory::desc weights_desc(int k, int n, int f32 = 0) {
    using namespace dnnl;
    return memory::desc({k, n}, type_of(f32), memory::format_tag::ba);
}

// The primitive for the shape, built and kept the first time; *built says
// whether this call built it.
dnnl::matmul &matmul_for(Handle &h, int m, int k, int n, int gelu, int residual, int f32, int has_bias, int *built) {
    auto key = std::make_tuple(m, k, n, gelu, residual, f32, has_bias);
    auto it = h.matmuls.find(key);
    *built = it == h.matmuls.end();
    if (!*built) {
        return it->second;
    }
    using namespace dnnl;
    memory::desc a({m, k}, type_of(f32), memory::format_tag::ab);
    memory::desc bias = has_bias ? memory::desc({1, n}, memory::data_type::f32, memory::format_tag::ab)
                                 : memory::desc();
    memory::desc c({m, n}, type_of(f32), memory::format_tag::ab);
    primitive_attr attr;
    post_ops po;
    if (residual) {
        po.append_binary(algorithm::binary_add, c);
    }
    if (gelu) {
        po.append_eltwise(algorithm::eltwise_gelu_erf, 0.f, 0.f);
    }
    attr.set_post_ops(po);
    matmul::primitive_desc pd(h.engine, a, weights_desc(k, n, f32), bias, c, attr);
    return h.matmuls.emplace(key, matmul(pd)).first->second;
}

dnnl::layer_normalization_forward &norm_for(Handle &h, int m, int n, float eps, int *built) {
    uint32_t bits;
    std::memcpy(&bits, &eps, sizeof bits);
    auto key = std::make_tuple(m, n, bits);
    auto it = h.norms.find(key);
    *built = it == h.norms.end();
    if (!*built) {
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

// Bytes the packed (transposed) weights of a [k, n] F16 matrix take.
int turbo_dnnl_weights_bytes(void *hp, int k, int n, size_t *bytes, char *err, size_t n_err) {
    Handle &h = *static_cast<Handle *>(hp);
    return guarded(err, n_err, [&] { *bytes = weights_desc(k, n).get_size(); });
}

// src, [k, n] F16 row-major, packed (transposed) into dst; waits.
int turbo_dnnl_pack(void *hp, const void *src, int k, int n, void *dst, char *err, size_t n_err) {
    Handle &h = *static_cast<Handle *>(hp);
    return guarded(err, n_err, [&] {
        using namespace dnnl;
        memory::desc s({k, n}, memory::data_type::f16, memory::format_tag::ab);
        memory from = usm(h, s, src), to = usm(h, weights_desc(k, n), dst);
        reorder(from, to).execute(h.stream, from, to);
        h.stream.wait();
    });
}

// c [m, n] = a [m, k] times the weights, [k, n] stored as [n, k], plus
// bias (F32, n) where given, plus residual [m, n] where given, then GELU
// where asked; in F32 where f32 is set, else in F16 from the packed
// weights. Queued after the Level Zero event ze_wait where given; the
// event it signals is handed out in ze_signal, valid until turbo_dnnl_wait.
// *built says whether the primitive was built for this call.
int turbo_dnnl_matmul(void *hp, const void *a, const void *packed, const float *bias, const void *residual, void *c,
                      int m, int k, int n, int gelu, int f32, void *ze_wait, void **ze_signal, int *built, char *err,
                      size_t n_err) {
    Handle &h = *static_cast<Handle *>(hp);
    return guarded(err, n_err, [&] {
        using namespace dnnl;
        matmul &mm = matmul_for(h, m, k, n, gelu ? 1 : 0, residual ? 1 : 0, f32 ? 1 : 0, bias ? 1 : 0, built);
        memory::desc amd({m, k}, type_of(f32), memory::format_tag::ab);
        memory::desc biasmd({1, n}, memory::data_type::f32, memory::format_tag::ab);
        memory::desc cmd({m, n}, type_of(f32), memory::format_tag::ab);
        std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, usm(h, amd, a)},
                                                {DNNL_ARG_WEIGHTS, usm(h, weights_desc(k, n, f32), packed)},
                                                {DNNL_ARG_DST, usm(h, cmd, c)}};
        if (bias) {
            args.emplace(DNNL_ARG_BIAS, usm(h, biasmd, bias));
        }
        if (residual) {
            args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, usm(h, cmd, residual));
        }
        *ze_signal = signal_of(h, sycl_interop::execute(mm, h.stream, args, deps_of(h, ze_wait)));
    });
}

// dst [m, n] F16 = LayerNorm of src over n, scaled by gamma and shifted by
// beta (F32, n). Queued; *built as for turbo_dnnl_matmul.
int turbo_dnnl_layer_norm(void *hp, const void *src, const float *gamma, const float *beta, float eps, void *dst,
                          int m, int n, void *ze_wait, void **ze_signal, int *built, char *err, size_t n_err) {
    Handle &h = *static_cast<Handle *>(hp);
    return guarded(err, n_err, [&] {
        using namespace dnnl;
        layer_normalization_forward &ln = norm_for(h, m, n, eps, built);
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
