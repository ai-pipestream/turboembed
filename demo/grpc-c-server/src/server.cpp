// SPDX-License-Identifier: Apache-2.0
//
// Turbo gRPC demo server: one model on one device, a pool of sessions
// (one per worker), Embed and Info RPCs. Every libturbo failure becomes a
// gRPC status: INVALID_ARGUMENT for refused options and over-capacity
// requests, RESOURCE_EXHAUSTED when every session is busy, INTERNAL for
// anything else, always with the status name and the library's message.
//
//   turbo-grpc-server --bundle <dir> [--provider-lib <so>] [--provider <id> --ordinal <n>]
//                     [--listen 0.0.0.0:50051] [--sessions 2] [--max-batch 16]
#include "turbo/turbo.h"
#include "turbo_demo.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

turbo_text text_of(const std::string &s) { return turbo_text{s.data(), s.size()}; }

struct Failure {
    int32_t code;
    uint32_t field;
    std::string message;
};

/// Run a libturbo call and throw a Failure with the record's contents.
template <typename F>
void call(const char *what, F &&fn) {
    turbo_error err{};
    err.struct_size = sizeof err;
    const int32_t rc = fn(&err);
    if (rc != TURBO_OK) {
        throw Failure{rc, err.field, std::string(what) + ": " + turbo_status_name(rc) +
                                          (err.field ? " (field " + std::to_string(err.field) + ")" : "") + ": " + err.message};
    }
}

grpc::Status to_status(const Failure &f) {
    switch (f.code) {
    case TURBO_E_INVALID_ARGUMENT:
    case TURBO_E_INVALID_UTF8:
    case TURBO_E_INVALID_ENUM:
    case TURBO_E_UNSUPPORTED_OPTION:
    case TURBO_E_CAPACITY:
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, f.message);
    case TURBO_E_BUSY:
    case TURBO_E_OVERLOADED:
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, f.message);
    default:
        return grpc::Status(grpc::StatusCode::INTERNAL, f.message);
    }
}

/// A session with the lock that serializes its use; the pool hands them out.
struct Worker {
    turbo_session *session = nullptr;
    std::mutex mu;
};

class EmbedderService final : public turbo::demo::v1::Embedder::Service {
  public:
    EmbedderService(turbo_model *model, turbo_model_info info, turbo_device_info device, uint32_t sessions,
                    uint32_t max_batch)
        : model_(model), info_(info), device_(device), max_batch_(max_batch) {
        for (uint32_t i = 0; i < sessions; ++i) {
            auto w = std::make_unique<Worker>();
            turbo_session_desc sd{};
            sd.struct_size = sizeof sd;
            sd.max_batch = max_batch;
            sd.max_seq = 0; // the bundle's limit
            call("turbo_session_create", [&](turbo_error *e) { return turbo_session_create(model, &sd, &w->session, e); });
            workers_.push_back(std::move(w));
        }
    }

    ~EmbedderService() override {
        for (auto &w : workers_) {
            turbo_session_release(w->session);
        }
        turbo_model_release(model_);
    }

    grpc::Status Info(grpc::ServerContext *, const turbo::demo::v1::InfoRequest *,
                      turbo::demo::v1::InfoResponse *out) override {
        out->set_device_name(device_.name);
        out->set_provider_id(device_.provider_id);
        out->set_ordinal(device_.ordinal);
        out->set_runtime_version(device_.runtime_version);
        out->set_model_id(info_.model_id);
        out->set_dim(info_.dim);
        out->set_max_seq(info_.max_seq);
        out->set_max_batch(max_batch_);
        out->set_fully_accelerated(info_.fully_accelerated != 0);
        return grpc::Status::OK;
    }

    grpc::Status Embed(grpc::ServerContext *, const turbo::demo::v1::EmbedRequest *req,
                       turbo::demo::v1::EmbedResponse *out) override {
        const uint32_t n = static_cast<uint32_t>(req->texts_size());
        if (n == 0) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "no texts");
        }
        if (n > max_batch_) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                "request has " + std::to_string(n) + " texts but the server's batch is " +
                                    std::to_string(max_batch_) + "; split the request");
        }
        // Take the first free worker; none free is RESOURCE_EXHAUSTED, not a queue.
        for (auto &w : workers_) {
            std::unique_lock<std::mutex> lock(w->mu, std::try_to_lock);
            if (!lock.owns_lock()) {
                continue;
            }
            try {
                std::vector<turbo_text> views;
                views.reserve(n);
                for (const auto &t : req->texts()) {
                    views.push_back(text_of(t));
                }
                turbo_embed_options opts{};
                opts.struct_size = sizeof opts;
                opts.max_tokens = req->max_tokens();
                call("turbo_session_write_text",
                     [&](turbo_error *e) { return turbo_session_write_text(w->session, views.data(), n, &opts, e); });
                turbo_result *result = nullptr;
                call("turbo_session_run", [&](turbo_error *e) { return turbo_session_run(w->session, nullptr, &result, e); });
                turbo_result_info ri{};
                ri.struct_size = sizeof ri;
                try {
                    call("turbo_result_get_info", [&](turbo_error *e) { return turbo_result_get_info(result, &ri, e); });
                    std::vector<float> buf(ri.bytes / sizeof(float));
                    uint64_t written = 0;
                    call("turbo_result_read", [&](turbo_error *e) {
                        return turbo_result_read(result, 0, buf.data(), ri.bytes, &written, e);
                    });
                    out->set_dim(ri.dim);
                    for (uint32_t r = 0; r < ri.batch; ++r) {
                        auto *emb = out->add_embeddings();
                        emb->mutable_values()->Add(buf.begin() + r * ri.dim, buf.begin() + (r + 1) * ri.dim);
                    }
                } catch (...) {
                    turbo_result_release(result);
                    throw;
                }
                turbo_result_release(result);
                return grpc::Status::OK;
            } catch (const Failure &f) {
                return to_status(f);
            }
        }
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "every session is busy; retry");
    }

  private:
    turbo_model *model_;
    turbo_model_info info_;
    turbo_device_info device_;
    uint32_t max_batch_;
    std::vector<std::unique_ptr<Worker>> workers_;
};

} // namespace

int main(int argc, char **argv) {
    std::string bundle, provider_lib, provider, listen = "0.0.0.0:50051";
    uint32_t ordinal = 0, sessions = 2, max_batch = 16;
    bool have_ordinal = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << name << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--bundle") bundle = next("--bundle");
        else if (a == "--provider-lib") provider_lib = next("--provider-lib");
        else if (a == "--provider") provider = next("--provider");
        else if (a == "--ordinal") { ordinal = std::stoul(next("--ordinal")); have_ordinal = true; }
        else if (a == "--listen") listen = next("--listen");
        else if (a == "--sessions") sessions = std::stoul(next("--sessions"));
        else if (a == "--max-batch") max_batch = std::stoul(next("--max-batch"));
        else { std::cerr << "unknown argument " << a << "\n"; return 2; }
    }
    if (bundle.empty() || provider.empty() != !have_ordinal) {
        std::cerr << "usage: turbo-grpc-server --bundle <dir> [--provider-lib <so>] [--provider <id> --ordinal <n>] "
                     "[--listen host:port] [--sessions n] [--max-batch n]\n";
        return 2;
    }
    try {
        turbo_runtime_desc rd{};
        rd.struct_size = sizeof rd;
        turbo_text lib_text{};
        if (!provider_lib.empty()) {
            lib_text = text_of(provider_lib);
            rd.n_provider_paths = 1;
            rd.provider_paths = &lib_text;
        }
        turbo_runtime *rt = nullptr;
        call("turbo_runtime_create", [&](turbo_error *e) { return turbo_runtime_create(&rd, &rt, e); });
        uint32_t dev = 0;
        if (!provider.empty()) {
            turbo_device_selector sel{};
            sel.struct_size = sizeof sel;
            sel.policy = TURBO_SELECT_EXPLICIT;
            sel.provider_id = text_of(provider);
            sel.ordinal = ordinal;
            call("turbo_runtime_select_device", [&](turbo_error *e) { return turbo_runtime_select_device(rt, &sel, &dev, e); });
        } else {
            call("turbo_runtime_select_device", [&](turbo_error *e) { return turbo_runtime_select_device(rt, nullptr, &dev, e); });
        }
        turbo_device_info di{};
        di.struct_size = sizeof di;
        call("turbo_runtime_device_info", [&](turbo_error *e) { return turbo_runtime_device_info(rt, dev, &di, e); });
        turbo_context *ctx = nullptr;
        call("turbo_context_create", [&](turbo_error *e) { return turbo_context_create(rt, dev, nullptr, &ctx, e); });
        turbo_model *model = nullptr;
        call("turbo_model_load", [&](turbo_error *e) { return turbo_model_load(ctx, text_of(bundle), nullptr, &model, e); });
        turbo_model_info mi{};
        mi.struct_size = sizeof mi;
        call("turbo_model_get_info", [&](turbo_error *e) { return turbo_model_get_info(model, &mi, e); });
        if (mi.kind != TURBO_MODEL_EMBEDDING) {
            std::cerr << bundle << " is not an embedding bundle (kind " << mi.kind << ")\n";
            return 1;
        }
        // The service holds the model; the context and runtime stay alive through it.
        turbo_context_release(ctx);
        turbo_runtime_release(rt);
        EmbedderService service(model, mi, di, sessions, max_batch);
        grpc::ServerBuilder builder;
        builder.AddListeningPort(listen, grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
        if (!server) {
            std::cerr << "could not listen on " << listen << "\n";
            return 1;
        }
        std::cout << "turbo-grpc-server listening on " << listen << ": " << mi.model_id << " (dim " << mi.dim << ") on "
                  << di.name << " (" << di.provider_id << ":" << di.ordinal << "), " << sessions << " sessions x batch "
                  << max_batch << std::endl;
        server->Wait();
    } catch (const Failure &f) {
        std::cerr << "error: " << f.message << "\n";
        return 1;
    }
    return 0;
}
