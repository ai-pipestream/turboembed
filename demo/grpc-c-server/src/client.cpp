// SPDX-License-Identifier: Apache-2.0
//
// Turbo gRPC demo client: print the server's Info, embed the texts given on
// the command line, and print their cosine similarities.
//
//   turbo-grpc-client [--target localhost:50051] text...
#include "turbo_demo.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    std::string target = "localhost:50051";
    std::vector<std::string> texts;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--target" && i + 1 < argc) {
            target = argv[++i];
        } else {
            texts.push_back(a);
        }
    }
    if (texts.empty()) {
        std::cerr << "usage: turbo-grpc-client [--target host:port] text...\n";
        return 2;
    }
    auto stub = turbo::demo::v1::Embedder::NewStub(grpc::CreateChannel(target, grpc::InsecureChannelCredentials()));
    turbo::demo::v1::InfoResponse info;
    {
        grpc::ClientContext cc;
        const grpc::Status st = stub->Info(&cc, turbo::demo::v1::InfoRequest(), &info);
        if (!st.ok()) {
            std::cerr << "Info failed: " << st.error_message() << "\n";
            return 1;
        }
    }
    std::cout << "server: " << info.model_id() << " (dim " << info.dim() << ") on " << info.device_name() << " ("
              << info.provider_id() << ":" << info.ordinal() << ", runtime " << info.runtime_version() << ")\n";
    turbo::demo::v1::EmbedRequest req;
    for (const auto &t : texts) {
        req.add_texts(t);
    }
    turbo::demo::v1::EmbedResponse resp;
    grpc::ClientContext cc;
    const grpc::Status st = stub->Embed(&cc, req, &resp);
    if (!st.ok()) {
        std::cerr << "Embed failed (" << st.error_code() << "): " << st.error_message() << "\n";
        return 1;
    }
    const int n = resp.embeddings_size();
    std::cout << "embeddings: " << n << " x " << resp.dim() << "\ncosine similarity:\n";
    for (int a = 0; a < n; ++a) {
        const auto &va = resp.embeddings(a).values();
        for (int b = 0; b < n; ++b) {
            const auto &vb = resp.embeddings(b).values();
            double dot = 0, na = 0, nb = 0;
            for (int k = 0; k < va.size(); ++k) {
                dot += double(va[k]) * vb[k];
                na += double(va[k]) * va[k];
                nb += double(vb[k]) * vb[k];
            }
            std::printf(" %6.3f", dot / (std::sqrt(na) * std::sqrt(nb)));
        }
        std::cout << "  " << texts[a] << "\n";
    }
    return 0;
}
