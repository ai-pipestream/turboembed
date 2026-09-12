import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2
import InferstreamCore
import MlxEngine

enum AppleServer {
    static func run(configPath: String) async throws {
        var config = try ServerConfig.load(path: configPath)
        let catalog: Catalog
        if let catalogPath = config.catalogPath {
            catalog = try Catalog.load(
                path: Paths.resolveExistingFileOrDir(catalogPath, configURL: config.configURL).path)
        } else {
            catalog = try Catalog.builtin(relativeTo: config.configURL)
        }
        try config.expandServe(catalog: catalog)

        fputs("[inferstream-apple] loading tokenizers\n", stderr)
        let tokenizers = await TokenizerMap.load(
            models: config.models, configURL: config.configURL)

        let engine = Engine()
        let ping = try engine.ping()
        fputs(
            "[inferstream-apple] mlx device=\(ping.device) metal=\(ping.metalAvailable)\n",
            stderr)

        let registry = try Registry.build(config: config, engine: engine, tokenizers: tokenizers)
        let listen = try config.parseListen()
        let tokens = config.auth.effectiveTokens()

        var interceptors: [any ServerInterceptor] = []
        if config.auth.mode == .bearer {
            if tokens.isEmpty {
                throw ServeError.invalid(
                    "auth.mode = bearer but no tokens configured (bearer_tokens or INFERSTREAM_API_KEYS)"
                )
            }
            interceptors.append(BearerInterceptor(tokens: tokens))
        }

        let server = GRPCServer(
            transport: .http2NIOPosix(
                address: .ipv4(host: listen.host, port: listen.port),
                transportSecurity: .plaintext
            ),
            services: [
                OIPService(registry: registry),
                ExtensionService(registry: registry, tokenizers: tokenizers),
            ],
            interceptors: interceptors
        )

        try await withThrowingDiscardingTaskGroup { group in
            group.addTask { try await server.serve() }
            if let address = try await server.listeningAddress {
                fputs("[inferstream-apple] listening on \(address)\n", stderr)
            }
        }
    }
}
