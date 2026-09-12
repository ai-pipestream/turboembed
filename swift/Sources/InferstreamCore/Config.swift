import Foundation

public enum AuthMode: String, Sendable {
    case none
    case bearer
}

public struct AuthConfig: Sendable {
    public var mode: AuthMode
    public var bearerTokens: [String]

    public init(mode: AuthMode = .none, bearerTokens: [String] = []) {
        self.mode = mode
        self.bearerTokens = bearerTokens
    }

    public func effectiveTokens() -> Set<String> {
        var tokens = Set(bearerTokens.filter { !$0.isEmpty })
        if let env = ProcessInfo.processInfo.environment["INFERSTREAM_API_KEYS"] {
            for part in env.split(separator: ",") {
                let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { tokens.insert(trimmed) }
            }
        }
        return tokens
    }
}

public enum BackendKind: String, Sendable {
    case mock
    case mlx
    case llamaCpp = "llama-cpp"
    case ort
    case openvino
    case ovms
    case trtLlm = "trt-llm"
}

public struct ModelConfig: Sendable {
    public var name: String
    public var backend: BackendKind
    public var path: String?
    public var device: String?
    public var tokenizerDir: String?
    public var maxBatchSize: UInt32?
    public var normalize: Bool?
    public var pooling: String?
    public var maxSeqLen: UInt32?
    public var nCtx: UInt32?

    public init(
        name: String,
        backend: BackendKind,
        path: String? = nil,
        device: String? = nil,
        tokenizerDir: String? = nil,
        maxBatchSize: UInt32? = nil,
        normalize: Bool? = nil,
        pooling: String? = nil,
        maxSeqLen: UInt32? = nil,
        nCtx: UInt32? = nil
    ) {
        self.name = name
        self.backend = backend
        self.path = path
        self.device = device
        self.tokenizerDir = tokenizerDir
        self.maxBatchSize = maxBatchSize
        self.normalize = normalize
        self.pooling = pooling
        self.maxSeqLen = maxSeqLen
        self.nCtx = nCtx
    }
}

public struct ServerConfig: Sendable {
    public var listen: String
    public var auth: AuthConfig
    public var serve: [String]
    public var catalogPath: String?
    public var models: [ModelConfig]
    public var configURL: URL?

    public static func load(path: String) throws -> ServerConfig {
        let url = URL(filePath: path)
        let table = try Toml.parseFile(path)
        var auth = AuthConfig()
        if let authTable = table.table("auth") {
            if let mode = authTable.string("mode"), let parsed = AuthMode(rawValue: mode) {
                auth.mode = parsed
            }
            auth.bearerTokens = authTable.stringArray("bearer_tokens") ?? []
        }
        var models: [ModelConfig] = []
        if let rows = table.tables("models") {
            for row in rows {
                if let model = try? modelFromTable(row, name: row.string("name")) {
                    models.append(model)
                }
            }
        }
        return ServerConfig(
            listen: table.string("listen") ?? "127.0.0.1:8461",
            auth: auth,
            serve: table.stringArray("serve") ?? [],
            catalogPath: table.string("catalog"),
            models: models,
            configURL: url
        )
    }

    public mutating func expandServe(catalog: Catalog) throws {
        var seen = Set(models.map(\.name))
        for alias in serve {
            if seen.contains(alias) {
                throw ConfigError.duplicateModel(alias)
            }
            let resolved = try catalog.resolve(alias: alias, arch: .apple)
            models.append(resolved)
            seen.insert(alias)
        }
    }

    public func parseListen() throws -> (host: String, port: Int) {
        let parts = listen.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count >= 2, let port = Int(parts.last!) else {
            throw ConfigError.badListen(listen)
        }
        let host = parts.dropLast().joined(separator: ":")
        return (host.isEmpty ? "127.0.0.1" : host, port)
    }
}

public enum ConfigError: Error, LocalizedError, Sendable {
    case duplicateModel(String)
    case badListen(String)
    case missingName
    case unknownBackend(String)

    public var errorDescription: String? {
        switch self {
        case .duplicateModel(let name):
            "model \(name) is listed more than once"
        case .badListen(let listen):
            "cannot parse listen address \(listen)"
        case .missingName:
            "[[models]] entry is missing name"
        case .unknownBackend(let backend):
            "unknown backend \(backend)"
        }
    }
}

func modelFromTable(_ table: TomlTable, name: String?) throws -> ModelConfig {
    guard let name, !name.isEmpty else { throw ConfigError.missingName }
    guard let backendRaw = table.string("backend"),
        let backend = BackendKind(rawValue: backendRaw)
    else {
        throw ConfigError.unknownBackend(table.string("backend") ?? "")
    }
    return ModelConfig(
        name: name,
        backend: backend,
        path: table.string("path"),
        device: table.string("device"),
        tokenizerDir: table.string("tokenizer_dir"),
        maxBatchSize: table.int("max_batch_size").map { UInt32($0) },
        normalize: table.bool("normalize"),
        pooling: table.string("pooling"),
        maxSeqLen: table.int("max_seq_len").map { UInt32($0) },
        nCtx: table.int("n_ctx").map { UInt32($0) }
    )
}
