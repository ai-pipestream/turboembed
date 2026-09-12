import Foundation

public enum Arch: String, Sendable {
    case nvidia
    case intel
    case apple
}

public struct Catalog: Sendable {
    public var models: [String: CatalogEntry]

    public init(models: [String: CatalogEntry] = [:]) {
        self.models = models
    }

    public static func builtin(relativeTo configURL: URL? = nil) throws -> Catalog {
        let roots = Paths.searchRoots(configURL: configURL)
        for root in roots {
            let candidate = root.appending(path: "config/catalog.toml")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return try load(path: candidate.path)
            }
        }
        throw CatalogError.missingFile("config/catalog.toml")
    }

    public static func load(path: String) throws -> Catalog {
        let table = try Toml.parseFile(path)
        guard let modelsTable = table.table("models") else {
            return Catalog()
        }
        var models: [String: CatalogEntry] = [:]
        for (alias, value) in modelsTable.values {
            guard case .table(let entry) = value else { continue }
            models[alias] = CatalogEntry(
                description: entry.string("description"),
                nvidia: try entry.table("nvidia").map { try spec($0) },
                intel: try entry.table("intel").map { try spec($0) },
                apple: try entry.table("apple").map { try spec($0) }
            )
        }
        return Catalog(models: models)
    }

    public func resolve(alias: String, arch: Arch) throws -> ModelConfig {
        guard let entry = models[alias] else {
            let known = models.keys.sorted().joined(separator: ", ")
            throw CatalogError.unknownAlias(alias, known: known)
        }
        let spec: ModelConfig?
        switch arch {
        case .nvidia: spec = entry.nvidia
        case .intel: spec = entry.intel
        case .apple: spec = entry.apple
        }
        guard var resolved = spec else {
            var available: [String] = []
            if entry.nvidia != nil { available.append("nvidia") }
            if entry.intel != nil { available.append("intel") }
            if entry.apple != nil { available.append("apple") }
            throw CatalogError.notAvailable(alias, arch: arch.rawValue, available: available.joined(separator: ", "))
        }
        resolved.name = alias
        return resolved
    }

    public var aliases: [String] { models.keys.sorted() }
}

public struct CatalogEntry: Sendable {
    public var description: String?
    public var nvidia: ModelConfig?
    public var intel: ModelConfig?
    public var apple: ModelConfig?
}

public enum CatalogError: Error, LocalizedError, Sendable {
    case missingFile(String)
    case unknownAlias(String, known: String)
    case notAvailable(String, arch: String, available: String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let path):
            "catalog not found at \(path)"
        case .unknownAlias(let alias, let known):
            "unknown model alias \(alias); catalog defines: \(known)"
        case .notAvailable(let alias, let arch, let available):
            "model alias \(alias) is not available on the \(arch) arch (available on: \(available))"
        }
    }
}

private func spec(_ table: TomlTable) throws -> ModelConfig {
    try modelFromTable(table, name: "_")
}
