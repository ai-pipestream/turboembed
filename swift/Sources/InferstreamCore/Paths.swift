import Foundation

/// Resolve relative model / tokenizer paths against the workspace root.
public enum Paths {
    /// Directories to try when expanding a relative catalog path.
    public static func searchRoots(configURL: URL? = nil) -> [URL] {
        var roots: [URL] = []
        if let env = ProcessInfo.processInfo.environment["INFERSTREAM_ROOT"], !env.isEmpty {
            roots.append(URL(filePath: env))
        }
        roots.append(URL(filePath: FileManager.default.currentDirectoryPath))
        if let configURL {
            // config/apple.toml → repo root
            roots.append(configURL.deletingLastPathComponent().deletingLastPathComponent())
            roots.append(configURL.deletingLastPathComponent())
        }
        var seen = Set<String>()
        return roots.filter { root in
            let key = root.standardizedFileURL.path
            if seen.contains(key) { return false }
            seen.insert(key)
            return true
        }
    }

    public static func resolveExistingDirectory(_ raw: String, configURL: URL? = nil) -> URL {
        let asURL = URL(filePath: raw)
        if asURL.hasDirectoryPath || FileManager.default.fileExists(atPath: asURL.path) {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: asURL.path, isDirectory: &isDir), isDir.boolValue
            {
                return asURL
            }
        }
        if let alias = aliasFromModel(raw) {
            for root in searchRoots(configURL: configURL) {
                let local = root.appending(path: "models/mlx").appending(path: alias)
                if directoryExists(local) { return local }
            }
        }
        if !asURL.isFileURL || !raw.hasPrefix("/") {
            for root in searchRoots(configURL: configURL) {
                let candidate = root.appending(path: raw)
                if directoryExists(candidate) { return candidate }
            }
        }
        return asURL
    }

    public static func resolveExistingFileOrDir(_ raw: String, configURL: URL? = nil) -> URL {
        let asURL = URL(filePath: raw)
        if FileManager.default.fileExists(atPath: asURL.path) {
            return asURL
        }
        for root in searchRoots(configURL: configURL) {
            let candidate = root.appending(path: raw)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        return asURL
    }

    public static func directoryExists(_ url: URL) -> Bool {
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
            && isDir.boolValue
    }

    public static func aliasFromModel(_ model: String) -> String? {
        if !model.contains("/") { return model }
        guard let last = model.split(separator: "/").last else { return nil }
        var name = String(last)
        if name.hasSuffix("-4bit") {
            name = String(name.dropLast("-4bit".count))
        }
        if name.hasSuffix("-Instruct") {
            name = String(name.dropLast("-Instruct".count))
        }
        return name
    }
}
