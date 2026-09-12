import Foundation
import Tokenizers

public struct TokenizeOptions: Sendable {
    public var addSpecialTokens: Bool
    public var withOffsets: Bool
    public var truncateTo: Int?
    public var padToLongest: Bool

    public init(
        addSpecialTokens: Bool = true,
        withOffsets: Bool = false,
        truncateTo: Int? = nil,
        padToLongest: Bool = false
    ) {
        self.addSpecialTokens = addSpecialTokens
        self.withOffsets = withOffsets
        self.truncateTo = truncateTo
        self.padToLongest = padToLongest
    }
}

public struct TokenEncoding: Sendable {
    public var inputIds: [UInt32]
    public var attentionMask: [UInt32]
    public var tokens: [String]
    public var offsets: [(UInt32, UInt32)]

    public init(
        inputIds: [UInt32],
        attentionMask: [UInt32],
        tokens: [String],
        offsets: [(UInt32, UInt32)]
    ) {
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.tokens = tokens
        self.offsets = offsets
    }
}

public enum TokenizerError: Error, LocalizedError, Sendable {
    case missing(String)
    case failed(String)

    public var errorDescription: String? {
        switch self {
        case .missing(let path): "failed to load tokenizer from \(path)"
        case .failed(let message): message
        }
    }
}

/// HuggingFace `tokenizer.json` loaded via swift-transformers. No Python.
public final class LocalTokenizer: @unchecked Sendable {
    private let inner: any Tokenizer
    private let lock = NSLock()

    public init(inner: any Tokenizer) {
        self.inner = inner
    }

    public static func load(path: String, configURL: URL? = nil) async throws -> LocalTokenizer {
        let resolved = Paths.resolveExistingFileOrDir(path, configURL: configURL)
        var folder = resolved
        let attrs = try? FileManager.default.attributesOfItem(atPath: resolved.path)
        let isDir = (attrs?[.type] as? FileAttributeType) == .typeDirectory
        if !isDir {
            folder = resolved.deletingLastPathComponent()
        }
        let json = folder.appending(path: "tokenizer.json")
        guard FileManager.default.fileExists(atPath: json.path) else {
            throw TokenizerError.missing(json.path)
        }
        if FileManager.default.fileExists(atPath: folder.appending(path: "config.json").path)
            || FileManager.default.fileExists(
                atPath: folder.appending(path: "tokenizer_config.json").path)
        {
            do {
                let tokenizer = try await AutoTokenizer.from(modelFolder: folder)
                return LocalTokenizer(inner: tokenizer)
            } catch {
                // tokenizer.json-only folders (GGUF sidecar) have no model config.json
            }
        }
        throw TokenizerError.missing(
            "\(folder.path) (need tokenizer.json plus tokenizer_config.json or a sibling MLX dir)")
    }

    public func tokenize(_ texts: [String], options: TokenizeOptions) throws -> [TokenEncoding] {
        lock.lock()
        defer { lock.unlock() }
        var encodings: [TokenEncoding] = texts.map { text in
            var ids = inner.encode(text: text, addSpecialTokens: options.addSpecialTokens).map {
                UInt32($0)
            }
            if let max = options.truncateTo, ids.count > max {
                ids = Array(ids.prefix(max))
            }
            let tokens = ids.map { inner.convertIdToToken(Int($0)) ?? "" }
            let mask = [UInt32](repeating: 1, count: ids.count)
            return TokenEncoding(
                inputIds: ids, attentionMask: mask, tokens: tokens, offsets: [])
        }
        if options.padToLongest {
            let longest = encodings.map(\.inputIds.count).max() ?? 0
            let padId = UInt32(
                inner.convertTokenToId("[PAD]")
                    ?? inner.convertTokenToId("<pad>")
                    ?? 0)
            for i in encodings.indices {
                let pad = longest - encodings[i].inputIds.count
                if pad > 0 {
                    encodings[i].inputIds.append(contentsOf: repeatElement(padId, count: pad))
                    encodings[i].attentionMask.append(contentsOf: repeatElement(0, count: pad))
                    encodings[i].tokens.append(contentsOf: repeatElement("[PAD]", count: pad))
                }
            }
        }
        return encodings
    }

    public func detokenize(_ sequences: [[UInt32]], skipSpecialTokens: Bool) throws -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return sequences.map { ids in
            inner.decode(tokens: ids.map { Int($0) }, skipSpecialTokens: skipSpecialTokens)
        }
    }
}

public final class TokenizerMap: Sendable {
    private let map: [String: LocalTokenizer]

    public init(_ map: [String: LocalTokenizer]) {
        self.map = map
    }

    public func get(_ name: String) -> LocalTokenizer? { map[name] }

    public func contains(_ name: String) -> Bool { map[name] != nil }

    public static func load(models: [ModelConfig], configURL: URL?) async -> TokenizerMap {
        var map: [String: LocalTokenizer] = [:]
        for model in models {
            let candidates = [model.tokenizerDir, model.path].compactMap { $0 }.filter { !$0.isEmpty }
            var loaded: LocalTokenizer?
            var lastError: (any Error)?
            for dir in candidates {
                do {
                    loaded = try await LocalTokenizer.load(path: dir, configURL: configURL)
                    break
                } catch {
                    lastError = error
                }
            }
            if let loaded {
                map[model.name] = loaded
            } else if let lastError {
                fputs(
                    "[inferstream-apple] tokenizer for \(model.name) failed: \(lastError)\n",
                    stderr)
            }
        }
        return TokenizerMap(map)
    }
}
