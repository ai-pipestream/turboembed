import Foundation
import MLXLMCommon
import Tokenizers

/// Loads a HuggingFace `tokenizer.json` from a local model directory and
/// adapts it to mlx-swift-lm's `Tokenizer` protocol. No Python.
public struct HFTokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let inner = try await AutoTokenizer.from(modelFolder: directory)
        return HFTokenizerAdapter(inner: inner)
    }
}

struct HFTokenizerAdapter: MLXLMCommon.Tokenizer, @unchecked Sendable {
    let inner: any Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        inner.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        inner.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        inner.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        inner.convertIdToToken(id)
    }

    var bosToken: String? { inner.bosToken }
    var eosToken: String? { inner.eosToken }
    var unknownToken: String? { inner.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let mapped: [[String: String]] = messages.map { msg in
            var out: [String: String] = [:]
            for (k, v) in msg {
                if let s = v as? String {
                    out[k] = s
                }
            }
            return out
        }
        return try inner.applyChatTemplate(messages: mapped)
    }
}
