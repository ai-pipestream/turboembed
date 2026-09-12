import Foundation
import GRPCCore

/// Bearer-token interceptor. Same contract as the Rust façade:
/// `authorization: Bearer <token>`, constant-time compare per candidate.
struct BearerInterceptor: ServerInterceptor {
    let tokens: Set<String>

    init(tokens: Set<String>) {
        self.tokens = tokens
    }

    func intercept<Input: Sendable, Output: Sendable>(
        request: StreamingServerRequest<Input>,
        context: ServerContext,
        next: @Sendable (
            StreamingServerRequest<Input>, ServerContext
        ) async throws -> StreamingServerResponse<Output>
    ) async throws -> StreamingServerResponse<Output> {
        let header = request.metadata[stringValues: "authorization"].first { _ in true }
        guard let header else {
            throw RPCError(code: .unauthenticated, message: "missing authorization metadata")
        }
        let token: String
        if header.hasPrefix("Bearer ") {
            token = String(header.dropFirst("Bearer ".count))
        } else if header.hasPrefix("bearer ") {
            token = String(header.dropFirst("bearer ".count))
        } else {
            throw RPCError(
                code: .unauthenticated,
                message: "authorization metadata must be \"Bearer <token>\"")
        }
        if tokens.contains(where: { constantTimeEq($0, token) }) {
            return try await next(request, context)
        }
        throw RPCError(code: .unauthenticated, message: "invalid bearer token")
    }
}

private func constantTimeEq(_ a: String, _ b: String) -> Bool {
    let aa = Array(a.utf8)
    let bb = Array(b.utf8)
    if aa.count != bb.count { return false }
    var acc: UInt8 = 0
    for (x, y) in zip(aa, bb) { acc |= x ^ y }
    return acc == 0
}
