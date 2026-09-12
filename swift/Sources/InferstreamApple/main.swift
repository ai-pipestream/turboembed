import ArgumentParser
import Foundation

@main
struct InferstreamApple: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "inferstream-apple",
        abstract: "All-Swift gRPC inference server for Apple silicon (native MLX, no Rust façade)."
    )

    @Option(name: .long, help: "Path to apple.toml (reads config/catalog.toml for aliases).")
    var config: String = "config/apple.toml"

    func run() async throws {
        try await AppleServer.run(configPath: config)
    }
}
