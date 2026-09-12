// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "inferstream-apple",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .executable(name: "inferstream-apple", targets: ["inferstream-apple"]),
        .library(name: "MlxEngine", targets: ["MlxEngine"]),
        .library(name: "InferstreamCore", targets: ["InferstreamCore"]),
        .library(name: "TurboEmbed", type: .dynamic, targets: ["TurboEmbed"]),
    ],
    dependencies: [
        .package(url: "https://github.com/grpc/grpc-swift-2.git", from: "2.1.0"),
        .package(url: "https://github.com/grpc/grpc-swift-protobuf.git", from: "2.1.0"),
        .package(url: "https://github.com/grpc/grpc-swift-nio-transport.git", from: "2.1.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.6"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.15"),
    ],
    targets: [
        .target(
            name: "MlxEngine",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MlxEngine",
            swiftSettings: [
                .enableUpcomingFeature("ExistentialAny")
            ]
        ),
        .target(
            name: "InferstreamCore",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/InferstreamCore"
        ),
        .executableTarget(
            name: "inferstream-apple",
            dependencies: [
                "MlxEngine",
                "InferstreamCore",
                .product(name: "GRPCCore", package: "grpc-swift-2"),
                .product(name: "GRPCNIOTransportHTTP2", package: "grpc-swift-nio-transport"),
                .product(name: "GRPCProtobuf", package: "grpc-swift-protobuf"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/InferstreamApple",
            plugins: [
                .plugin(name: "GRPCProtobufGenerator", package: "grpc-swift-protobuf")
            ]
        ),
        .target(
            name: "TurboEmbedC",
            path: "Sources/TurboEmbedC",
            publicHeadersPath: "include"
        ),
        .target(
            name: "TurboEmbed",
            dependencies: ["TurboEmbedC", "MlxEngine", "InferstreamCore"],
            path: "Sources/TurboEmbed",
            swiftSettings: [
                .enableUpcomingFeature("ExistentialAny")
            ]
        ),
        .testTarget(
            name: "InferstreamAppleTests",
            dependencies: ["InferstreamCore"],
            path: "Tests/InferstreamAppleTests"
        ),
    ]
)
