// swift-tools-version:6.0
// Turbo Swift demo: embed sentences on the best device through the
// PipestreamTurbo package (a path dependency on bindings/swift).
//
//   cargo build -p turbo-shared                       # target/debug/libturbo.dylib
//   cd demo/swift && DYLD_LIBRARY_PATH=../../target/debug swift run turbo-demo \
//       --bundle ../../testdata/bundles/mock/embedding "text one" "text two"
import PackageDescription

let package = Package(
    name: "turbo-demo-swift",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "PipestreamTurbo", path: "../../bindings/swift")],
    targets: [
        .executableTarget(
            name: "turbo-demo",
            dependencies: [.product(name: "PipestreamTurbo", package: "PipestreamTurbo")],
            path: "Sources/TurboDemo"
        )
    ]
)
