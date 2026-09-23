// swift-tools-version:6.0
// Turbo Swift demo: embed sentences on the best device through the
// PipestreamTurbo package (a path dependency on bindings/swift).
//
//   cargo build -p turbo-shared                       # target/debug/libturbo.dylib
//   cd demo/swift/TurboDemo && DYLD_LIBRARY_PATH=../../../target/debug swift run turbo-demo \
//       --bundle ../../../testdata/bundles/mock/embedding "text one" "text two"
//
// The package lives one directory down because SwiftPM derives a package's
// identity from its directory name, and demo/swift would collide with
// bindings/swift.
import PackageDescription

let package = Package(
    name: "turbo-demo-swift",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "PipestreamTurbo", path: "../../../bindings/swift")],
    targets: [
        .executableTarget(
            name: "turbo-demo",
            dependencies: [.product(name: "PipestreamTurbo", package: "PipestreamTurbo")],
            path: "Sources/TurboDemo"
        )
    ]
)
