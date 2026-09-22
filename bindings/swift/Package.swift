// swift-tools-version:5.9
// Pipestream Turbo for Swift: a thin, typed layer over libturbo's C ABI.
//
// `CTurbo` exposes include/turbo/turbo.h through a clang module;
// `PipestreamTurbo` is
// the Swift API. The tests run the conformance cases against the mock
// provider and need libturbo on the dynamic loader path
// (DYLD_LIBRARY_PATH=../../target/debug).
import PackageDescription

// libturbo from the repository's cargo target directory; an installed
// package supplies its own search path.
let turboLink: [LinkerSetting] = [
    .linkedLibrary("turbo"),
    .unsafeFlags(["-L" + Context.packageDirectory + "/../../target/debug",
                  "-L" + Context.packageDirectory + "/../../target/release"]),
]

let package = Package(
    name: "PipestreamTurbo",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "PipestreamTurbo", targets: ["PipestreamTurbo"]),
    ],
    targets: [
        .target(
            name: "CTurbo",
            path: "Sources/CTurbo",
            // The generated headers live at the repository root; the shim
            // header includes them by relative path (SwiftPM allows no
            // header search path outside the package).
            linkerSettings: turboLink
        ),
        // Not named `Turbo`: on a case-insensitive file system the module's own
        // static library, libTurbo.a, would satisfy the linker's `-lturbo`
        // before libturbo itself.
        .target(name: "PipestreamTurbo", dependencies: ["CTurbo"], path: "Sources/PipestreamTurbo", linkerSettings: turboLink),
        // The conformance cases as an executable: `swift run turbo-conformance`
        // (no XCTest or Swift Testing dependency, so it runs with the command
        // line tools alone).
        .executableTarget(name: "turbo-conformance", dependencies: ["PipestreamTurbo"], path: "Sources/TurboConformance", linkerSettings: turboLink),
    ],
    swiftLanguageVersions: [.v5]
)
