#if canImport(TurboRerankC)
import TurboRerankC
#endif
#if canImport(TurboBufferC)
import TurboBufferC
#endif
import XCTest

final class TurboRerankHeaderTests: XCTestCase {
    func testAbiVersionIsOne() {
        XCTAssertEqual(turborerank_abi_version(), 1)
        let metal = String(cString: turborerank_device_name(TURBORERANK_DEVICE_METAL))
        XCTAssertEqual(metal, "METAL")
        let ok = String(cString: turborerank_status_name(TURBORERANK_OK))
        XCTAssertEqual(ok, "OK")
    }

    func testTurboBufferMetalDeviceIsSeven() {
        #if canImport(TurboBufferC)
        XCTAssertEqual(TURBO_BUFFER_ABI_VERSION, 1)
        XCTAssertEqual(Int(TURBO_BUFFER_DEVICE_METAL.rawValue), 7)
        XCTAssertEqual(Int(TURBO_BUFFER_PLACE_SHARED.rawValue), 3)
        #endif
    }

    /// The Swift client has no token-buffer allocator. Tokens come from
    /// `turborerank_buffer_alloc` / engine work rents in libturborerank_apple.
    func testSwiftClientHasNoTokenAllocatorSymbolInThisModule() {
        XCTAssertEqual(TURBORERANK_DEVICE_METAL, 7)
    }
}
