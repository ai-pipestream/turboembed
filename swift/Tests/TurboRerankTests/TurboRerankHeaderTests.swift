#if canImport(TurboRerankC)
import TurboRerankC
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
}
