import Foundation

/// gRPC / OIP output freelist (SOLIDIFY 6).
///
/// Embed `PACKED_BYTES` and related LE FP32 blobs rent a pre-sized `Data`.
/// After warmup of a given byte count, `allocs` stays flat. Swift-Protobuf
/// may COW-copy when a slab is mutated while a prior response still holds
/// it; sequential unary after warmup does not grow.
public final class OutputScratch: @unchecked Sendable {
    public static let shared = OutputScratch()

    public init() {}

    private let lock = NSLock()
    private var byteSlabs: [Data] = []
    private var floatSlabs: [ContiguousArray<Float>] = []
    private var _allocs: UInt64 = 0
    private let maxFree = 32

    public var allocs: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return _allocs
    }

    public func resetCounters() {
        lock.lock()
        defer { lock.unlock() }
        _allocs = 0
    }

    public func rentBytes(minCap: Int) -> Data {
        lock.lock()
        defer { lock.unlock() }
        if let idx = bestFit(byteSlabs.map(\.capacity), min: minCap) {
            var slab = byteSlabs.remove(at: idx)
            slab.removeAll(keepingCapacity: true)
            return slab
        }
        _allocs += 1
        return Data(capacity: minCap)
    }

    public func recycleBytes(_ data: Data) {
        guard data.capacity > 0 else { return }
        lock.lock()
        defer { lock.unlock() }
        guard byteSlabs.count < maxFree else { return }
        var slab = data
        slab.removeAll(keepingCapacity: true)
        byteSlabs.append(slab)
    }

    public func rentFloats(minCap: Int) -> ContiguousArray<Float> {
        lock.lock()
        defer { lock.unlock() }
        if let idx = bestFit(floatSlabs.map(\.capacity), min: minCap) {
            var slab = floatSlabs.remove(at: idx)
            slab.removeAll(keepingCapacity: true)
            return slab
        }
        _allocs += 1
        var slab = ContiguousArray<Float>()
        slab.reserveCapacity(minCap)
        return slab
    }

    public func recycleFloats(_ values: ContiguousArray<Float>) {
        guard values.capacity > 0 else { return }
        lock.lock()
        defer { lock.unlock() }
        guard floatSlabs.count < maxFree else { return }
        var slab = values
        slab.removeAll(keepingCapacity: true)
        floatSlabs.append(slab)
    }

    private func bestFit(_ caps: [Int], min: Int) -> Int? {
        caps.enumerated()
            .filter { $0.element >= min }
            .min { $0.element < $1.element }
            .map(\.offset)
    }
}
