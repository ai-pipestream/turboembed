// SPDX-License-Identifier: Apache-2.0
//
// Turbo Swift demo: load a bundle on the best device, embed sentences, and
// print their cosine similarities. Every failure is a TurboError with the
// status name, the field index when an option was refused, and the message.
import Foundation
import PipestreamTurbo

var providerLib: String? = nil
var provider: String? = nil
var ordinal: UInt32? = nil
var bundle: String? = nil
var texts: [String] = []
var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--provider-lib": providerLib = args.isEmpty ? nil : args.removeFirst()
    case "--provider": provider = args.isEmpty ? nil : args.removeFirst()
    case "--ordinal": ordinal = args.isEmpty ? nil : UInt32(args.removeFirst())
    case "--bundle": bundle = args.isEmpty ? nil : args.removeFirst()
    default: texts.append(a)
    }
}
guard let bundle, !texts.isEmpty else {
    FileHandle.standardError.write("usage: turbo-demo [--provider-lib <dylib>] [--provider <id> --ordinal <n>] --bundle <dir> text...\n".data(using: .utf8)!)
    exit(2)
}
guard (provider == nil) == (ordinal == nil) else {
    FileHandle.standardError.write("--provider and --ordinal go together; omit both for AUTO\n".data(using: .utf8)!)
    exit(2)
}

func cosine(_ a: [Float], _ b: [Float]) -> Float {
    var dot: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0..<a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
    return dot / (na.squareRoot() * nb.squareRoot())
}

do {
    let rt = try Runtime(providerPaths: providerLib.map { [$0] } ?? [])
    let index: UInt32
    if let provider, let ordinal {
        index = try rt.selectDevice(policy: .explicit, providerId: provider, ordinal: ordinal)
    } else {
        index = try rt.selectDevice() // AUTO: never a CPU
    }
    let dev = try rt.device(index)
    print("device: \(dev.name) (\(dev.providerId):\(dev.ordinal), \(dev.kind), runtime \(dev.runtimeVersion))")
    let ctx = try rt.createContext(device: index)
    let model = try ctx.loadModel(bundlePath: bundle)
    let mi = model.info
    guard mi.task == .embed else {
        FileHandle.standardError.write("\(bundle) is not an embedding bundle (\(mi.task))\n".data(using: .utf8)!)
        exit(1)
    }
    print("model: \(mi.modelId) dim=\(mi.dim) max_seq=\(mi.maxSeq) provider=\(mi.providerId) fully_accelerated=\(mi.fullyAccelerated)")
    let session = try model.createSession(maxBatch: UInt32(texts.count))
    try session.writeText(texts)
    let result = try session.run()
    let (_, shape) = try result.output(0)
    let dim = Int(shape[1])
    let flat = try result.readFloats(0)
    print("embeddings: \(texts.count) x \(dim) (placement \(try result.placement))")
    result.close()
    let rows = (0..<texts.count).map { Array(flat[$0 * dim ..< ($0 + 1) * dim]) }
    print("cosine similarity:")
    for (i, row) in rows.enumerated() {
        let line = rows.map { String(format: " %6.3f", cosine(row, $0)) }.joined()
        print("\(line)  \(texts[i])")
    }
} catch let e as TurboError {
    FileHandle.standardError.write("error: \(e)\n".data(using: .utf8)!)
    exit(1)
}
