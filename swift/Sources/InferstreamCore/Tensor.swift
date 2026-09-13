import Foundation

/// OIP V2 raw-tensor helpers. Same wire rules as `crates/protocol/src/tensor.rs`.
public enum OipDataType: String, Sendable {
    case bool = "BOOL"
    case uint8 = "UINT8"
    case uint16 = "UINT16"
    case uint32 = "UINT32"
    case uint64 = "UINT64"
    case int8 = "INT8"
    case int16 = "INT16"
    case int32 = "INT32"
    case int64 = "INT64"
    case fp16 = "FP16"
    case bf16 = "BF16"
    case fp32 = "FP32"
    case fp64 = "FP64"
    case bytes = "BYTES"
}

public enum TensorError: Error, LocalizedError, Sendable {
    case truncatedBytes(String)
    case invalidShape([Int64])
    case lengthMismatch(String)

    public var errorDescription: String? {
        switch self {
        case .truncatedBytes(let m), .lengthMismatch(let m): m
        case .invalidShape(let shape): "invalid shape \(shape)"
        }
    }
}

public enum Tensor {
    public static func packBytes<S: Collection>(_ elements: S) -> Data where S.Element: Collection, S.Element.Element == UInt8 {
        var out = Data()
        for element in elements {
            let bytes = Array(element)
            var len = UInt32(bytes.count).littleEndian
            withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
            out.append(contentsOf: bytes)
        }
        return out
    }

    public static func packUTF8(_ texts: [String]) -> Data {
        packBytes(texts.map { Array($0.utf8) })
    }

    public static func unpackBytes(_ raw: Data) throws -> [Data] {
        var out: [Data] = []
        var cursor = 0
        let bytes = [UInt8](raw)
        while cursor < bytes.count {
            if cursor + 4 > bytes.count {
                throw TensorError.truncatedBytes(
                    "length prefix at offset \(cursor) runs past end (\(bytes.count) bytes total)")
            }
            let len = UInt32(littleEndian: readU32(bytes, cursor))
            cursor += 4
            let n = Int(len)
            if cursor + n > bytes.count {
                throw TensorError.truncatedBytes(
                    "element of \(n) bytes at offset \(cursor) runs past end (\(bytes.count) bytes total)"
                )
            }
            out.append(Data(bytes[cursor..<(cursor + n)]))
            cursor += n
        }
        return out
    }

    public static func unpackUTF8(_ raw: Data) throws -> [String] {
        try unpackBytes(raw).map { data in
            String(data: data, encoding: .utf8) ?? String(decoding: data, as: UTF8.self)
        }
    }

    public static func packFP32(_ values: [Float]) -> Data {
        var out = Data(capacity: values.count * 4)
        packFP32(values, into: &out)
        return out
    }

    /// Write LE FP32 into `out`, keeping capacity (gRPC output scratch).
    public static func packFP32(_ values: [Float], into out: inout Data) {
        out.removeAll(keepingCapacity: true)
        let nbytes = values.count * 4
        out.reserveCapacity(nbytes)
        values.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            base.withMemoryRebound(to: UInt8.self, capacity: nbytes) { bytes in
                out.append(bytes, count: nbytes)
            }
        }
    }

    public static func unpackFP32(_ raw: Data) throws -> [Float] {
        let bytes = [UInt8](raw)
        if bytes.count % 4 != 0 {
            throw TensorError.lengthMismatch(
                "FP32 blob length \(bytes.count) is not a multiple of 4")
        }
        var out: [Float] = []
        out.reserveCapacity(bytes.count / 4)
        var i = 0
        while i < bytes.count {
            let bits = UInt32(littleEndian: readU32(bytes, i))
            out.append(Float(bitPattern: bits))
            i += 4
        }
        return out
    }

    private static func readU32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        UInt32(bytes[offset])
            | UInt32(bytes[offset + 1]) << 8
            | UInt32(bytes[offset + 2]) << 16
            | UInt32(bytes[offset + 3]) << 24
    }
}
