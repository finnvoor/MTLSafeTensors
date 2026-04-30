import Foundation
import Metal
@testable import MTLSafeTensors
import Testing

@Test func loadsMetadataAndBuffersWithoutCopyingTensorBytes() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let url = try makeSafeTensorsFile(
        tensors: [
            TensorFixture(name: "a", dtype: "F32", shape: [2], bytes: bytes([Float32(1), Float32(2)])),
            TensorFixture(name: "b", dtype: "I8", shape: [3], bytes: [3, 4, 5]),
        ],
        metadata: ["format": "test"]
    )
    defer { try? FileManager.default.removeItem(at: url) }

    let weights = try device.makeSafeTensors(from: url)

    #expect(weights.names == ["a", "b"])
    #expect(weights.metadata == ["format": "test"])
    let a = try weights.tensor(named: "a")
    #expect(a.dtype == .f32)
    #expect(a.shape == [2])
    #expect(a.elementCount == 2)
    #expect(a.name == "a")
    #expect(a.buffer.length == 8)
    #expect(a.buffer.label == "a")

    let values = a.buffer.contents().assumingMemoryBound(to: Float32.self)
    #expect(values[0] == 1)
    #expect(values[1] == 2)
}

@Test func nonMappedBuffersCanUseManagedStorage() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let url = try makeSafeTensorsFile(tensors: [
        TensorFixture(name: "x", dtype: "U8", shape: [1], bytes: [42])
    ])
    defer { try? FileManager.default.removeItem(at: url) }

    let weights = try device.makeSafeTensors(from: url, mmap: false)
    let x = try weights.tensor(named: "x", options: [.storageModeManaged])

    #expect(x.dtype == .u8)
    #expect(x.buffer.label == "x")
    #expect(x.buffer.storageMode == .managed)
}

@Test func nonMappedPrivateStorageThrowsWithoutUploadPath() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let url = try makeSafeTensorsFile(tensors: [
        TensorFixture(name: "x", dtype: "U8", shape: [1], bytes: [42])
    ])
    defer { try? FileManager.default.removeItem(at: url) }

    let weights = try device.makeSafeTensors(from: url, mmap: false)
    #expect(throws: SafeTensors.Error.self) {
        _ = try weights.tensor(named: "x", options: [.storageModePrivate])
    }
}

@Test func mappedPrivateStorageThrows() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let url = try makeSafeTensorsFile(tensors: [
        TensorFixture(name: "x", dtype: "U8", shape: [1], bytes: [42])
    ])
    defer { try? FileManager.default.removeItem(at: url) }

    #expect(throws: SafeTensors.Error.self) {
        let weights = try device.makeSafeTensors(from: url, mmap: true)
        _ = try weights.tensor(named: "x", options: [.storageModePrivate])
    }
}

@Test func missingTensorThrows() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let url = try makeSafeTensorsFile(tensors: [
        TensorFixture(name: "x", dtype: "U8", shape: [1], bytes: [42])
    ])
    defer { try? FileManager.default.removeItem(at: url) }
    let weights = try device.makeSafeTensors(from: url)

    #expect(throws: SafeTensors.Error.self) {
        _ = try weights.tensor(named: "missing")
    }
}

@available(iOS 26.0, macOS 26.0, *)
@Test func deviceCreatesMetalTensorFromSafeTensor() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let url = try makeSafeTensorsFile(tensors: [
        TensorFixture(name: "w", dtype: "F32", shape: [2, 2], bytes: bytes([Float32(1), Float32(2), Float32(3), Float32(4)]))
    ])
    defer { try? FileManager.default.removeItem(at: url) }
    let weights = try device.makeSafeTensors(from: url)

    let safeTensor = try weights.tensor(named: "w")
    let tensor = try safeTensor.mtlTensor()

    #expect(tensor.label == "w")
    #expect(tensor.dataType == .float32)
    #expect(tensor.dimensions.rank == 2)
    #expect(tensor.dimensions.extents == [2, 2])
    #expect(tensor.buffer?.label == "w")
}

@Test func malformedShapeByteCountThrows() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let url = try makeSafeTensorsFile(
        tensors: [TensorFixture(name: "bad", dtype: "F32", shape: [2], bytes: [0, 1, 2, 3])],
        validateFixtureByteCounts: false
    )
    defer { try? FileManager.default.removeItem(at: url) }

    #expect(throws: SafeTensors.Error.self) {
        _ = try device.makeSafeTensors(from: url)
    }
}

// MARK: - TensorFixture

private struct TensorFixture {
    var name: String
    var dtype: String
    var shape: [Int]
    var bytes: [UInt8]
}

private func makeSafeTensorsFile(
    tensors: [TensorFixture],
    metadata: [String: String] = [:],
    validateFixtureByteCounts: Bool = true
) throws -> URL {
    var offset = 0
    var header: [String: Any] = [:]
    if !metadata.isEmpty { header["__metadata__"] = metadata }

    for tensor in tensors {
        if validateFixtureByteCounts {
            #expect(tensor.shape.reduce(byteCount(for: tensor.dtype), *) == tensor.bytes.count)
        }
        header[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offset, offset + tensor.bytes.count],
        ]
        offset += tensor.bytes.count
    }

    let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var file = [UInt8]()
    var headerLength = UInt64(headerData.count).littleEndian
    withUnsafeBytes(of: &headerLength) { file.append(contentsOf: $0) }
    file.append(contentsOf: headerData)
    for tensor in tensors {
        file.append(contentsOf: tensor.bytes)
    }

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")
    try Data(file).write(to: url)
    return url
}

private func bytes(_ values: [some Any]) -> [UInt8] {
    values.withUnsafeBytes { Array($0) }
}

private func byteCount(for dtype: String) -> Int {
    switch dtype {
    case "BOOL", "U8", "I8": 1
    case "U16", "I16", "F16", "BF16": 2
    case "U32", "I32", "F32": 4
    case "U64", "I64", "F64": 8
    default: 0
    }
}
