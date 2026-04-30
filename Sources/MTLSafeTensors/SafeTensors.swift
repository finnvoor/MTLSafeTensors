import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

// MARK: - SafeTensor

/// A tensor stored in a safetensors file and exposed as a Metal buffer.
public struct SafeTensor {
    /// The mmap-backed or copied Metal buffer containing the tensor bytes.
    public let buffer: any MTLBuffer

    /// The tensor element type declared by the safetensors header.
    public let dtype: SafeTensors.DType

    /// The tensor shape in safetensors order.
    public let shape: [Int]

    /// The tensor name, taken from the Metal buffer label.
    public var name: String? { buffer.label }

    /// The number of elements implied by ``shape``.
    public var elementCount: Int { shape.reduce(1, *) }

    /// Creates an `MTLTensor` that shares storage with this tensor's Metal buffer.
    ///
    /// The dtype must have a matching Metal tensor data type. `U64`, `I64`, and `F64` do not.
    /// The returned tensor is not cached.
    ///
    /// - Parameter usage: Metal tensor usage. Defaults to compute.
    /// - Returns: A Metal tensor sharing storage with ``buffer``.
    @available(iOS 26.0, macOS 26.0, *)
    public func mtlTensor(usage: MTLTensorUsage = [.compute]) throws -> any MTLTensor {
        guard let dataType = dtype.metalTensorDataType else {
            throw SafeTensors.Error.unsupportedDType(dtype.rawValue)
        }

        let descriptor = MTLTensorDescriptor()
        descriptor.dataType = dataType
        let dimensions = shape.map { NSInteger($0) }
        descriptor.dimensions = MTLTensorExtents(__rank: shape.count, values: dimensions)!
        var stride = 1
        let strides = shape.map { dimension in
            defer { stride *= dimension }
            return NSInteger(stride)
        }
        descriptor.strides = MTLTensorExtents(__rank: strides.count, values: strides)!
        descriptor.usage = usage
        descriptor.storageMode = buffer.storageMode

        do {
            let tensor = try buffer.makeTensor(descriptor: descriptor, offset: 0)
            tensor.label = name
            return tensor
        } catch {
            throw SafeTensors.Error.metalTensorCreationFailed(name ?? "<unnamed>", error)
        }
    }
}

public extension MTLDevice {
    /// Opens a safetensors file for use with Metal.
    ///
    /// - Parameters:
    ///   - url: The `.safetensors` file URL.
    ///   - mmap: When `true`, maps the file and creates zero-copy shared buffers. When `false`, reads only the
    ///     header up front and lazily copies requested tensor bytes into Metal buffers.
    /// - Returns: A ``SafeTensors`` archive bound to this device.
    func makeSafeTensors(from url: URL, mmap: Bool = true) throws -> SafeTensors {
        try SafeTensors(url: url, device: self, mmap: mmap)
    }

}

// MARK: - SafeTensors

/// A safetensors archive bound to a Metal device.
///
/// Create instances with ``Metal/MTLDevice/makeSafeTensors(from:mmap:)``. The type is thread-safe for concurrent
/// metadata lookup and resource creation. Keep the archive alive while using resources returned from mmap-backed archives.
public final class SafeTensors {
    // MARK: Lifecycle

    fileprivate init(url: URL, device: any MTLDevice, mmap: Bool) throws {
        self.url = url
        self.device = device

        if mmap {
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else { throw Error.openFailed(url, errno) }
            defer { close(fd) }

            var st = stat()
            guard fstat(fd, &st) == 0 else { throw Error.statFailed(url, errno) }
            let size = Int(st.st_size)
            guard size >= 8 else { throw Error.malformedFile("file is smaller than safetensors header") }

            let ptr = Darwin.mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0)
            guard ptr != MAP_FAILED, let ptr else { throw Error.mmapFailed(url, errno) }

            do {
                let parsed = try Self.parseHeader(bytes: ptr, size: size)
                storage = .mapped(ptr, size)
                dataStart = parsed.dataStart
                records = parsed.records
                metadata = parsed.metadata
            } catch {
                munmap(ptr, size)
                throw error
            }
        } else {
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else { throw Error.openFailed(url, errno) }
            defer { close(fd) }

            var st = stat()
            guard fstat(fd, &st) == 0 else { throw Error.statFailed(url, errno) }
            let size = Int(st.st_size)
            guard size >= 8 else { throw Error.malformedFile("file is smaller than safetensors header") }

            var headerLengthBytes = [UInt8](repeating: 0, count: 8)
            guard read(fd, &headerLengthBytes, 8) == 8 else { throw Error.malformedFile("could not read safetensors header length") }
            let headerLength = Self.headerLength(from: headerLengthBytes)
            let headerEnd = 8 + Int(headerLength)
            guard headerLength <= UInt64(Int.max), headerEnd <= size else { throw Error.malformedFile("invalid header length") }

            var headerData = Data(count: Int(headerLength))
            let readCount = headerData.withUnsafeMutableBytes { rawBuffer in
                read(fd, rawBuffer.baseAddress, Int(headerLength))
            }
            guard readCount == Int(headerLength) else { throw Error.malformedFile("could not read safetensors header") }

            let parsed = try Self.parseHeader(headerData: headerData, dataStart: headerEnd, fileSize: size)
            storage = .file(url)
            dataStart = parsed.dataStart
            records = parsed.records
            metadata = parsed.metadata
        }
    }

    deinit {
        if case let .mapped(pointer, size) = storage {
            munmap(pointer, size)
        }
    }

    // MARK: Public

    /// Errors thrown while parsing safetensors files or creating Metal resources.
    public enum Error: Swift.Error, CustomStringConvertible {
        /// Opening the safetensors file failed. The associated value is the URL and `errno`.
        case openFailed(URL, Int32)
        /// Reading file metadata failed. The associated value is the URL and `errno`.
        case statFailed(URL, Int32)
        /// Mapping the file failed. The associated value is the URL and `errno`.
        case mmapFailed(URL, Int32)
        /// The file did not match the safetensors format or contained inconsistent tensor metadata.
        case malformedFile(String)
        /// No tensor exists for the requested name.
        case tensorNotFound(String)
        /// The safetensors dtype cannot be represented as an `MTLTensorDataType`.
        case unsupportedDType(String)
        /// Metal could not create a buffer for the requested tensor.
        case metalBufferCreationFailed(String)
        /// Mmap-backed resources must use shared storage; use a blit into a private buffer for GPU-only storage.
        case invalidResourceOptionsForMMap(MTLResourceOptions)
        /// The requested options cannot be initialized directly from CPU bytes; use a staging buffer and blit.
        case invalidResourceOptionsForCopy(MTLResourceOptions)
        /// Metal could not create an `MTLTensor` for the requested tensor.
        case metalTensorCreationFailed(String, Swift.Error?)

        // MARK: Public

        /// A human-readable description of the error.
        public var description: String {
            switch self {
            case let .openFailed(url, errno): "Could not open \(url.path): errno \(errno)"
            case let .statFailed(url, errno): "Could not stat \(url.path): errno \(errno)"
            case let .mmapFailed(url, errno): "Could not mmap \(url.path): errno \(errno)"
            case let .malformedFile(message): "Malformed safetensors file: \(message)"
            case let .tensorNotFound(name): "No tensor named \(name)"
            case let .unsupportedDType(dtype): "Unsupported dtype for MTLTensor: \(dtype)"
            case let .metalBufferCreationFailed(name): "Could not create Metal buffer for \(name)"
            case let .invalidResourceOptionsForMMap(options): "mmap-backed buffers must use shared storage, got options rawValue \(options.rawValue). To use private GPU-only storage, create a private buffer and blit/copy from the shared mmap-backed buffer."
            case let .invalidResourceOptionsForCopy(options): "copied buffers cannot be initialized directly with these resource options, got rawValue \(options.rawValue). To use private GPU-only storage, create a CPU-visible staging buffer first, then blit/copy into a private buffer."
            case let .metalTensorCreationFailed(name, error): "Could not create MTLTensor for \(name): \(error.map(String.init(describing:)) ?? "unknown error")"
            }
        }
    }

    /// Safetensors element data types supported by the parser.
    public enum DType: String, Sendable, Codable {
        /// Boolean value, stored as one byte.
        case bool = "BOOL"
        /// Unsigned 8-bit integer.
        case u8 = "U8"
        /// Signed 8-bit integer.
        case i8 = "I8"
        /// Unsigned 16-bit integer.
        case u16 = "U16"
        /// Signed 16-bit integer.
        case i16 = "I16"
        /// Unsigned 32-bit integer.
        case u32 = "U32"
        /// Signed 32-bit integer.
        case i32 = "I32"
        /// Unsigned 64-bit integer.
        case u64 = "U64"
        /// Signed 64-bit integer.
        case i64 = "I64"
        /// IEEE 754 half precision float.
        case f16 = "F16"
        /// Brain floating point 16-bit value.
        case bf16 = "BF16"
        /// IEEE 754 single precision float.
        case f32 = "F32"
        /// IEEE 754 double precision float.
        case f64 = "F64"

        // MARK: Public

        /// Size of one element in bytes.
        public var byteCount: Int {
            switch self {
            case .bool, .u8, .i8: 1
            case .u16, .i16, .f16, .bf16: 2
            case .u32, .i32, .f32: 4
            case .u64, .i64, .f64: 8
            }
        }

        /// The corresponding Metal tensor data type, when Metal supports one.
        ///
        /// `U64`, `I64`, and `F64` return `nil` because Metal tensors do not have matching data types.
        @available(iOS 26.0, macOS 26.0, *) public var metalTensorDataType: MTLTensorDataType? {
            switch self {
            case .u8, .bool: .uint8
            case .i8: .int8
            case .u16: .uint16
            case .i16: .int16
            case .u32: .uint32
            case .i32: .int32
            case .f16: .float16
            case .bf16: .bfloat16
            case .f32: .float32
            case .u64, .i64, .f64: nil
            }
        }
    }

    /// The source safetensors file.
    public let url: URL

    /// The Metal device used to create buffers.
    public let device: any MTLDevice

    /// Optional safetensors `__metadata__` values.
    public let metadata: [String: String]

    /// All tensor names sorted lexicographically.
    public var names: [String] {
        records.keys.sorted()
    }

    /// Returns a named tensor and its Metal buffer.
    ///
    /// For mmap-backed archives, the returned buffer is a zero-copy `.storageModeShared` wrapper over the mapped file,
    /// and `options` must specify shared storage. For non-mmap archives, only the requested tensor bytes are read from
    /// disk and copied into the returned buffer. Options that require GPU-only storage, such as `.storageModePrivate`,
    /// are rejected because they require an explicit staging-buffer-to-private-buffer blit.
    ///
    /// Returned buffers are cached by tensor name and resource options. The buffer label is set to `name`.
    ///
    /// - Parameters:
    ///   - name: Tensor name from the safetensors header.
    ///   - options: Metal resource options for the buffer. Defaults to shared storage.
    /// - Returns: A tensor containing dtype, shape, and Metal buffer.
    public func tensor(
        named name: String,
        options: MTLResourceOptions = [.storageModeShared]
    ) throws -> SafeTensor {
        guard let record = records[name] else { throw Error.tensorNotFound(name) }
        let buffer = try metalBuffer(name: name, record: record, options: options)
        return SafeTensor(buffer: buffer, dtype: record.dtype, shape: record.shape)
    }

    // MARK: Private

    private struct TensorRecord {
        let dtype: DType
        let shape: [Int]
        let dataByteRange: Range<Int>

        var byteCount: Int { dataByteRange.count }
    }

    private enum Storage {
        case mapped(UnsafeMutableRawPointer, Int)
        case file(URL)
    }

    private struct BufferCacheKey: Hashable {
        // MARK: Lifecycle

        init(name: String, options: MTLResourceOptions) {
            self.name = name
            optionsRawValue = options.rawValue
        }

        // MARK: Internal

        let name: String
        let optionsRawValue: UInt
    }

    private let records: [String: TensorRecord]
    private let storage: Storage
    private let dataStart: Int
    private let cacheLock = NSLock()
    private var bufferCache: [BufferCacheKey: any MTLBuffer] = [:]


    private static func headerLength(from bytes: some Collection<UInt8>) -> UInt64 {
        var headerLength: UInt64 = 0
        for (i, byte) in bytes.prefix(8).enumerated() {
            headerLength |= UInt64(byte) << UInt64(8 * i)
        }
        return headerLength
    }

    private static func isSharedStorage(_ options: MTLResourceOptions) -> Bool {
        let storageMask = MTLResourceOptions.storageModeShared.rawValue
            | MTLResourceOptions.storageModePrivate.rawValue
            | MTLResourceOptions.storageModeMemoryless.rawValue
        return options.rawValue & storageMask == MTLResourceOptions.storageModeShared.rawValue
    }

    private static func parseHeader(bytes pointer: UnsafeMutableRawPointer, size: Int) throws -> (dataStart: Int, records: [String: TensorRecord], metadata: [String: String]) {
        let bytes = pointer.assumingMemoryBound(to: UInt8.self)
        let headerLength = headerLength(from: UnsafeBufferPointer(start: bytes, count: 8))
        let headerEnd = 8 + Int(headerLength)
        guard headerLength <= UInt64(Int.max), headerEnd <= size else { throw Error.malformedFile("invalid header length") }
        let headerData = Data(bytes: pointer.advanced(by: 8), count: Int(headerLength))
        return try parseHeader(headerData: headerData, dataStart: headerEnd, fileSize: size)
    }

    private static func parseHeader(headerData data: Data, dataStart headerEnd: Int, fileSize size: Int) throws -> (dataStart: Int, records: [String: TensorRecord], metadata: [String: String]) {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw Error.malformedFile("header is not a JSON object") }

        var metadata: [String: String] = [:]
        var records: [String: TensorRecord] = [:]
        for (name, value) in object where name != "__metadata__" {
            guard let dict = value as? [String: Any],
                  let dtypeString = dict["dtype"] as? String,
                  let dtype = DType(rawValue: dtypeString),
                  let shape = dict["shape"] as? [Int],
                  let offsets = dict["data_offsets"] as? [Int], offsets.count == 2 else {
                throw Error.malformedFile("invalid entry for \(name)")
            }
            guard offsets[0] <= offsets[1], headerEnd + offsets[1] <= size else { throw Error.malformedFile("invalid data_offsets for \(name)") }
            let expected = shape.reduce(dtype.byteCount) { $0 * $1 }
            guard expected == offsets[1] - offsets[0] else { throw Error.malformedFile("shape/dtype byte count mismatch for \(name)") }
            records[name] = TensorRecord(dtype: dtype, shape: shape, dataByteRange: offsets[0]..<offsets[1])
        }
        if let raw = object["__metadata__"] as? [String: String] { metadata = raw }
        return (headerEnd, records, metadata)
    }

    private func metalBuffer(name: String, record: TensorRecord, options: MTLResourceOptions) throws -> any MTLBuffer {
        switch storage {
        case .mapped where !Self.isSharedStorage(options):
            throw Error.invalidResourceOptionsForMMap(options)
        case .file where options.contains(.storageModePrivate) || options.contains(.storageModeMemoryless):
            throw Error.invalidResourceOptionsForCopy(options)
        default:
            break
        }
        let cacheKey = BufferCacheKey(name: name, options: options)

        cacheLock.lock()
        defer { cacheLock.unlock() }

        if let cached = bufferCache[cacheKey] { return cached }
        let buffer: (any MTLBuffer)?
        switch storage {
        case let .mapped(pointer, _):
            let ptr = pointer.advanced(by: dataStart + record.dataByteRange.lowerBound)
            buffer = device.makeBuffer(bytesNoCopy: ptr, length: record.byteCount, options: options, deallocator: nil)
        case let .file(url):
            let data = try tensorData(name: name, record: record, url: url)
            buffer = data.withUnsafeBytes { rawBuffer in
                guard let baseAddress = rawBuffer.baseAddress else { return nil }
                return device.makeBuffer(bytes: baseAddress, length: record.byteCount, options: options)
            }
        }
        guard let buffer else { throw Error.metalBufferCreationFailed(name) }
        buffer.label = name
        bufferCache[cacheKey] = buffer
        return buffer
    }

    private func tensorData(name: String, record: TensorRecord, url: URL) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(dataStart + record.dataByteRange.lowerBound))
        let data = try handle.read(upToCount: record.byteCount) ?? Data()
        guard data.count == record.byteCount else { throw Error.metalBufferCreationFailed(name) }
        return data
    }
}
