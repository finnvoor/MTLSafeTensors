# MTLSafeTensors

Small Swift package for loading `.safetensors` weights into Metal buffers.

```swift
import Metal
import MTLSafeTensors

let device = MTLCreateSystemDefaultDevice()!
let weights = try device.makeSafeTensors(from: url) // mmap-backed by default

let weight = try weights.tensor(named: "model.layers.0.weight")
print(weight.name, weight.dtype, weight.shape)
computeEncoder.setBuffer(weight.buffer, offset: 0, index: 0)

if #available(iOS 26.0, macOS 26.0, *) {
    let metalTensor = try weight.mtlTensor()
}
```

Keep the `SafeTensors` object alive while using tensors returned from mmap-backed archives.

## Loading modes

```swift
let mapped = try device.makeSafeTensors(from: url)
let copied = try device.makeSafeTensors(from: url, mmap: false)
```

- `mmap: true` maps the file and creates zero-copy shared `MTLBuffer`s with `makeBuffer(bytesNoCopy:)`.
- `mmap: false` reads only the header at init. Requested tensor bytes are read lazily and copied into `MTLBuffer`s.
- Private GPU-only buffers cannot be initialized directly from file/CPU bytes. To use private storage, create a private buffer and blit/copy from a shared/staging buffer.

## API

- `device.makeSafeTensors(from: url, mmap: true)` opens a safetensors file.
- `weights.names` and `weights.metadata` inspect contents.
- `weights.tensor(named: name, options: [.storageModeShared])` returns a `SafeTensor`.
- `SafeTensor` contains `buffer`, `dtype`, `shape`, computed `name`, and computed `elementCount`.
- `safeTensor.mtlTensor()` creates an `MTLTensor` from a `SafeTensor` on iOS/macOS 26+.

All safetensors dtypes parsed by this package can be exposed as `MTLBuffer` contents: `BOOL`, `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `U64`, `I64`, `F16`, `BF16`, `F32`, and `F64`.
