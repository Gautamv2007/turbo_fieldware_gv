// Adapted from Apple's MLX Swift Gemma 4 vision encoder (Apache-2.0).
// Kept local because MLXLMCommon exposes the vision classes with package access.
import Foundation
import ImageIO
import CoreGraphics
import MLX
import MLXNN

public struct Gemma4ProjectedImage: Sendable {
    public let values: [Float16]
    public let rowCount: Int
    public let hiddenSize: Int
    public let geometry: Gemma4ImageGeometry
}

public enum Gemma4ImagePreprocessor {
    public static let maximumFileBytes: UInt64 = 32 * 1_024 * 1_024
    public static let maximumSourceDimension = 100_000
    public static let maximumSourcePixels = 100_000_000

    public static func pixels(from url: URL,
                              imageTokenBudget: Int = Gemma4ImageGeometry.defaultSoftTokenBudget)
        throws -> (values: [Float16], geometry: Gemma4ImageGeometry) {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        guard let fileSize = values.fileSize, fileSize > 0,
              UInt64(fileSize) <= maximumFileBytes else {
            throw Gemma4ImageError.imageTooLarge
        }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              CGImageSourceGetCount(source) > 0,
              let type = CGImageSourceGetType(source) as String? else {
            throw Gemma4ImageError.malformedImage
        }
        guard type == "public.jpeg" || type == "public.png" else {
            throw Gemma4ImageError.unsupportedFormat
        }
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let rawWidth = properties[kCGImagePropertyPixelWidth] as? Int,
              let rawHeight = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw Gemma4ImageError.malformedImage
        }
        guard rawWidth > 0, rawHeight > 0,
              rawWidth <= maximumSourceDimension, rawHeight <= maximumSourceDimension,
              rawWidth.multipliedReportingOverflow(by: rawHeight).overflow == false,
              rawWidth * rawHeight <= maximumSourcePixels else {
            throw Gemma4ImageError.imageTooLarge
        }
        let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
        let swapsDimensions = (5...8).contains(orientation)
        let sourceWidth = swapsDimensions ? rawHeight : rawWidth
        let sourceHeight = swapsDimensions ? rawWidth : rawHeight
        let geometry = try Gemma4ImageGeometry(sourceWidth: sourceWidth,
                                               sourceHeight: sourceHeight,
                                               maxSoftTokens: imageTokenBudget)
        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(geometry.width, geometry.height),
            kCGImageSourceShouldCacheImmediately: false,
        ]
        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbnailOptions as CFDictionary),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw Gemma4ImageError.malformedImage
        }
        let rowBytes = geometry.width * 4
        guard let context = CGContext(data: nil,
                                      width: geometry.width,
                                      height: geometry.height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: rowBytes,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw Gemma4ImageError.malformedImage
        }
        context.interpolationQuality = .high
        context.draw(thumbnail, in: CGRect(x: 0, y: 0,
                                           width: geometry.width, height: geometry.height))
        guard let rgba = context.data?.assumingMemoryBound(to: UInt8.self) else {
            throw Gemma4ImageError.malformedImage
        }
        let pixelCount = geometry.width * geometry.height
        var planar = [Float16](repeating: 0, count: pixelCount * 3)
        let unpremultiply: (UInt8, Float) -> Float16 = { channel, alpha in
            guard alpha > 0 else { return 0 }
            // Both channel and alpha are bytes. For premultiplied RGBA,
            // channel / alpha already restores the normalized [0, 1] color.
            return Float16(min(Float(channel) / alpha, 1))
        }
        for y in 0..<geometry.height {
            for x in 0..<geometry.width {
                let source = y * rowBytes + x * 4
                let destination = y * geometry.width + x
                let alpha = Float(rgba[source + 3])
                planar[destination] = unpremultiply(rgba[source], alpha)
                planar[pixelCount + destination] = unpremultiply(rgba[source + 1], alpha)
                planar[2 * pixelCount + destination] = unpremultiply(rgba[source + 2], alpha)
            }
        }
        return (planar, geometry)
    }
}

private struct Gemma4VisionConfig {
    let hiddenLayers = 27
    let hiddenSize = 1_152
    let intermediateSize = 4_304
    let attentionHeads = 16
    let keyValueHeads = 16
    let headDim = 72
    let patchSize = 16
    let rmsNormEps: Float = 1e-6
    let positionEmbeddingSize = 10_240
    let poolingKernelSize = 3
    let ropeTheta: Float = 100
}

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape[x.ndim - 1] / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

private func gemma4OneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    expandedDimensions(indices, axis: -1) .== MLXArray(0..<numClasses)
}

private func applyAxialRoPE(_ x: MLXArray, positions: MLXArray, base: Float) -> MLXArray {
    let headDim = x.shape[x.ndim - 1]
    let dims = positions.shape[positions.ndim - 1]
    let channels = 2 * (headDim / (2 * dims))
    let half = channels / 2
    var parts: [MLXArray] = []
    for d in 0..<dims {
        let start = d * channels
        let part = x[.ellipsis, start..<(start + channels)]
        let exponents = (2.0 / Float(channels)) * MLXArray(0..<half).asType(.float32)
        let timescale = MLX.pow(MLXArray(base), exponents)
        let position = positions[.ellipsis, d..<(d + 1)].asType(.float32)
        let angle = position / timescale
        let cosine = concatenated([cos(angle), cos(angle)], axis: -1).asType(x.dtype).expandedDimensions(axis: 2)
        let sine = concatenated([sin(angle), sin(angle)], axis: -1).asType(x.dtype).expandedDimensions(axis: 2)
        parts.append(part * cosine + rotateHalf(part) * sine)
    }
    return concatenated(parts, axis: -1)
}

private func fusedAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
    let dim = q.shape[q.ndim - 1]
    let padded = [64, 80, 128].first(where: { dim <= $0 }) ?? dim
    guard padded != dim else {
        return MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                 scale: 1, mask: .none)
    }
    let widths: [IntOrPair] = [0, 0, 0, .init((0, padded - dim))]
    return MLXFast.scaledDotProductAttention(
        queries: MLX.padded(q, widths: widths),
        keys: MLX.padded(k, widths: widths),
        values: MLX.padded(v, widths: widths), scale: 1, mask: .none)[.ellipsis, ..<dim]
}

private func checkProcessFootprint(_ processFootprint: @Sendable () -> UInt64?,
                                  maximum: UInt64) throws {
    guard let current = processFootprint(), current <= maximum else {
        throw Gemma4ImageError.memoryBudgetExceeded(
            current: processFootprint() ?? UInt64.max,
            limit: maximum)
    }
}

private final class VisionRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray
    init(_ dimensions: Int, eps: Float, zeroShift: Bool = true) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions], dtype: .bfloat16)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        return (xf * rsqrt(mean(xf.square(), axis: -1, keepDims: true) + eps)
                * weight.asType(.float32)).asType(x.dtype)
    }
}

private final class VisionRMSNormNoScale: Module, UnaryLayer {
    let eps: Float
    init(eps: Float) { self.eps = eps; super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        return (xf * rsqrt(mean(xf.square(), axis: -1, keepDims: true) + eps)).asType(x.dtype)
    }
}

/// Preserves the nested `linear.weight` tensor keys from MLX's Gemma 4 tower.
private final class VisionClippableLinear: Module, UnaryLayer {
    @ModuleInfo(key: "linear") var linear: Linear
    init(_ input: Int, _ output: Int) {
        _linear.wrappedValue = Linear(
            weight: MLXArray.zeros([output, input], dtype: .bfloat16))
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(x) }
}

private final class VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var q: VisionClippableLinear
    @ModuleInfo(key: "k_proj") var k: VisionClippableLinear
    @ModuleInfo(key: "v_proj") var v: VisionClippableLinear
    @ModuleInfo(key: "o_proj") var o: VisionClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: VisionRMSNorm
    @ModuleInfo(key: "_v_norm") var vNorm: VisionRMSNormNoScale
    let config: Gemma4VisionConfig
    init(_ config: Gemma4VisionConfig) {
        self.config = config
        _q.wrappedValue = VisionClippableLinear(config.hiddenSize, config.attentionHeads * config.headDim)
        _k.wrappedValue = VisionClippableLinear(config.hiddenSize, config.keyValueHeads * config.headDim)
        _v.wrappedValue = VisionClippableLinear(config.hiddenSize, config.keyValueHeads * config.headDim)
        _o.wrappedValue = VisionClippableLinear(config.attentionHeads * config.headDim, config.hiddenSize)
        _qNorm.wrappedValue = VisionRMSNorm(config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = VisionRMSNorm(config.headDim, eps: config.rmsNormEps)
        _vNorm.wrappedValue = VisionRMSNormNoScale(eps: config.rmsNormEps)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var qv = qNorm(q(x).reshaped(batch, length, config.attentionHeads, config.headDim))
        var kv = kNorm(k(x).reshaped(batch, length, config.keyValueHeads, config.headDim))
        let vv = vNorm(v(x).reshaped(batch, length, config.keyValueHeads, config.headDim))
        qv = applyAxialRoPE(qv, positions: positions, base: config.ropeTheta).transposed(0, 2, 1, 3)
        kv = applyAxialRoPE(kv, positions: positions, base: config.ropeTheta).transposed(0, 2, 1, 3)
        return o(fusedAttention(qv, kv, vv.transposed(0, 2, 1, 3))
            .transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private final class VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: VisionClippableLinear
    @ModuleInfo(key: "up_proj") var up: VisionClippableLinear
    @ModuleInfo(key: "down_proj") var down: VisionClippableLinear
    init(_ c: Gemma4VisionConfig) {
        _gate.wrappedValue = VisionClippableLinear(c.hiddenSize, c.intermediateSize)
        _up.wrappedValue = VisionClippableLinear(c.hiddenSize, c.intermediateSize)
        _down.wrappedValue = VisionClippableLinear(c.intermediateSize, c.hiddenSize)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geluApproximate(gate(x)) * up(x)) }
}

private final class VisionBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: VisionAttention
    @ModuleInfo var mlp: VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: VisionRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: VisionRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFFNorm: VisionRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFFNorm: VisionRMSNorm
    init(_ c: Gemma4VisionConfig) {
        _attention.wrappedValue = VisionAttention(c); _mlp.wrappedValue = VisionMLP(c)
        _inputNorm.wrappedValue = VisionRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionNorm.wrappedValue = VisionRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        _preFFNorm.wrappedValue = VisionRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        _postFFNorm.wrappedValue = VisionRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let h = x + postAttentionNorm(attention(inputNorm(x), positions: positions))
        return h + postFFNorm(mlp(preFFNorm(h)))
    }
}

private final class VisionPatchEmbedder: Module {
    @ModuleInfo(key: "input_proj") var inputProjection: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray
    let c: Gemma4VisionConfig
    init(_ c: Gemma4VisionConfig) {
        self.c = c
        _inputProjection.wrappedValue = Linear(
            weight: MLXArray.zeros([c.hiddenSize, 3 * c.patchSize * c.patchSize], dtype: .bfloat16))
        _positionEmbeddingTable.wrappedValue = MLXArray.zeros(
            [2, c.positionEmbeddingSize, c.hiddenSize], dtype: .bfloat16)
        super.init()
    }
    func callAsFunction(_ pixels: MLXArray, positions: MLXArray) -> MLXArray {
        let (batch, _, height, width) = (pixels.dim(0), pixels.dim(1), pixels.dim(2), pixels.dim(3))
        let ph = height / c.patchSize, pw = width / c.patchSize
        var patches = pixels.reshaped(batch, 3, ph, c.patchSize, pw, c.patchSize)
            .transposed(0, 2, 4, 3, 5, 1)
            .reshaped(batch, ph * pw, 3 * c.patchSize * c.patchSize)
        patches = 2 * (patches - 0.5)
        let xIndex = positions[0..., 0..., 0].flattened().asType(.int32)
        let yIndex = positions[0..., 0..., 1].flattened().asType(.int32)
        let xPos = take(positionEmbeddingTable[0], xIndex, axis: 0).reshaped(batch, ph * pw, c.hiddenSize)
        let yPos = take(positionEmbeddingTable[1], yIndex, axis: 0).reshaped(batch, ph * pw, c.hiddenSize)
        return inputProjection(patches.asType(inputProjection.weight.dtype)) + xPos + yPos
    }
}

private final class VisionPooler: Module {
    let c: Gemma4VisionConfig
    init(_ c: Gemma4VisionConfig) { self.c = c; super.init() }
    func callAsFunction(_ x: MLXArray, positions: MLXArray, patchesW: Int, outputLength: Int) -> MLXArray {
        let scale = MLXArray(Float(c.hiddenSize).squareRoot(), dtype: x.dtype)
        let count = x.dim(1)
        guard count > outputLength else { return x * scale }
        let kernel = max(Int(sqrt(Double(count / outputLength))), 1)
        let indices = floor(positions[0].asType(.float32) / Float(kernel)).asType(.int32)
        let flattened = indices[0..., 0] + MLXArray(Int32(max(patchesW / kernel, 1))) * indices[0..., 1]
        let weights = gemma4OneHot(flattened, numClasses: outputLength).asType(.float32) / Float(kernel * kernel)
        return einsum("lL,bld->bLd", weights, x).asType(x.dtype) * scale
    }
}

private final class VisionTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [VisionBlock]
    init(_ c: Gemma4VisionConfig) {
        _layers.wrappedValue = (0..<c.hiddenLayers).map { _ in VisionBlock(c) }
        super.init()
    }
    func callAsFunction(_ input: MLXArray,
                        positions: MLXArray,
                        checkMemory: () throws -> Void) throws -> MLXArray {
        var hidden = input
        for layer in layers {
            try checkMemory()
            hidden = layer(hidden, positions: positions)
            // MLX evaluates lazily. Materialize each layer so its activation
            // peak is included in the process-footprint check before continuing.
            hidden.eval()
            try checkMemory()
        }
        return hidden
    }
}

private final class Gemma4VisionOnlyModel: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: VisionPatchEmbedder
    @ModuleInfo(key: "encoder") var encoder: VisionTransformer
    @ModuleInfo(key: "pooler") var pooler: VisionPooler
    @ModuleInfo(key: "std_bias") var stdBias: MLXArray?
    @ModuleInfo(key: "std_scale") var stdScale: MLXArray?
    let c = Gemma4VisionConfig()
    override init() {
        _patchEmbedder.wrappedValue = VisionPatchEmbedder(c)
        _encoder.wrappedValue = VisionTransformer(c)
        _pooler.wrappedValue = VisionPooler(c)
        _stdBias.wrappedValue = MLXArray.zeros([c.hiddenSize])
        _stdScale.wrappedValue = MLXArray.ones([c.hiddenSize])
        super.init()
    }
    func callAsFunction(_ pixels: MLXArray, checkMemory: () throws -> Void) throws -> MLXArray {
        let height = pixels.dim(2), width = pixels.dim(3)
        let ph = height / c.patchSize, pw = width / c.patchSize
        let grids = meshGrid([MLXArray.arange(pw, dtype: .int32), MLXArray.arange(ph, dtype: .int32)])
        let positions = stacked([grids[0].flattened(), grids[1].flattened()], axis: 1)
            .reshaped(1, ph * pw, 2)
        var h = patchEmbedder(pixels, positions: positions)
        h = try encoder(h, positions: positions, checkMemory: checkMemory)
        h = pooler(h, positions: positions, patchesW: pw,
                   outputLength: ph * pw / (c.poolingKernelSize * c.poolingKernelSize))
        return (h - stdBias!) * stdScale!
    }
}

private final class Gemma4VisionProjection: Module {
    @ModuleInfo(key: "embedding_projection") var projection: QuantizedLinear
    @ModuleInfo(key: "embedding_pre_projection_norm") var norm: VisionRMSNormNoScale
    override init() {
        _projection.wrappedValue = QuantizedLinear(
            weight: MLXArray.zeros([2_816, 144], dtype: .uint32),
            bias: nil,
            scales: MLXArray.zeros([2_816, 18], dtype: .bfloat16),
            biases: MLXArray.zeros([2_816, 18], dtype: .bfloat16),
            groupSize: 64, bits: 4, mode: .affine)
        _norm.wrappedValue = VisionRMSNormNoScale(eps: 1e-6)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(norm(x)) }
}

/// Loads only `vision_tower.*` and `embed_vision.*` from the optional sidecar.
/// No language model or text weights are constructed by this target.
public enum Gemma4VisionEncoder {
    public static func project(imageURL: URL,
                               weightsURL: URL,
                               imageTokenBudget: Int = Gemma4ImageGeometry.defaultSoftTokenBudget,
                               maximumProcessFootprint: UInt64,
                               processFootprint: @Sendable () -> UInt64?) throws -> Gemma4ProjectedImage {
        guard maximumProcessFootprint > 0 else {
            throw Gemma4ImageError.memoryBudgetExceeded(current: 0, limit: 0)
        }
        let image = try Gemma4ImagePreprocessor.pixels(from: imageURL,
                                                       imageTokenBudget: imageTokenBudget)
        try checkProcessFootprint(processFootprint, maximum: maximumProcessFootprint)
        let weightFile = try weightsURL.resourceValues(forKeys: [.fileSizeKey])
        guard let weightFileSize = weightFile.fileSize, weightFileSize > 0,
              weightFileSize <= Gemma4ImageMemoryPolicy.maximumVisionSidecarBytes else {
            throw Gemma4ImageError.incompatibleWeights("vision sidecar exceeds the supported 1.5 GiB size")
        }
        let oldMLXSettings = Gemma4ImageMemoryPolicy.configureMLX()
        defer {
            Gemma4ImageMemoryPolicy.restoreMLX(oldMLXSettings)
        }
        let vision = Gemma4VisionOnlyModel()
        let projection = Gemma4VisionProjection()
        // MLX's Metal backend does not implement the safetensors Load primitive on GPU.
        // Read tensors on CPU; the vision ops will transfer them to Metal when evaluated.
        let parameters = try MLX.loadArrays(url: weightsURL, stream: .cpu)
        let filtered = parameters.filter { $0.key.hasPrefix("vision_tower.") || $0.key.hasPrefix("embed_vision.") }
        guard filtered.count == parameters.count,
              filtered.keys.contains(where: { $0.hasPrefix("vision_tower.") }),
              filtered.keys.contains(where: { $0.hasPrefix("embed_vision.") }) else {
            throw Gemma4ImageError.incompatibleWeights("sidecar contains unexpected or missing tensors")
        }
        guard filtered.filter({ $0.key.hasPrefix("vision_tower.") })
                .allSatisfy({ $0.value.dtype == .bfloat16 }),
              filtered["embed_vision.embedding_projection.weight"]?.dtype == .uint32,
              filtered["embed_vision.embedding_projection.scales"]?.dtype == .bfloat16,
              filtered["embed_vision.embedding_projection.biases"]?.dtype == .bfloat16 else {
            throw Gemma4ImageError.incompatibleWeights("tensor dtypes do not match the pinned Gemma 4 vision checkpoint")
        }
        try vision.update(parameters: ModuleParameters.unflattened(filtered.filter { $0.key.hasPrefix("vision_tower.") }.map { (String($0.key.dropFirst("vision_tower.".count)), $0.value) }), verify: .all)
        try projection.update(parameters: ModuleParameters.unflattened(filtered.filter { $0.key.hasPrefix("embed_vision.") }.map { (String($0.key.dropFirst("embed_vision.".count)), $0.value) }), verify: .all)
        try checkProcessFootprint(processFootprint, maximum: maximumProcessFootprint)
        let input = MLXArray(image.values, [1, 3, image.geometry.height, image.geometry.width])
        let features = try withError {
            try vision(input) {
                try checkProcessFootprint(processFootprint, maximum: maximumProcessFootprint)
            }
        }
        let projected = try withError {
            let output = projection(features).asType(.float16)
            output.eval()
            return output
        }
        try checkProcessFootprint(processFootprint, maximum: maximumProcessFootprint)
        let rows = projected.dim(1)
        guard rows == image.geometry.softTokenCount, rows <= Gemma4ImageGeometry.maxSoftTokens else {
            throw Gemma4ImageError.incompatibleWeights("projected token count does not match Gemma preprocessing")
        }
        let values = projected.asArray(Float16.self)
        try checkProcessFootprint(processFootprint, maximum: maximumProcessFootprint)
        let output = Gemma4ProjectedImage(values: values,
                                         rowCount: rows,
                                         hiddenSize: Gemma4ImageGeometry.outputHiddenSize,
                                         geometry: image.geometry)
        return output
    }
}
