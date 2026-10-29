import Foundation
import MLX

/// The single-image profile used by TurboFieldfare. Gemma's budget counts
/// pooled soft tokens; the encoder receives nine raw 16x16 patches per token.
public struct Gemma4ImageGeometry: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let patchCount: Int
    public let softTokenCount: Int

    public static let patchSize = 16
    public static let poolingKernelSize = 3
    public static let supportedSoftTokenBudgets = [70, 140, 280, 560]
    public static let defaultSoftTokenBudget = 560
    public static let maxSoftTokens = 560
    public static let maxPatches = maxSoftTokens * poolingKernelSize * poolingKernelSize
    public static let outputHiddenSize = 2_816

    public init(sourceWidth: Int, sourceHeight: Int,
                maxSoftTokens: Int = Self.defaultSoftTokenBudget) throws {
        guard sourceWidth > 0, sourceHeight > 0,
              Self.supportedSoftTokenBudgets.contains(maxSoftTokens) else {
            throw Gemma4ImageError.invalidDimensions
        }
        let maxPatchCount = maxSoftTokens * Self.poolingKernelSize * Self.poolingKernelSize
        let sideMultiple = Self.patchSize * Self.poolingKernelSize
        let targetPixels = Double(maxPatchCount * Self.patchSize * Self.patchSize)
        let scale = sqrt(targetPixels / (Double(sourceWidth) * Double(sourceHeight)))
        var targetHeight = Int(floor(scale * Double(sourceHeight) / Double(sideMultiple))) * sideMultiple
        var targetWidth = Int(floor(scale * Double(sourceWidth) / Double(sideMultiple))) * sideMultiple
        let maxSide = (maxPatchCount / (Self.poolingKernelSize * Self.poolingKernelSize)) * sideMultiple

        if targetHeight == 0 && targetWidth == 0 {
            throw Gemma4ImageError.invalidDimensions
        } else if targetHeight == 0 {
            targetHeight = sideMultiple
            targetWidth = min((sourceWidth / sourceHeight) * sideMultiple, maxSide)
        } else if targetWidth == 0 {
            targetWidth = sideMultiple
            targetHeight = min((sourceHeight / sourceWidth) * sideMultiple, maxSide)
        }

        let patchesHigh = targetHeight / Self.patchSize
        let patchesWide = targetWidth / Self.patchSize
        let patches = patchesHigh * patchesWide
        let softTokens = patches / (Self.poolingKernelSize * Self.poolingKernelSize)
        guard patches > 0, patches <= maxPatchCount,
              targetHeight % sideMultiple == 0, targetWidth % sideMultiple == 0,
              softTokens > 0, softTokens <= maxSoftTokens else {
            throw Gemma4ImageError.invalidDimensions
        }
        self.width = targetWidth
        self.height = targetHeight
        self.patchCount = patches
        self.softTokenCount = softTokens
    }
}

public enum Gemma4ImageError: Error, LocalizedError, CustomStringConvertible, Sendable, Equatable {
    case invalidDimensions
    case unsupportedFormat
    case malformedImage
    case imageTooLarge
    case unavailable(String)
    case memoryBudgetExceeded(current: UInt64, limit: UInt64)
    case incompatibleWeights(String)

    public var errorDescription: String? {
        switch self {
        case .invalidDimensions:
            "Image dimensions cannot fit the selected Gemma 4 image token budget."
        case .unsupportedFormat:
            "Unsupported image format. Choose a JPEG or PNG image."
        case .malformedImage:
            "The selected file is not a valid image."
        case .imageTooLarge:
            "The image file or decoded dimensions exceed the supported size."
        case .unavailable(let detail):
            "Image support unavailable: \(detail)"
        case .memoryBudgetExceeded(let current, let limit):
            "Image request stopped at \(current) bytes of process memory; the image safety limit is \(limit) bytes."
        case .incompatibleWeights(let detail):
            "Image support unavailable: vision weights are incompatible (\(detail))."
        }
    }

    public var description: String { errorDescription ?? "Image processing failed." }
}

/// Process memory is sampled independently from MLX's allocator. The process
/// cap adapts to installed RAM and is bounded at 6 GiB; this policy includes
/// the language model, vision tower, and TurboFieldfare's Metal runtime.
public enum Gemma4ImageMemoryPolicy {
    private static let maximumProcessBudgetBytes: UInt64 = 6 * 1_024 * 1_024 * 1_024
    private static let systemMemoryFloorBytes: UInt64 = 1 * 1_024 * 1_024 * 1_024
    public static let processBudgetBytes = min(
        maximumProcessBudgetBytes,
        max(systemMemoryFloorBytes, ProcessInfo.processInfo.physicalMemory / 2))
    public static let maximumVisionSidecarBytes = 1536 * 1_024 * 1_024
    public static let stopLimitBytes = processBudgetBytes > 512 * 1_024 * 1_024
        ? processBudgetBytes - 512 * 1_024 * 1_024
        : processBudgetBytes
    public static let admissionLimitBytes = processBudgetBytes > 2 * 1_024 * 1_024 * 1_024
        ? processBudgetBytes - 2 * 1_024 * 1_024 * 1_024
        : processBudgetBytes / 2
    public static let mlxMemoryLimitBytes = 3 * 1_024 * 1_024 * 1_024
    public static let mlxCacheLimitBytes = 0

    public struct MLXSettings: Sendable {
        fileprivate let memoryLimit: Int
        fileprivate let cacheLimit: Int
    }

    @discardableResult
    public static func configureMLX() -> MLXSettings {
        let old = MLXSettings(memoryLimit: Memory.memoryLimit, cacheLimit: Memory.cacheLimit)
        Memory.memoryLimit = mlxMemoryLimitBytes
        Memory.cacheLimit = mlxCacheLimitBytes
        return old
    }

    public static func restoreMLX(_ settings: MLXSettings) {
        Memory.clearCache()
        Memory.memoryLimit = settings.memoryLimit
        Memory.cacheLimit = settings.cacheLimit
    }

    public static func admit(currentFootprint: UInt64?) throws {
        guard let currentFootprint, currentFootprint <= admissionLimitBytes else {
            throw Gemma4ImageError.memoryBudgetExceeded(
                current: currentFootprint ?? UInt64.max,
                limit: admissionLimitBytes)
        }
    }

    public static func checkDuringImageRequest(currentFootprint: UInt64?) throws {
        guard let currentFootprint, currentFootprint <= stopLimitBytes else {
            throw Gemma4ImageError.memoryBudgetExceeded(
                current: currentFootprint ?? UInt64.max,
                limit: stopLimitBytes)
        }
    }
}
