import Metal

/// Produces next-token logits for the `Generator`. The production
/// implementation is `RealForwardRunner`; tests use scripted logits so decode
/// behavior stays independent of the kernel stack.
public protocol LogitProducer: AnyObject, Sendable {
    /// Clear any per-generation state, such as KV cache.
    func reset()
    /// Run one token at `position`, leaving FP16 logits in `logits`.
    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws
}

public protocol ContinuableLogitProducer: LogitProducer {
    var continuationPosition: Int { get }
    func prepareForContinuation(expectedPosition: Int) throws
}

protocol ContextWindowReporting: Sendable {
    var maxContext: Int { get }
}

public enum PrefillOutputMode: Sendable, Equatable {
    case logits
    case greedyIfAvailable
}

public enum PrefillSeed: Sendable, Equatable {
    case logitsWritten
    case greedyToken(UInt32)
}

public struct PrefillResult: Sendable, Equatable {
    public let newPosition: Int
    public let seed: PrefillSeed

    public init(newPosition: Int, seed: PrefillSeed) {
        self.newPosition = newPosition
        self.seed = seed
    }
}

protocol ChunkedPrefillRunner: LogitProducer {
    /// Prefill a prompt slice using the chunked production runtime.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult
}

/// Per-request image rows projected by the optional Gemma 4 vision tower.
/// `tokenOffset` points at the first image soft-token row in the expanded
/// prompt; BOI/EOI markers remain normal token embeddings around this range.
public struct VisionPromptEmbeddings: Sendable {
    public let tokenOffset: Int
    public let rowCount: Int
    public let hiddenSize: Int
    public let values: [Float16]
    public let checkMemory: @Sendable () throws -> Void

    public init(tokenOffset: Int, rowCount: Int, hiddenSize: Int, values: [Float16],
                checkMemory: @escaping @Sendable () throws -> Void = {}) throws {
        guard tokenOffset >= 0, rowCount > 0,
              rowCount <= PrefillRuntimeConfig.maxImageChunkTokens,
              hiddenSize == 2_816,
              values.count == rowCount * hiddenSize else {
            throw GeneratorError.invalidVisionEmbeddings
        }
        self.tokenOffset = tokenOffset
        self.rowCount = rowCount
        self.hiddenSize = hiddenSize
        self.values = values
        self.checkMemory = checkMemory
    }
}

protocol VisionChunkedPrefillRunner: ChunkedPrefillRunner {
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        visionEmbeddings: VisionPromptEmbeddings,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult
}
