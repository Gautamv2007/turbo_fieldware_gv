import Foundation

public enum StopReason: Sendable, Equatable {
    case eos
    case endOfTurn
    case maxTokens
    case stopString
    case toolCalls
}

enum GeneratorError: Error, CustomStringConvertible, Equatable {
    case contextOverflow(prompt: Int, maxNew: Int, maxContext: Int)
    case invalidGenerationConfig(String)
    case invalidContinuation(String)
    case emptyPrompt
    case invalidVisionEmbeddings
    case visionContextLimit(prompt: Int, limit: Int)

    public var description: String {
        switch self {
        case .contextOverflow(let prompt, let maxNew, let maxContext):
            return "context overflow: prompt \(prompt) + maxNew \(maxNew) exceeds maxContext \(maxContext)"
        case .invalidGenerationConfig(let reason):
            return reason
        case .invalidContinuation(let reason):
            return reason
        case .emptyPrompt:
            return "empty prompt"
        case .invalidVisionEmbeddings:
            return "invalid image embedding rows"
        case .visionContextLimit(let prompt, let limit):
            return "image prompt has \(prompt) tokens, exceeding the \(limit)-token image context limit"
        }
    }
}
