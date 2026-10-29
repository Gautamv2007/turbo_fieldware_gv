import Foundation
import TurboFieldfareVision

public struct AppGenerationRequest: Equatable, Sendable {
    public var modelDirectory: URL
    public var prompt: String
    public var imageURL: URL?
    public var imageTokenBudget: Int
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var runtimeOptions: AppRuntimeOptions

    public init(modelDirectory: URL,
                prompt: String,
                imageURL: URL? = nil,
                imageTokenBudget: Int = Gemma4ImageGeometry.defaultSoftTokenBudget,
                maxNewTokens: Int = 4_096,
                maxContextTokens: Int = 4096,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                runtimeOptions: AppRuntimeOptions = AppRuntimeOptions()) {
        self.modelDirectory = modelDirectory
        self.prompt = prompt
        self.imageURL = imageURL
        self.imageTokenBudget = imageTokenBudget
        self.maxNewTokens = maxNewTokens
        self.maxContextTokens = maxContextTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.runtimeOptions = runtimeOptions
    }

    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

    public func validate(fileManager: FileManager = .default,
                         requireModelDirectory: Bool = true) throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AppInferenceError.invalidRequest("Prompt cannot be empty.")
        }
        guard maxNewTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max response length must be greater than zero.")
        }
        guard maxContextTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max context must be greater than zero.")
        }
        if imageURL != nil, maxContextTokens > 4_096 {
            throw AppInferenceError.invalidRequest("Image requests are limited to a 4K context.")
        }
        if imageURL != nil,
           !Gemma4ImageGeometry.supportedSoftTokenBudgets.contains(imageTokenBudget) {
            throw AppInferenceError.invalidRequest("Choose a supported image detail level.")
        }
        guard temperature >= 0 else {
            throw AppInferenceError.invalidRequest("Temperature cannot be negative.")
        }
        if let topK {
            guard (1...256).contains(topK) else {
                throw AppInferenceError.invalidRequest("Top-K must be between 1 and 256.")
            }
        }
        if let topP {
            guard topP > 0, topP <= 1 else {
                throw AppInferenceError.invalidRequest("Top-P must be greater than 0 and at most 1.")
            }
            if temperature > 0, topP < 1, topK == nil {
                throw AppInferenceError.invalidRequest(
                    "Top-P below 1 requires Top-K to be enabled.")
            }
        }
        guard repetitionPenalty >= 1 else {
            throw AppInferenceError.invalidRequest("Repetition penalty must be at least 1.")
        }
        try runtimeOptions.validate()

        if requireModelDirectory {
            var isDirectory = ObjCBool(false)
            guard fileManager.fileExists(atPath: modelDirectory.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw AppInferenceError.modelNotFound(modelDirectory.path)
            }
        }
        if let imageURL {
            let ext = imageURL.pathExtension.lowercased()
            guard ext == "jpg" || ext == "jpeg" || ext == "png" else {
                throw AppInferenceError.invalidRequest("Choose a JPEG or PNG image.")
            }
            let values = try imageURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values.isRegularFile == true, let size = values.fileSize, size > 0,
                  size <= 32 * 1_024 * 1_024 else {
                throw AppInferenceError.invalidRequest("Image file is missing or exceeds the 32 MiB input limit.")
            }
        }
    }
}
