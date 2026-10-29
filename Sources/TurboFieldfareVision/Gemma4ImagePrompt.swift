import Foundation

public struct Gemma4ExpandedImagePrompt: Sendable, Equatable {
    public let tokenIDs: [Int32]
    public let imageRowsOffset: Int
    public let rowCount: Int
}

public enum Gemma4ImagePromptExpander {
    public static let imageMarker = "<|image|>"
    // BOI and EOI are inserted by ID because their tokenizer spellings are
    // asymmetric (`<|image` and `<image|>`), not `<|boi|>` / `<|eoi|>`.
    public static let imageTokenID: Int32 = 258_880
    public static let beginImageTokenID: Int32 = 255_999
    public static let endImageTokenID: Int32 = 258_882

    public static func expand(_ tokenIDs: [Int32], rowCount: Int) throws -> Gemma4ExpandedImagePrompt {
        guard (1...Gemma4ImageGeometry.maxSoftTokens).contains(rowCount) else {
            throw Gemma4ImageError.invalidDimensions
        }
        let matches = tokenIDs.indices.filter { tokenIDs[$0] == imageTokenID }
        guard matches.count == 1 else {
            throw Gemma4ImageError.incompatibleWeights("prompt must contain one image placeholder")
        }
        let marker = matches[0]
        var expanded = Array(tokenIDs[..<marker])
        expanded.append(beginImageTokenID)
        let offset = expanded.count
        expanded.append(contentsOf: repeatElement(imageTokenID, count: rowCount))
        expanded.append(endImageTokenID)
        expanded.append(contentsOf: tokenIDs[(marker + 1)...])
        return Gemma4ExpandedImagePrompt(tokenIDs: expanded,
                                         imageRowsOffset: offset,
                                         rowCount: rowCount)
    }
}
