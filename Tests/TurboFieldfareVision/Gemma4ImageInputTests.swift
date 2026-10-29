import CoreGraphics
import Foundation
import ImageIO
import MLX
import Testing
@testable import TurboFieldfare
import TurboFieldfareVision

@Suite struct Gemma4ImageInputTests {
    @Test func geometryMatchesGemmaPatchPoolingBudget() throws {
        let square = try Gemma4ImageGeometry(sourceWidth: 1024, sourceHeight: 1024,
                                             maxSoftTokens: 70)
        #expect(square.width == 384)
        #expect(square.height == 384)
        #expect(square.patchCount == 576)
        #expect(square.softTokenCount == 64)

        let panoramic = try Gemma4ImageGeometry(sourceWidth: 2048, sourceHeight: 1024,
                                                 maxSoftTokens: 70)
        #expect(panoramic.width % 48 == 0)
        #expect(panoramic.height % 48 == 0)
        #expect(panoramic.patchCount <= 630)
        #expect(panoramic.softTokenCount <= 70)
        #expect(panoramic.softTokenCount > 0)
    }

    @Test func orientedPngIsResizedIntoPlanarRgbPixels() throws {
        let url = try makeImage(width: 48, height: 96, type: "public.png", orientation: 6)
        defer { try? FileManager.default.removeItem(at: url) }
        let prepared = try Gemma4ImagePreprocessor.pixels(from: url)
        #expect(prepared.geometry.width > prepared.geometry.height)
        #expect(prepared.values.count == prepared.geometry.width * prepared.geometry.height * 3)

        let pixel = (prepared.geometry.height / 2) * prepared.geometry.width
            + prepared.geometry.width / 2
        let pixelsPerChannel = prepared.geometry.width * prepared.geometry.height
        // ImageIO may color-convert the source values, so check that all
        // channels survive as visible normalized values instead of checking
        // for an exact sRGB byte-to-float mapping.
        #expect(Float(prepared.values[pixel]) > 0.9)
        #expect(Float(prepared.values[pixelsPerChannel + pixel]) > 0.1)
        #expect(Float(prepared.values[2 * pixelsPerChannel + pixel]) > 0.05)
    }

    @Test func malformedAndUnsupportedImagesFailClosed() throws {
        let malformed = FileManager.default.temporaryDirectory
            .appendingPathComponent("bad-\(UUID().uuidString).png")
        try Data("not an image".utf8).write(to: malformed)
        defer { try? FileManager.default.removeItem(at: malformed) }
        #expect(throws: Gemma4ImageError.self) {
            _ = try Gemma4ImagePreprocessor.pixels(from: malformed)
        }

        let tiff = try makeImage(width: 32, height: 32, type: "public.tiff")
        defer { try? FileManager.default.removeItem(at: tiff) }
        #expect(throws: Gemma4ImageError.unsupportedFormat) {
            _ = try Gemma4ImagePreprocessor.pixels(from: tiff)
        }
    }

    @Test func gemmaMarkersExpandAtTheOriginalImagePosition() throws {
        #expect(Gemma4ImagePromptExpander.beginImageTokenID == 255_999)
        #expect(Gemma4ImagePromptExpander.imageTokenID == 258_880)
        #expect(Gemma4ImagePromptExpander.endImageTokenID == 258_882)
        let expanded = try Gemma4ImagePromptExpander.expand([10, 20,
                                                              Gemma4ImagePromptExpander.imageTokenID,
                                                              30], rowCount: 3)
        #expect(expanded.imageRowsOffset == 3)
        #expect(expanded.rowCount == 3)
        #expect(expanded.tokenIDs == [10, 20,
                                      Gemma4ImagePromptExpander.beginImageTokenID,
                                      Gemma4ImagePromptExpander.imageTokenID,
                                      Gemma4ImagePromptExpander.imageTokenID,
                                      Gemma4ImagePromptExpander.imageTokenID,
                                      Gemma4ImagePromptExpander.endImageTokenID,
                                      30])
    }

    @Test func imageBlockStaysInsideOnePrefillChunk() {
        let spans = PrefillChunkPlanner.spans(tokenCount: 260, startPosition: 0,
                                               chunkTokens: 128,
                                               protectedRange: 120..<190)
        #expect(spans.map(\.tokenOffset) == [0, 120, 248])
        #expect(spans.map(\.tokenCount) == [120, 128, 12])
        #expect(spans[1].tokenOffset <= 120)
        #expect(spans[1].tokenOffset + spans[1].tokenCount >= 190)
    }

    @Test func memoryGateLeavesSafetyMarginAndFailsClosed() throws {
        #expect(Gemma4ImageMemoryPolicy.processBudgetBytes <= 6 * 1_024 * 1_024 * 1_024)
        #expect(Gemma4ImageMemoryPolicy.processBudgetBytes <= ProcessInfo.processInfo.physicalMemory / 2)
        #expect(Gemma4ImageMemoryPolicy.admissionLimitBytes < Gemma4ImageMemoryPolicy.stopLimitBytes)
        #expect(Gemma4ImageMemoryPolicy.stopLimitBytes < Gemma4ImageMemoryPolicy.processBudgetBytes)
        try Gemma4ImageMemoryPolicy.admit(currentFootprint: Gemma4ImageMemoryPolicy.admissionLimitBytes)
        #expect(throws: Gemma4ImageError.self) {
            try Gemma4ImageMemoryPolicy.admit(currentFootprint: Gemma4ImageMemoryPolicy.admissionLimitBytes + 1)
        }
        #expect(throws: Gemma4ImageError.self) {
            try Gemma4ImageMemoryPolicy.admit(currentFootprint: nil)
        }
        try Gemma4ImageMemoryPolicy.checkDuringImageRequest(
            currentFootprint: Gemma4ImageMemoryPolicy.stopLimitBytes)
        #expect(throws: Gemma4ImageError.self) {
            try Gemma4ImageMemoryPolicy.checkDuringImageRequest(
                currentFootprint: Gemma4ImageMemoryPolicy.stopLimitBytes + 1)
        }
    }

    @Test func mlxMemoryLimitIsAppliedAndRestored() {
        let old = Gemma4ImageMemoryPolicy.configureMLX()
        #expect(Memory.memoryLimit == Gemma4ImageMemoryPolicy.mlxMemoryLimitBytes)
        #expect(Memory.cacheLimit == Gemma4ImageMemoryPolicy.mlxCacheLimitBytes)
        Gemma4ImageMemoryPolicy.restoreMLX(old)
    }

    @Test func mlxEvaluationErrorsAreReturnedToTheCaller() throws {
        #expect(throws: MLXError.self) {
            try withError {
                let lhs = MLXArray(0..<10, [2, 5])
                let rhs = MLXArray(0..<15, [3, 5])
                eval(lhs + rhs)
            }
        }
    }

    private func makeImage(width: Int,
                           height: Int,
                           type: String,
                           orientation: Int? = nil) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("image-\(UUID().uuidString).\(type == "public.tiff" ? "tiff" : "img")")
        let color = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CGContext(data: nil, width: width, height: height,
                                bitsPerComponent: 8, bytesPerRow: width * 4,
                                space: color,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 1, green: 0.25, blue: 0.1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = context.makeImage()!
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, type as CFString, 1, nil) else {
            throw Gemma4ImageError.malformedImage
        }
        let options: CFDictionary? = orientation.map { [kCGImagePropertyOrientation: $0] as CFDictionary }
        CGImageDestinationAddImage(destination, image, options)
        guard CGImageDestinationFinalize(destination) else { throw Gemma4ImageError.malformedImage }
        return url
    }
}
