/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation
import os

private let pipelineLog = Logger(subsystem: "org.pytorch.executorch.ExecuWhisper", category: "TextPipeline")

struct TextProcessingResult: Sendable, Equatable {
    let rawText: String
    let outputText: String
    let tags: [String]

    var transformed: Bool {
        rawText != outputText
    }
}

@MainActor
final class TextPipeline {
    enum Context: Sendable {
        case standard
        case dictation
    }

    struct FormatterPaths: Sendable {
        let runnerPath: String
        let modelPath: String
        let tokenizerPath: String
        let tokenizerConfigPath: String
    }

    private let replacementStore: ReplacementStore
    private let formatterBridge: (any FormatterBridgeClient)?
    private let formatterPathsProvider: @MainActor () -> FormatterPaths?

    init(
        replacementStore: ReplacementStore,
        formatterBridge: (any FormatterBridgeClient)? = nil,
        formatterPathsProvider: @escaping @MainActor () -> FormatterPaths? = { nil }
    ) {
        self.replacementStore = replacementStore
        self.formatterBridge = formatterBridge
        self.formatterPathsProvider = formatterPathsProvider
    }

    func process(_ text: String, context: Context = .standard) -> TextProcessingResult {
        processReplacementsOnly(text)
    }

    func process(
        _ text: String,
        context: Context = .standard,
        smartFormattingEnabled: Bool
    ) async -> TextProcessingResult {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return TextProcessingResult(rawText: text, outputText: "", tags: [])
        }

        pipelineLog.info("Parakeet transcript: \(trimmed, privacy: .public)")

        guard smartFormattingEnabled else {
            pipelineLog.info("Smart formatting disabled; using replacement-only path")
            return processReplacementsOnly(trimmed)
        }

        guard let formatterBridge, let formatterPaths = formatterPathsProvider() else {
            pipelineLog.info("Formatter unavailable; falling back to replacement-only text")
            return fallbackResult(for: trimmed)
        }

        do {
            let prompt = FormatterPromptBuilder.prompt(transcript: trimmed)
            let formatted = try await formatterBridge.format(
                runnerPath: formatterPaths.runnerPath,
                modelPath: formatterPaths.modelPath,
                tokenizerPath: formatterPaths.tokenizerPath,
                tokenizerConfigPath: formatterPaths.tokenizerConfigPath,
                prompt: prompt,
                maxNewTokens: FormatterPromptBuilder.maxNewTokens(for: trimmed),
                temperature: FormatterPromptBuilder.temperature
            )
            pipelineLog.info("LFM2.5 raw output: \(formatted.text, privacy: .public)")
            guard let validated = validateFormatterOutput(
                formatted.text,
                prompt: prompt,
                transcript: trimmed
            ) else {
                pipelineLog.info("LFM2.5 output rejected by validator; falling back to transcript")
                return fallbackResult(for: trimmed)
            }
            let replaced = applyReplacements(to: validated)
            var tags = ["formatted"]
            if replaced != validated {
                tags.append("replacement")
            }
            pipelineLog.info("LFM2.5 final output: \(replaced, privacy: .public) tags=\(tags, privacy: .public)")
            return TextProcessingResult(rawText: trimmed, outputText: replaced, tags: tags)
        } catch {
            pipelineLog.error("Formatter error: \(error.localizedDescription, privacy: .public)")
            return fallbackResult(for: trimmed)
        }
    }

    func processReplacementsOnly(_ text: String) -> TextProcessingResult {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return TextProcessingResult(rawText: text, outputText: "", tags: [])
        }

        let replaced = applyReplacements(to: trimmed)
        let styled = applyStyle(to: replaced)
        let tags = styled == trimmed ? [] : ["replacement"]
        return TextProcessingResult(rawText: trimmed, outputText: styled, tags: tags)
    }

    private func fallbackResult(for trimmed: String) -> TextProcessingResult {
        let replacementOnly = processReplacementsOnly(trimmed)
        return TextProcessingResult(
            rawText: replacementOnly.rawText,
            outputText: replacementOnly.outputText,
            tags: replacementOnly.tags + ["formatter-fallback"]
        )
    }

    private func applyReplacements(to text: String) -> String {
        replacementStore.entries
            .filter(\.isEnabled)
            .sorted { $0.trigger.count > $1.trigger.count }
            .reduce(text) { partial, entry in
                replace(entry: entry, in: partial)
            }
    }

    private func replace(entry: ReplacementEntry, in text: String) -> String {
        guard !entry.trigger.isEmpty else { return text }

        let escaped = NSRegularExpression.escapedPattern(for: entry.trigger)
        let pattern = entry.requiresWordBoundary ? #"\b\#(escaped)\b"# : escaped
        let options: NSRegularExpression.Options = entry.isCaseSensitive ? [] : [.caseInsensitive]

        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
            return text
        }

        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        guard !matches.isEmpty else { return text }

        var output = text
        for match in matches.reversed() {
            guard let matchRange = Range(match.range, in: output) else { continue }
            let original = String(output[matchRange])
            let replacement = preserveCaseIfNeeded(original: original, replacement: entry.replacement)
            output.replaceSubrange(matchRange, with: replacement)
        }
        return output
    }

    private func preserveCaseIfNeeded(original: String, replacement: String) -> String {
        if original == original.uppercased() {
            return replacement.uppercased()
        }
        if original == original.lowercased() {
            return replacement
        }
        if let first = original.first, String(first) == String(first).uppercased() {
            return replacement.prefix(1).uppercased() + replacement.dropFirst()
        }
        return replacement
    }

    private func applyStyle(to text: String) -> String {
        text
    }

    private func validateFormatterOutput(_ output: String, prompt: String, transcript: String) -> String? {
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard trimmed != prompt else { return nil }

        let lowercased = trimmed.lowercased()
        if lowercased.hasPrefix("here is")
            || lowercased.hasPrefix("sure,")
            || lowercased.hasPrefix("sure.") {
            return nil
        }
        if lowercased.hasPrefix("mode:") {
            return nil
        }
        if trimmed.contains("<|startoftext|>")
            || trimmed.contains("<|im_start|>")
            || trimmed.contains("Transcript:\n\"\"\"") {
            return nil
        }
        if Self.isFormatterOutputSuspiciouslyShort(transcript: transcript, output: trimmed) {
            return nil
        }
        return trimmed
    }

    static func isFormatterOutputSuspiciouslyShort(transcript: String, output: String) -> Bool {
        let inputWordCount = transcript.split(whereSeparator: \.isWhitespace).count
        guard inputWordCount >= 4 else { return false }
        let outputWordCount = output.split(whereSeparator: \.isWhitespace).count
        let minimumExpected = max(2, Int((Double(inputWordCount) * 0.4).rounded(.up)))
        return outputWordCount < minimumExpected
    }
}
