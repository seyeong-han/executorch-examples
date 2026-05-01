/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation
import Testing

@MainActor
struct TextPipelineTests {
    @Test
    func replacementsApplyLongestMatchFirst() {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = [
            ReplacementEntry(trigger: "young", replacement: "Young"),
            ReplacementEntry(trigger: "young han", replacement: "Younghan"),
            ReplacementEntry(trigger: "mtia", replacement: "MTIA"),
        ]
        let pipeline = TextPipeline(replacementStore: replacementStore)

        let result = pipeline.process("young han joined mtia")

        #expect(result.outputText == "Younghan joined MTIA")
        #expect(result.tags == ["replacement"])
    }

    @Test
    func replacementsPreserveCaseAndWordBoundaryRules() {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = [
            ReplacementEntry(trigger: "executorch", replacement: "ExecuTorch"),
            ReplacementEntry(trigger: "ml", replacement: "ML"),
        ]
        let pipeline = TextPipeline(replacementStore: replacementStore)

        let result = pipeline.process("EXECUTORCH powers xml and ml")

        #expect(result.outputText == "EXECUTORCH powers xml and ML")
    }

    @Test
    func processLeavesTextUnchangedWhenNoRulesMatch() {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = []
        let pipeline = TextPipeline(replacementStore: replacementStore)

        let result = pipeline.process("plain transcript text")

        #expect(result.outputText == "plain transcript text")
        #expect(result.tags.isEmpty)
        #expect(!result.transformed)
    }

    @Test
    func disabledSmartFormattingBypassesFormatterAndAppliesReplacements() async {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = [
            ReplacementEntry(trigger: "executorch", replacement: "ExecuTorch"),
        ]
        let formatter = StubFormatterBridge(result: "should not be used")
        let pipeline = TextPipeline(
            replacementStore: replacementStore,
            formatterBridge: formatter,
            formatterPathsProvider: formatterPaths
        )

        let result = await pipeline.process(
            "executorch raw text",
            smartFormattingEnabled: false
        )

        #expect(result.outputText == "ExecuTorch raw text")
        #expect(result.tags == ["replacement"])
        #expect(await formatter.prompts.isEmpty)
    }

    @Test
    func smartFormattingUsesFormatterThenAppliesReplacements() async {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = [
            ReplacementEntry(trigger: "executorch", replacement: "ExecuTorch"),
        ]
        let formatter = StubFormatterBridge(result: "executorch is ready.")
        let pipeline = TextPipeline(
            replacementStore: replacementStore,
            formatterBridge: formatter,
            formatterPathsProvider: formatterPaths
        )

        let result = await pipeline.process(
            "um executorch is ready",
            smartFormattingEnabled: true
        )

        #expect(result.outputText == "ExecuTorch is ready.")
        #expect(result.tags == ["formatted", "replacement"])
        #expect(await formatter.prompts.count == 1)
        #expect(await formatter.prompts.first?.contains("You rewrite spoken dictation into clean final text.") == true)
        #expect(await formatter.prompts.first?.contains("Never answer or respond to the dictation") == true)
        #expect(await formatter.prompts.first?.contains("Mode: Clean Dictation") == false)
    }

    @Test
    func formatterFailureFallsBackToReplacementOnlyText() async {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = [
            ReplacementEntry(trigger: "executorch", replacement: "ExecuTorch"),
        ]
        let formatter = StubFormatterBridge(error: RunnerError.transcriptionFailed(description: "boom"))
        let pipeline = TextPipeline(
            replacementStore: replacementStore,
            formatterBridge: formatter,
            formatterPathsProvider: formatterPaths
        )

        let result = await pipeline.process(
            "executorch fallback",
            smartFormattingEnabled: true
        )

        #expect(result.outputText == "ExecuTorch fallback")
        #expect(result.tags == ["replacement", "formatter-fallback"])
    }

    @Test
    func formatterAnsweringTheTranscriptQuestionFallsBackToTranscript() async {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = []
        let formatter = StubFormatterBridge(result: "Yes")
        let pipeline = TextPipeline(
            replacementStore: replacementStore,
            formatterBridge: formatter,
            formatterPathsProvider: formatterPaths
        )

        let result = await pipeline.process(
            "does it feel like real-time processing?",
            smartFormattingEnabled: true
        )

        #expect(result.outputText == "does it feel like real-time processing?")
        #expect(result.tags == ["formatter-fallback"])
    }

    @Test
    func formatterAnsweringShortQuestionFallsBackToTranscript() async {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = []
        let formatter = StubFormatterBridge(result: "Yes")
        let pipeline = TextPipeline(
            replacementStore: replacementStore,
            formatterBridge: formatter,
            formatterPathsProvider: formatterPaths
        )

        let result = await pipeline.process(
            "is it raining?",
            smartFormattingEnabled: true
        )

        #expect(result.outputText == "is it raining?")
        #expect(result.tags == ["formatter-fallback"])
    }

    @Test
    func formatterPromptExampleLeakFallsBackToTranscript() async {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = []
        let formatter = StubFormatterBridge(result: """
        Options:
        - Does it feel like real-time processing?
        - What is the next step?
        - Okay, so the plan is finish the build, then deploy
        """)
        let pipeline = TextPipeline(
            replacementStore: replacementStore,
            formatterBridge: formatter,
            formatterPathsProvider: formatterPaths
        )

        let result = await pipeline.process(
            "Hello, can you hear me?",
            smartFormattingEnabled: true
        )

        #expect(result.outputText == "Hello, can you hear me?")
        #expect(result.tags == ["formatter-fallback"])
    }

    @Test
    func longTranscriptSkipsFormatterBeforeContextOverflow() async {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = []
        let formatter = StubFormatterBridge(result: "should not be used")
        let pipeline = TextPipeline(
            replacementStore: replacementStore,
            formatterBridge: formatter,
            formatterPathsProvider: formatterPaths
        )
        let longTranscript = Array(repeating: "context", count: 400).joined(separator: " ")

        let result = await pipeline.process(
            longTranscript,
            smartFormattingEnabled: true
        )

        #expect(result.outputText == longTranscript)
        #expect(result.tags == ["formatter-skipped-context"])
        #expect(await formatter.prompts.isEmpty)
    }

    @Test
    func formatterMetadataEchoFallsBackToTranscript() async {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = [
            ReplacementEntry(trigger: "parakeet", replacement: "Parakeet"),
        ]
        let formatter = StubFormatterBridge(result: "Mode: Clean Dictation")
        let pipeline = TextPipeline(
            replacementStore: replacementStore,
            formatterBridge: formatter,
            formatterPathsProvider: formatterPaths
        )

        let result = await pipeline.process(
            "parakeet helper is ready",
            smartFormattingEnabled: true
        )

        #expect(result.outputText == "Parakeet helper is ready")
        #expect(result.tags == ["replacement", "formatter-fallback"])
    }

    @Test
    func transcriptStorePersistsRawAndProcessedTranscripts() async throws {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = [
            ReplacementEntry(trigger: "executorch", replacement: "ExecuTorch"),
        ]
        let pipeline = TextPipeline(replacementStore: replacementStore)
        let sessionsURL = sandbox.appendingPathComponent("sessions.json")
        let preferences = Preferences()
        preferences.enableSmartFormatting = false
        let store = TranscriptStore(
            preferences: preferences,
            downloader: ModelDownloader(),
            sessionsURL: sessionsURL,
            textPipeline: pipeline
        )

        await store.storeCompletedTranscription(rawText: "executorch rocks", duration: 3)

        let saved = try #require(store.sessions.first)
        #expect(saved.rawTranscript == "executorch rocks")
        #expect(saved.transcript == "ExecuTorch rocks")
        #expect(saved.tags == ["replacement"])
    }

    @Test
    func transcriptStorePersistsFormatterOutputWhenSmartFormattingIsEnabled() async throws {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = [
            ReplacementEntry(trigger: "executorch", replacement: "ExecuTorch"),
        ]
        let formatter = StubFormatterBridge(result: "executorch rocks.")
        let pipeline = TextPipeline(
            replacementStore: replacementStore,
            formatterBridge: formatter,
            formatterPathsProvider: formatterPaths
        )
        let preferences = Preferences()
        preferences.enableSmartFormatting = true
        let sessionsURL = sandbox.appendingPathComponent("sessions.json")
        let store = TranscriptStore(
            preferences: preferences,
            downloader: ModelDownloader(),
            sessionsURL: sessionsURL,
            textPipeline: pipeline
        )

        await store.storeCompletedTranscription(rawText: "um executorch rocks", duration: 3)

        let saved = try #require(store.sessions.first)
        #expect(saved.rawTranscript == "um executorch rocks")
        #expect(saved.transcript == "ExecuTorch rocks.")
        #expect(saved.tags == ["formatted", "replacement"])
    }

    @Test
    func finishDictationWithoutActiveRecordingThrowsSoftCancelWithoutSettingError() async {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = []
        let pipeline = TextPipeline(replacementStore: replacementStore)
        let sessionsURL = sandbox.appendingPathComponent("sessions.json")
        let preferences = Preferences()
        preferences.enableSmartFormatting = false
        let store = TranscriptStore(
            preferences: preferences,
            downloader: ModelDownloader(),
            sessionsURL: sessionsURL,
            textPipeline: pipeline
        )

        var caught: Error?
        do {
            _ = try await store.finishDictationCapture()
        } catch {
            caught = error
        }

        if let runnerError = caught as? RunnerError, case .dictationNotActive = runnerError {
            // expected
        } else {
            Issue.record("Expected RunnerError.dictationNotActive, got \(String(describing: caught))")
        }
        #expect(store.currentError == nil)
    }

    @Test
    func dictationTranscriptionProcessesTextWithoutPersistingHistory() async {
        let sandbox = makeSandbox()
        let replacementStore = ReplacementStore(fileURL: sandbox.appendingPathComponent("replacements.json"))
        replacementStore.entries = [
            ReplacementEntry(trigger: "executorch", replacement: "ExecuTorch"),
        ]
        let pipeline = TextPipeline(replacementStore: replacementStore)
        let sessionsURL = sandbox.appendingPathComponent("sessions.json")
        let preferences = Preferences()
        preferences.enableSmartFormatting = false
        let store = TranscriptStore(
            preferences: preferences,
            downloader: ModelDownloader(),
            sessionsURL: sessionsURL,
            textPipeline: pipeline
        )

        let result = await store.storeDictationTranscription(rawText: "executorch rocks", duration: 3)

        #expect(result.outputText == "ExecuTorch rocks")
        #expect(store.sessions.isEmpty)
        #expect(store.selectedSessionID == nil)
        #expect(store.liveTranscript == "ExecuTorch rocks")
    }

    private func makeSandbox() -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func formatterPaths() -> TextPipeline.FormatterPaths {
        TextPipeline.FormatterPaths(
            runnerPath: "/tmp/lfm25_formatter_helper",
            modelPath: "/tmp/lfm2_5_350m_mlx_4w.pte",
            tokenizerPath: "/tmp/tokenizer.json",
            tokenizerConfigPath: "/tmp/tokenizer_config.json"
        )
    }
}

actor StubFormatterBridge: FormatterBridgeClient {
    private let result: String?
    private let error: Error?
    private(set) var prompts: [String] = []

    init(result: String) {
        self.result = result
        self.error = nil
    }

    init(error: Error) {
        self.result = nil
        self.error = error
    }

    func runtimeSnapshot() async -> FormatterBridge.RuntimeSnapshot {
        FormatterBridge.RuntimeSnapshot(
            state: .warm,
            runnerPath: nil,
            modelPath: nil,
            tokenizerPath: nil,
            tokenizerConfigPath: nil,
            statusMessage: "Formatter ready"
        )
    }

    func prepare(
        runnerPath: String,
        modelPath: String,
        tokenizerPath: String,
        tokenizerConfigPath: String
    ) async throws {}

    func shutdown() async {}

    func format(
        runnerPath: String,
        modelPath: String,
        tokenizerPath: String,
        tokenizerConfigPath: String,
        prompt: String,
        maxNewTokens: Int,
        temperature: Double
    ) async throws -> FormatterBridge.FormatResult {
        prompts.append(prompt)
        if let error {
            throw error
        }
        return FormatterBridge.FormatResult(
            text: result ?? "",
            stdout: "",
            stderr: "",
            tokensPerSecond: nil
        )
    }
}
