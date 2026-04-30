/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import AVFoundation
import Accelerate
import CoreAudio
import Foundation
import os

private let log = Logger(subsystem: "org.pytorch.executorch.ExecuWhisper", category: "AudioRecorder")

private final class NativeCaptureWriter: @unchecked Sendable {
    private let lock = NSLock()
    private var audioFile: AVAudioFile?
    private var captureURL: URL?

    func append(_ buffer: AVAudioPCMBuffer) throws {
        lock.lock()
        defer { lock.unlock() }

        if audioFile == nil {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("execuwhisper_capture_\(UUID().uuidString).wav")
            let file = try AVAudioFile(
                forWriting: url,
                settings: buffer.format.settings,
                commonFormat: buffer.format.commonFormat,
                interleaved: buffer.format.isInterleaved
            )
            audioFile = file
            captureURL = url
        }

        try audioFile?.write(from: buffer)
    }

    func finish() -> URL? {
        lock.lock()
        defer { lock.unlock() }
        audioFile = nil
        return captureURL
    }
}

actor AudioRecorder {
    struct InputDevice: Identifiable, Equatable, Sendable {
        let id: String
        let name: String
        let isDefault: Bool

        var displayName: String {
            isDefault ? "\(name) (System Default)" : name
        }
    }

    struct ResolvedInputDevice: Equatable, Sendable {
        let device: InputDevice
        let usedFallback: Bool
    }

    private let modelSampleRate: Double = 16_000
    private static let postStopTailTrimDurationMs: Double = 256
    private var engine: AVAudioEngine?
    private var writer = NativeCaptureWriter()

    func startRecording(
        selectedMicrophoneID: String? = nil,
        levelHandler: @Sendable @escaping (Float) -> Void
    ) throws {
        if engine != nil {
            stopCaptureOnly()
        }

        writer = NativeCaptureWriter()

        let availableDevices = Self.availableInputDevices()
        guard let resolvedDevice = Self.resolvePreferredMicrophone(
            selectedMicrophoneID: selectedMicrophoneID,
            availableDevices: availableDevices
        ) else {
            throw RunnerError.microphoneNotAvailable
        }

        let audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode

        if let deviceID = Self.audioDeviceID(forDeviceNamed: resolvedDevice.device.name) {
            try Self.bindEngineInput(inputNode: inputNode, to: deviceID)
        }

        let hwFormat = inputNode.outputFormat(forBus: 0)

        log.info("Hardware audio format: \(hwFormat.sampleRate)Hz, \(hwFormat.channelCount)ch")

        guard hwFormat.sampleRate > 0, hwFormat.channelCount > 0 else {
            throw RunnerError.microphoneNotAvailable
        }

        let captureWriter = writer
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: hwFormat) { buffer, _ in
            guard buffer.frameLength > 0 else { return }

            if let channelData = buffer.floatChannelData {
                var rms: Float = 0
                vDSP_rmsqv(channelData[0], 1, &rms, vDSP_Length(buffer.frameLength))
                levelHandler(rms)
            }

            do {
                try captureWriter.append(buffer)
            } catch {
                log.error("Failed to write capture buffer: \(error.localizedDescription, privacy: .public)")
            }
        }

        do {
            try audioEngine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            throw error
        }
        self.engine = audioEngine

        if resolvedDevice.usedFallback {
            log.info("Selected microphone unavailable; falling back to system default '\(resolvedDevice.device.name, privacy: .public)'")
        }
        log.info("Audio recording started with microphone '\(resolvedDevice.device.name, privacy: .public)'")
    }

    func stopRecording() throws -> Data {
        stopCaptureOnly()

        guard let captureURL = writer.finish() else {
            throw RunnerError.transcriptionFailed(description: "No audio was captured.")
        }
        defer { try? FileManager.default.removeItem(at: captureURL) }

        let decoded = try ImportedAudioDecoder().decodeAudioFile(at: captureURL)
        guard !decoded.pcmData.isEmpty else {
            throw RunnerError.transcriptionFailed(description: "No audio was captured.")
        }

        let trimmedPCM = Self.trimTrailingPCM(
            decoded.pcmData,
            sampleRate: modelSampleRate,
            trimDurationMs: Self.postStopTailTrimDurationMs
        )
        log.info("Captured \(trimmedPCM.count) bytes of 16kHz float32 PCM")
        return trimmedPCM
    }

    func cancelRecording() {
        stopCaptureOnly()
        if let url = writer.finish() {
            try? FileManager.default.removeItem(at: url)
        }
    }

    // MARK: - Device enumeration

    static func availableInputDevices() -> [InputDevice] {
        let defaultDeviceID = AVCaptureDevice.default(for: .audio)?.uniqueID
        var seenIDs: Set<String> = []

        return discoveredAudioCaptureDevices()
            .compactMap { device in
                guard seenIDs.insert(device.uniqueID).inserted else { return nil }
                return InputDevice(
                    id: device.uniqueID,
                    name: device.localizedName,
                    isDefault: device.uniqueID == defaultDeviceID
                )
            }
            .sorted { lhs, rhs in
                if lhs.isDefault != rhs.isDefault {
                    return lhs.isDefault && !rhs.isDefault
                }
                return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
    }

    static func resolvePreferredMicrophone(
        selectedMicrophoneID: String?,
        availableDevices: [InputDevice]
    ) -> ResolvedInputDevice? {
        guard !availableDevices.isEmpty else { return nil }

        let normalizedSelection = selectedMicrophoneID?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let hasExplicitSelection = normalizedSelection.map { !$0.isEmpty } ?? false

        if let normalizedSelection, !normalizedSelection.isEmpty,
           let exactMatch = availableDevices.first(where: { $0.id == normalizedSelection }) {
            return ResolvedInputDevice(device: exactMatch, usedFallback: false)
        }

        let fallbackDevice = availableDevices.first(where: \.isDefault) ?? availableDevices[0]
        return ResolvedInputDevice(device: fallbackDevice, usedFallback: hasExplicitSelection)
    }

    // MARK: - Utilities

    static func trimTrailingPCM(
        _ pcmData: Data,
        sampleRate: Double,
        trimDurationMs: Double
    ) -> Data {
        guard trimDurationMs > 0 else { return pcmData }

        let bytesPerSample = MemoryLayout<Float>.size
        let trimSampleCount = Int((sampleRate * trimDurationMs) / 1000.0)
        let trimByteCount = trimSampleCount * bytesPerSample
        guard trimByteCount > 0, pcmData.count > trimByteCount + bytesPerSample else {
            return pcmData
        }

        return Data(pcmData.prefix(pcmData.count - trimByteCount))
    }

    // MARK: - Private

    private static func discoveredAudioCaptureDevices() -> [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.microphone],
            mediaType: .audio,
            position: .unspecified
        ).devices
    }

    private func stopCaptureOnly() {
        if let engine {
            engine.inputNode.removeTap(onBus: 0)
            if engine.isRunning {
                engine.stop()
            }
            engine.reset()
        }
        engine = nil
        log.info("Audio recording stopped")
    }

    private static func bindEngineInput(inputNode: AVAudioInputNode, to deviceID: AudioDeviceID) throws {
        guard let audioUnit = inputNode.audioUnit else {
            log.error("Input node has no audio unit; cannot bind to device id=\(deviceID)")
            return
        }
        var mutableID = deviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &mutableID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        if status != noErr {
            log.error("Failed to bind input AU to device id=\(deviceID) status=\(status)")
            throw RunnerError.microphoneNotAvailable
        }
    }

    private static func audioDeviceID(forDeviceNamed name: String) -> AudioDeviceID? {
        var size: UInt32 = 0
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0, nil,
            &size
        ) == noErr, size > 0 else { return nil }

        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0, nil,
            &size,
            &deviceIDs
        ) == noErr else { return nil }

        for deviceID in deviceIDs {
            var nameSize: UInt32 = 0
            var nameAddress = AudioObjectPropertyAddress(
                mSelector: kAudioObjectPropertyName,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            guard AudioObjectGetPropertyDataSize(deviceID, &nameAddress, 0, nil, &nameSize) == noErr else { continue }

            var cfName: CFString = "" as CFString
            var cfNameSize = UInt32(MemoryLayout<CFString>.size)
            guard AudioObjectGetPropertyData(deviceID, &nameAddress, 0, nil, &cfNameSize, &cfName) == noErr else { continue }

            if (cfName as String) == name {
                var inputChannels: UInt32 = 0
                var streamAddress = AudioObjectPropertyAddress(
                    mSelector: kAudioDevicePropertyStreamConfiguration,
                    mScope: kAudioObjectPropertyScopeInput,
                    mElement: kAudioObjectPropertyElementMain
                )
                var streamSize: UInt32 = 0
                if AudioObjectGetPropertyDataSize(deviceID, &streamAddress, 0, nil, &streamSize) == noErr,
                   streamSize > 0 {
                    let bufferListPointer = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: Int(streamSize))
                    defer { bufferListPointer.deallocate() }
                    if AudioObjectGetPropertyData(deviceID, &streamAddress, 0, nil, &streamSize, bufferListPointer) == noErr {
                        let bufferList = UnsafeMutableAudioBufferListPointer(bufferListPointer)
                        for buf in bufferList {
                            inputChannels += buf.mNumberChannels
                        }
                    }
                }
                if inputChannels > 0 { return deviceID }
            }
        }
        return nil
    }

}
