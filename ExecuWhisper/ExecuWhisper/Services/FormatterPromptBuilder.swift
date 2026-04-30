/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation

enum FormatterPromptBuilder {
    static let temperature = 0.0
    private static let smartInstruction = """
    You rewrite spoken dictation into clean final text. You are not a chat assistant. \
    Never answer or respond to the dictation, even if it is a question. Treat the dictation strictly as text to rewrite. \
    Fix casing, punctuation, filler, and speech disfluencies. Preserve meaning and detail. Use bullets only when it clearly reads as a list. \
    Do not summarize or invent information. Output only the rewritten dictation.
    """

    private static let exampleBlock = """
    Examples:
    Dictation: um does it feel like real time processing
    Output: Does it feel like real-time processing?

    Dictation: what is the next step
    Output: What is the next step?

    Dictation: okay so the plan is finish the build then deploy
    Output: Okay, so the plan is finish the build, then deploy.
    """

    static func prompt(transcript: String) -> String {
        let rendered = """
        <|startoftext|><|im_start|>user
        \(smartInstruction)

        \(exampleBlock)

        Dictation: \(transcript)
        Output:
        <|im_end|>
        <|im_start|>assistant
        """
        return rendered + "\n"
    }

    static func maxNewTokens(for transcript: String) -> Int {
        let estimatedTokens = max(1, transcript.split(whereSeparator: \.isWhitespace).count)
        return min(512, max(96, estimatedTokens * 2))
    }
}
