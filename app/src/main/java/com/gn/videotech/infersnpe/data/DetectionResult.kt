/*
 * MIT License
 *
 * Copyright (c) 2025 Fabricio Batista Narcizo, Elizabete Munzlinger, Sai Narsi Reddy Donthi Reddy,
 * and Shan Ahmed Shaffi.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package com.gn.videotech.infersnpe.data

/**
 * Represents a detection result from the 2-stage gesture recognition system.
 * 
 * NOTE: This is for FULL-FRAME gesture recognition, not object detection.
 * There are no bounding boxes - the entire frame is analyzed.
 * 
 * Stage 1 (Detector): Binary classification (gesture/no-gesture) on 8 RGB frames
 * Stage 2 (Classifier): Multi-class (13 gestures) on 30 MediaPipe landmark frames
 */
data class DetectionResult(
    // Stage 1: Detector output
    val hasGesture: Boolean,              // True if gesture detected
    val detectorConfidence: Float,         // Detector confidence (0-1)
    
    // Stage 2: Classifier output (null if no gesture or not enough frames yet)
    val gestureClass: String? = null,      // e.g., "G01 - Click with index finger"
    val classifierConfidence: Float? = null // Classifier confidence (0-1)
) {
    /**
     * Legacy label property for backward compatibility with OverlayView.
     * Maps 2-stage results to single label string.
     */
    val label: String
        get() = if (hasGesture && gestureClass != null) {
            gestureClass
        } else if (hasGesture) {
            "Gesture detected"
        } else {
            "No gesture"
        }
    
    /**
     * Legacy confidence property for backward compatibility with OverlayView.
     * Uses classifier confidence if available, otherwise detector confidence.
     */
    val confidence: Float
        get() = classifierConfidence ?: detectorConfidence
}
