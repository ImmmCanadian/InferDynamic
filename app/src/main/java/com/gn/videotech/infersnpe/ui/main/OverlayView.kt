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
package com.gn.videotech.infersnpe.ui.main

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import com.gn.videotech.infersnpe.data.DetectionResult
import com.google.android.material.color.MaterialColors

/**
 * A custom view for rendering gesture recognition results overlay.
 *
 * MODIFIED for full-frame gesture recognition:
 * - Removed bounding box drawing (no object detection)
 * - Shows gesture label centered on screen
 * - Uses detector/classifier confidence appropriately
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    /**
     * The primary color used for drawing overlays.
     */
    private val primaryColor = MaterialColors.getColor(
        context, com.google.android.material.R.attr.colorPrimary, Color.RED)

    /**
     * The color used for text labels on overlays.
     */
    private val primaryTextColor = MaterialColors.getColor(
        context, com.google.android.material.R.attr.colorOnPrimary, Color.YELLOW)

    /**
     * The width of the source image used for reference.
     */
    var imageWidth: Int = 480

    /**
     * The height of the source image used for reference.
     */
    var imageHeight: Int = 640

    /**
     * A flag indicating whether the camera is in front or back position.
     */
    var isFrontCamera: Boolean = false

    /**
     * A reusable [Rect] instance used to measure text dimensions.
     */
    private val textBounds = Rect()

    /**
     * The list of detection results to be rendered on the overlay.
     */
    private var detectionResults: List<DetectionResult> = emptyList()

    /**
     * Paint used to draw gesture labels.
     *
     * Configured with anti-aliasing, primary text color, and a text size of 40sp for visibility.
     */
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = primaryTextColor
        textSize = 40f
        style = Paint.Style.FILL
        textAlign = Paint.Align.CENTER
    }

    /**
     * Paint used to draw background rectangles for label annotations.
     *
     * Configured with anti-aliasing, same color as primary color, and a fill style.
     */
    private val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = primaryColor
        style = Paint.Style.FILL
        alpha = 200  // Semi-transparent
    }

    /**
     * Updates the list of detection results and triggers a redraw of the overlay.
     *
     * This method replaces the current detection list with the provided results and calls
     * [invalidate] to refresh the view, causing [onDraw] to be invoked.
     *
     * @param results A list of new [DetectionResult] objects to render on the overlay.
     */
    fun updateDetections(results: List<DetectionResult>) {
        detectionResults = results
        invalidate()
    }

    /**
     * Draws gesture recognition results on the canvas overlay.
     *
     * MODIFIED for full-frame gesture recognition:
     * - No bounding boxes (entire frame is analyzed)
     * - Shows gesture label centered on screen
     * - Different colors for different states:
     *   - No gesture: dim text
     *   - Gesture buffering: yellow text
     *   - Gesture classified: bright green text
     *
     * @param canvas The canvas on which the overlay content is drawn.
     */
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (imageWidth == 0 || imageHeight == 0 || detectionResults.isEmpty()) return

        val padding = 20f

        // Get first result (we only have one for full-frame detection)
        val result = detectionResults.firstOrNull() ?: return

        // Determine text and color based on detection state
        val (label, textColor) = when {
            !result.hasGesture -> {
                "No gesture" to Color.GRAY
            }
            result.gestureClass == null -> {
                "Detecting..." to Color.YELLOW
            }
            else -> {
                result.label to Color.GREEN
            }
        }

        labelPaint.color = textColor

        // Calculate text position (center of screen, lowered by 250 pixels)
        val centerX = width / 2f
        val centerY = height / 2f + 250f

        // Handle multi-line text
        val lines = label.split("\n")
        val lineHeight = labelPaint.textSize * 1.2f
        val totalTextHeight = lines.size * lineHeight
        var currentY = centerY - (totalTextHeight / 2f) + lineHeight

        // Measure first line for background width
        labelPaint.getTextBounds(lines[0], 0, lines[0].length, textBounds)
        val bgWidth = textBounds.width() + padding * 4
        val bgHeight = totalTextHeight + padding * 2

        // Draw background
        val bgLeft = centerX - bgWidth / 2f
        val bgTop = centerY - totalTextHeight / 2f - padding
        val bgRight = centerX + bgWidth / 2f
        val bgBottom = centerY + totalTextHeight / 2f + padding

        // Different background colors for different states
        backgroundPaint.color = when {
            !result.hasGesture -> Color.argb(150, 50, 50, 50)  // Dark gray
            result.gestureClass == null -> Color.argb(150, 255, 200, 0)  // Orange
            else -> primaryColor  // Primary color
        }

        canvas.drawRoundRect(bgLeft, bgTop, bgRight, bgBottom, 20f, 20f, backgroundPaint)

        // Draw each line of text
        for (line in lines) {
            canvas.drawText(line, centerX, currentY, labelPaint)
            currentY += lineHeight
        }

        // Draw detector confidence bar at bottom if gesture detected
        if (result.hasGesture) {
            drawConfidenceBar(canvas, result)
        }
    }

    /**
     * Draws a confidence bar at the bottom of the screen.
     *
     * @param canvas The canvas to draw on.
     * @param result The detection result with confidence values.
     */
    private fun drawConfidenceBar(canvas: Canvas, result: DetectionResult) {
        val barHeight = 30f
        val barWidth = width * 0.8f
        val barLeft = (width - barWidth) / 2f
        val barTop = height - barHeight - 50f
        val barRight = barLeft + barWidth
        val barBottom = barTop + barHeight

        // Background bar
        backgroundPaint.color = Color.argb(150, 50, 50, 50)
        canvas.drawRoundRect(barLeft, barTop, barRight, barBottom, 15f, 15f, backgroundPaint)

        // Filled bar (detector confidence)
        val fillWidth = barWidth * result.detectorConfidence
        val fillColor = if (result.detectorConfidence > 0.5f) {
            Color.GREEN
        } else {
            Color.YELLOW
        }
        backgroundPaint.color = fillColor
        canvas.drawRoundRect(barLeft, barTop, barLeft + fillWidth, barBottom, 15f, 15f, backgroundPaint)

        // Text label
        labelPaint.textSize = 24f
        labelPaint.color = Color.WHITE
        val confidenceText = "Confidence: ${(result.detectorConfidence * 100).toInt()}%"
        canvas.drawText(confidenceText, width / 2f, barTop - 10f, labelPaint)

        // Reset text size
        labelPaint.textSize = 40f
    }
}