/*
 * MIT License
 *
 * Copyright (c) 2025
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
package com.gn.videotech.infersnpe.ml

import android.app.Application
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import com.gn.videotech.infersnpe.data.DetectionResult
import com.gn.videotech.infersnpe.utils.BitmapUtility
import com.gn.videotech.infersnpe.utils.resized
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.qualcomm.qti.QnnDelegate
import com.qualcomm.qti.snpe.FloatTensor
import com.qualcomm.qti.snpe.NeuralNetwork
import com.qualcomm.qti.snpe.SNPE
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.LinkedList

class TwoStageGestureHelper(
    private val application: Application,
    private val isUnsignedPD: Boolean = true
) {
    // === SNPE (Detector - always SNPE, Classifier - when using .dlc) ===
    private var detectorNetwork: NeuralNetwork? = null
    private var classifierNetwork: NeuralNetwork? = null
    private var detectorInputTensor: FloatTensor? = null
    private var classifierInputTensor: FloatTensor? = null
    private var detectorInputMap: Map<String, FloatTensor>? = null
    private var classifierInputMap: Map<String, FloatTensor>? = null
    private var detectorInputShape = IntArray(0)
    private var classifierInputShape = IntArray(0)

    // === TFLite (Classifier only - when using .tflite) ===
    private var classifierInterpreter: Interpreter? = null
    private var qnnDelegate: QnnDelegate? = null
    private var gpuDelegate: GpuDelegate? = null
    private var isTFLiteClassifier = false

    // === MediaPipe ===
    private var handLandmarker: HandLandmarker? = null

    // === Frame queues ===
    private val detectorQueue = LinkedList<FloatArray>()
    private val classifierQueue = LinkedList<FloatArray>()

    private val DETECTOR_QUEUE_SIZE = 8
    private val CLASSIFIER_QUEUE_SIZE = 30
    private val DETECTOR_INPUT_SIZE = 112
    private val LANDMARK_DIM = 63

    private val IPN_MEAN = floatArrayOf(0.3890f, 0.3937f, 0.3851f)
    private val IPN_STD = floatArrayOf(0.2416f, 0.2375f, 0.2359f)

    private val bitmapUtility = BitmapUtility()

    private val gestureClasses = arrayOf(
        "B0A - Pointing with one finger",
        "B0B - Point with two fingers",
        "G01 - Click with index finger",
        "G02 - Click with two fingers",
        "G03 - Throw up",
        "G04 - Throw down",
        "G05 - Throw left",
        "G06 - Throw right",
        "G07 - Open twice",
        "G08 - Double click with index",
        "G09 - Double click with two fingers",
        "G10 - Zoom in",
        "G11 - Zoom out"
    )

    private var latestLandmarks: FloatArray? = null
    private val landmarkLock = Any()

    companion object {
        private const val TAG = "TwoStageGesture"
        private const val DETECTOR_INPUT = "clip_input"
        private const val DETECTOR_OUTPUT = "output_0"
        private const val CLASSIFIER_INPUT = "sequence_input"
        private const val CLASSIFIER_OUTPUT = "output_0"
    }

    private fun logVersionInfo() {
        Log.d(TAG, "--- Runtime Version Info ---")
        Log.d(TAG, "Android Version: ${Build.VERSION.RELEASE}")
        Log.d(TAG, "Android Build ID: ${Build.ID}")
        Log.d(TAG, "---------------------------")
    }

    fun loadModels(runtimeChar: Char, classifierModelName: String = "classifier_mediapipe_tcn.dlc"): Boolean {
        logVersionInfo()

        if (!initializeMediaPipe()) {
            Log.e(TAG, "Failed to initialize MediaPipe")
            return false
        }

        disposeNetworks()

        val runtime = when (runtimeChar) {
            'G' -> NeuralNetwork.Runtime.GPU
            'D' -> NeuralNetwork.Runtime.DSP
            else -> NeuralNetwork.Runtime.CPU
        }

        Log.d(TAG, "Loading detector (SNPE)...")
        val detector = loadSNPEModelFromAssets("detector_tsm_int8.dlc", runtime) ?: return false
        val detectorShape = detector.inputTensorsShapes[DETECTOR_INPUT] ?: return false

        detectorInputShape = detectorShape
        detectorInputTensor = detector.createFloatTensor(*detectorShape)
        detectorInputMap = mapOf(DETECTOR_INPUT to detectorInputTensor!!)
        detectorNetwork = detector
        Log.d(TAG, "Detector loaded: input shape = ${detectorShape.contentToString()}")

        isTFLiteClassifier = classifierModelName.endsWith(".tflite")

        return if (isTFLiteClassifier) {
            Log.d(TAG, "Loading classifier (TFLite): $classifierModelName")
            loadTFLiteClassifier(classifierModelName, runtimeChar)
        } else {
            Log.d(TAG, "Loading classifier (SNPE): $classifierModelName")
            loadSNPEClassifier(classifierModelName, runtime)
        }
    }

    private fun loadSNPEClassifier(modelName: String, runtime: NeuralNetwork.Runtime): Boolean {
        val classifier = loadSNPEModelFromAssets(modelName, runtime) ?: return false
        val classifierShape = classifier.inputTensorsShapes[CLASSIFIER_INPUT] ?: return false

        classifierInputShape = classifierShape
        classifierInputTensor = classifier.createFloatTensor(*classifierShape)
        classifierInputMap = mapOf(CLASSIFIER_INPUT to classifierInputTensor!!)
        classifierNetwork = classifier
        Log.d(TAG, "SNPE Classifier loaded: input shape = ${classifierShape.contentToString()}")
        return true
    }

    private fun loadTFLiteClassifier(modelName: String, runtimeChar: Char): Boolean {
        return try {
            val options = Interpreter.Options()

            when (runtimeChar) {
                'D' -> {
                    if (!tryAddQnnDelegate(options)) {
                        Log.w(TAG, "QNN delegate failed, falling back to CPU")
                        options.setNumThreads(4)
                    }
                }
                'G' -> {
                    val compatList = CompatibilityList()
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        gpuDelegate = GpuDelegate()
                        options.addDelegate(gpuDelegate)
                        Log.d(TAG, "TFLite GPU delegate enabled")
                    } else {
                        Log.w(TAG, "GPU delegate not supported, using CPU")
                        options.setNumThreads(4)
                    }
                }
                else -> {
                    options.setNumThreads(4)
                    Log.d(TAG, "TFLite CPU mode with 4 threads")
                }
            }

            val modelBuffer = loadTFLiteModelFromAssets(modelName) ?: return false
            classifierInterpreter = Interpreter(modelBuffer, options)

            val inputTensor = classifierInterpreter?.getInputTensor(0)
            classifierInputShape = inputTensor?.shape() ?: return false

            Log.d(TAG, "TFLite Classifier loaded: input shape = ${classifierInputShape.contentToString()}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite classifier: $modelName", e)
            false
        }
    }

    private fun tryAddQnnDelegate(options: Interpreter.Options): Boolean {
        return try {
            val qnnOptions = QnnDelegate.Options()
            qnnOptions.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            qnnOptions.setSkelLibraryDir(application.applicationInfo.nativeLibraryDir)

            qnnDelegate = QnnDelegate(qnnOptions)
            options.addDelegate(qnnDelegate)

            Log.d(TAG, "QNN delegate enabled (HTP backend)")
            true
        } catch (e: UnsupportedOperationException) {
            Log.w(TAG, "QNN delegate creation failed: ${e.message}")
            false
        } catch (e: Exception) {
            Log.e(TAG, "QNN delegate error: ${e.message}", e)
            false
        }
    }

    private fun loadSNPEModelFromAssets(filePath: String, runtime: NeuralNetwork.Runtime): NeuralNetwork? {
        return try {
            application.assets.open(filePath).use { stream ->
                SNPE.NeuralNetworkBuilder(application)
                    .setRuntimeCheckOption(
                        if (isUnsignedPD) NeuralNetwork.RuntimeCheckOption.UNSIGNEDPD_CHECK
                        else NeuralNetwork.RuntimeCheckOption.NORMAL_CHECK
                    )
                    .setModel(stream, stream.available())
                    .setPerformanceProfile(NeuralNetwork.PerformanceProfile.HIGH_PERFORMANCE)
                    .setRuntimeOrder(runtime)
                    .setCpuFallbackEnabled(true)
                    .build()
            }
        } catch (e: Exception) {
            Log.e(TAG, "SNPE model loading error: $filePath", e)
            null
        }
    }

    private fun loadTFLiteModelFromAssets(modelName: String): ByteBuffer? {
        return try {
            application.assets.openFd(modelName).use { assetFd ->
                assetFd.createInputStream().channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    assetFd.startOffset,
                    assetFd.declaredLength
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "TFLite model loading error: $modelName", e)
            null
        }
    }

    private fun initializeMediaPipe(): Boolean {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setResultListener(this::onMediaPipeResult)
                .setErrorListener { error ->
                    Log.e(TAG, "MediaPipe error: ${error.message}")
                }
                .build()

            handLandmarker = HandLandmarker.createFromOptions(application, options)
            Log.d(TAG, "MediaPipe initialized successfully (LIVE_STREAM mode)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "MediaPipe initialization failed", e)
            false
        }
    }

    private fun onMediaPipeResult(result: HandLandmarkerResult, image: MPImage) {
        synchronized(landmarkLock) {
            latestLandmarks = if (result.landmarks().isNotEmpty()) {
                val handLandmarks = result.landmarks()[0]

                // Extract landmarks into a flat array
                val rawLandmarks = FloatArray(LANDMARK_DIM)
                var idx = 0
                for (landmark in handLandmarks) {
                    rawLandmarks[idx++] = landmark.x()
                    rawLandmarks[idx++] = landmark.y()
                    rawLandmarks[idx++] = landmark.z()
                }

                // Calculate the center of all landmarks
                var centerX = 0f
                var centerY = 0f
                var centerZ = 0f
                val numLandmarks = LANDMARK_DIM / 3
                for (i in 0 until numLandmarks) {
                    centerX += rawLandmarks[i * 3]
                    centerY += rawLandmarks[i * 3 + 1]
                    centerZ += rawLandmarks[i * 3 + 2]
                }
                centerX /= numLandmarks
                centerY /= numLandmarks
                centerZ /= numLandmarks

                // Subtract the center to normalize the landmarks
                val normalizedLandmarks = FloatArray(LANDMARK_DIM)
                for (i in 0 until numLandmarks) {
                    normalizedLandmarks[i * 3] = rawLandmarks[i * 3] - centerX
                    normalizedLandmarks[i * 3 + 1] = rawLandmarks[i * 3 + 1] - centerY
                    normalizedLandmarks[i * 3 + 2] = rawLandmarks[i * 3 + 2] - centerZ
                }
                normalizedLandmarks
            } else {
                null
            }
        }
    }

    fun inference(bitmap: Bitmap, timestampMs: Long, threshold: Float = 0.4f): List<DetectionResult> {
        runMediaPipeAsync(bitmap, timestampMs)
        addFrameToDetectorQueue(bitmap)

        synchronized(landmarkLock) {
            addLandmarksToClassifierQueue(latestLandmarks)
        }

        if (detectorQueue.size < DETECTOR_QUEUE_SIZE) {
            return emptyList()
        }

        val (hasGesture, detectorConf) = runDetector(threshold)

        if (!hasGesture) {
            return listOf(
                DetectionResult(
                    hasGesture = false,
                    detectorConfidence = detectorConf
                )
            )
        }

        if (classifierQueue.size < CLASSIFIER_QUEUE_SIZE) {
            return listOf(
                DetectionResult(
                    hasGesture = true,
                    detectorConfidence = detectorConf,
                    gestureClass = "gesture (buffering...)",
                    classifierConfidence = null
                )
            )
        }

        val (gestureClass, classifierConf) = runClassifier()

        return listOf(
            DetectionResult(
                hasGesture = true,
                detectorConfidence = detectorConf,
                gestureClass = gestureClass,
                classifierConfidence = classifierConf
            )
        )
    }

    private fun addFrameToDetectorQueue(bitmap: Bitmap) {
        val resized = bitmap.resized(DETECTOR_INPUT_SIZE)
        bitmapUtility.convertBitmapToBuffer(resized)
        val rgbFloats = bitmapUtility.bufferToFloatsRGB()

        if (bitmapUtility.isBufferBlack()) {
            return
        }

        val normalized = FloatArray(rgbFloats.size)
        for (i in rgbFloats.indices) {
            val channel = i % 3
            normalized[i] = (rgbFloats[i] - IPN_MEAN[channel]) / IPN_STD[channel]
        }

        if (detectorQueue.size >= DETECTOR_QUEUE_SIZE) {
            detectorQueue.removeFirst()
        }
        detectorQueue.addLast(normalized)
    }

    private fun runMediaPipeAsync(bitmap: Bitmap, timestampMs: Long) {
        try {
            val argbBitmap = if (bitmap.config == Bitmap.Config.ARGB_8888) {
                bitmap
            } else {
                bitmap.copy(Bitmap.Config.ARGB_8888, false)
            }
            val mpImage = BitmapImageBuilder(argbBitmap).build()
            handLandmarker?.detectAsync(mpImage, timestampMs)
        } catch (e: Exception) {
            Log.e(TAG, "MediaPipe async detection error", e)
        }
    }

    private fun addLandmarksToClassifierQueue(landmarks: FloatArray?) {
        val landmarkData = landmarks ?: FloatArray(LANDMARK_DIM) { 0f }

        if (classifierQueue.size >= CLASSIFIER_QUEUE_SIZE) {
            classifierQueue.removeFirst()
        }
        classifierQueue.addLast(landmarkData)
    }

    private fun runDetector(threshold: Float): Pair<Boolean, Float> {
        if (detectorNetwork == null || detectorInputTensor == null || detectorInputMap == null) {
            return false to 0f
        }

        val output = runCatching {
            val inputData = FloatArray(1 * 8 * 3 * 112 * 112)
            var idx = 0

            for (t in 0 until 8) {
                val frame = detectorQueue[t]
                for (c in 0 until 3) {
                    for (h in 0 until 112) {
                        for (w in 0 until 112) {
                            val pixelIdx = (h * 112 + w) * 3 + c
                            inputData[idx++] = frame[pixelIdx]
                        }
                    }
                }
            }

            detectorInputTensor?.write(inputData, 0, inputData.size)
            detectorNetwork?.execute(detectorInputMap)
        }.onFailure {
            Log.e(TAG, "Detector inference error", it)
        }.getOrNull()

        if (output == null) {
            return false to 0f
        }

        val outputTensor = output[DETECTOR_OUTPUT] ?: return false to 0f

        val outputData = FloatArray(2)
        outputTensor.read(outputData, 0, 2)

        val maxLogit = maxOf(outputData[0], outputData[1])
        val exp0 = kotlin.math.exp(outputData[0] - maxLogit)
        val exp1 = kotlin.math.exp(outputData[1] - maxLogit)
        val sum = exp0 + exp1
        val gestureProb = exp1 / sum

        val hasGesture = gestureProb >= threshold

        Log.d(TAG, "Detector: gesture_prob=$gestureProb (logits: [${outputData[0]}, ${outputData[1]}]), hasGesture=$hasGesture")

        return hasGesture to gestureProb
    }

    private fun runClassifier(): Pair<String, Float> {
        return if (isTFLiteClassifier) {
            runTFLiteClassifier()
        } else {
            runSNPEClassifier()
        }
    }

    private fun runSNPEClassifier(): Pair<String, Float> {
        if (classifierNetwork == null || classifierInputTensor == null || classifierInputMap == null) {
            return "unknown" to 0f
        }

        val output = runCatching {
            val inputData = FloatArray(30 * 63)
            var offset = 0

            for (landmarks in classifierQueue) {
                System.arraycopy(landmarks, 0, inputData, offset, landmarks.size)
                offset += landmarks.size
            }

            classifierInputTensor?.write(inputData, 0, inputData.size)
            classifierNetwork?.execute(classifierInputMap)
        }.onFailure {
            Log.e(TAG, "SNPE Classifier inference error", it)
        }.getOrNull()

        if (output == null) {
            return "unknown" to 0f
        }

        val outputTensor = output[CLASSIFIER_OUTPUT] ?: return "unknown" to 0f
        val numClasses = gestureClasses.size
        val outputData = FloatArray(numClasses)
        outputTensor.read(outputData, 0, numClasses)

        var bestIdx = 0
        var maxScore = outputData[0]
        for (i in 1 until numClasses) {
            if (outputData[i] > maxScore) {
                maxScore = outputData[i]
                bestIdx = i
            }
        }

        val predictedClass = gestureClasses[bestIdx]
        Log.d(TAG, "SNPE Classifier: class=$predictedClass, confidence=$maxScore")

        return predictedClass to maxScore
    }

    private fun runTFLiteClassifier(): Pair<String, Float> {
        val interpreter = classifierInterpreter ?: return "unknown" to 0f

        return try {
            val inputSize = 30 * 63
            val inputBuffer = ByteBuffer.allocateDirect(inputSize * 4).order(ByteOrder.nativeOrder())

            for (landmarks in classifierQueue) {
                for (value in landmarks) {
                    inputBuffer.putFloat(value)
                }
            }
            inputBuffer.rewind()

            val numClasses = gestureClasses.size
            val outputBuffer = ByteBuffer.allocateDirect(numClasses * 4).order(ByteOrder.nativeOrder())

            interpreter.run(inputBuffer, outputBuffer)

            outputBuffer.rewind()
            var bestIdx = 0
            var maxScore = -Float.MAX_VALUE
            for (i in 0 until numClasses) {
                val score = outputBuffer.float
                if (score > maxScore) {
                    maxScore = score
                    bestIdx = i
                }
            }

            val predictedClass = gestureClasses[bestIdx]
            Log.d(TAG, "TFLite Classifier: class=$predictedClass, confidence=$maxScore")

            predictedClass to maxScore
        } catch (e: Exception) {
            Log.e(TAG, "TFLite Classifier inference error", e)
            "unknown" to 0f
        }
    }

    fun disposeNetworks() {
        // SNPE cleanup
        detectorNetwork?.release()
        classifierNetwork?.release()
        detectorNetwork = null
        classifierNetwork = null
        detectorInputShape = IntArray(0)
        classifierInputShape = IntArray(0)
        detectorInputTensor = null
        classifierInputTensor = null
        detectorInputMap = null
        classifierInputMap = null

        // TFLite cleanup
        classifierInterpreter?.close()
        classifierInterpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
        qnnDelegate?.close()
        qnnDelegate = null
        isTFLiteClassifier = false

        detectorQueue.clear()
        classifierQueue.clear()
    }

    fun dispose() {
        handLandmarker?.close()
        disposeNetworks()
    }
}