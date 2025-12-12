/*
 * V8: FRAME-BASED VERSION with ENHANCED DEBUGGING
 * - Logs every 10 windows (not 50) to narrow down hang location
 * - Try-catch around window processing to capture silent crashes
 *
 * Expects frames at:
 *   - /sdcard/GestureTestData/frames_resized/<video_name>/frame_00001.jpg (112x112 for detector)
 *   - /sdcard/GestureTestData/frames/<video_name>/frame_00001.jpg (640x480 for MediaPipe)
 */
package com.gn.videotech.infersnpe

import android.app.Application
import android.os.Build
import android.os.Environment
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.gn.videotech.infersnpe.utils.BitmapUtility
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.qualcomm.qti.QnnDelegate
import com.qualcomm.qti.snpe.FloatTensor
import com.qualcomm.qti.snpe.NeuralNetwork
import com.qualcomm.qti.snpe.SNPE
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

@RunWith(AndroidJUnit4::class)
class GestureModelInstrumentedTest {

    companion object {
        private const val TAG = "GestureModelTest"

        private const val DETECTOR_WINDOW = 8
        private const val DETECTOR_STRIDE = 5
        private const val DETECTOR_INPUT_SIZE = 112
        private const val DETECTOR_MIN_POSITIVE_RATIO = 0.7f

        private const val CLASSIFIER_WINDOW = 30
        private const val CLASSIFIER_STRIDE = 10
        private const val LANDMARK_DIM = 63
        private const val NUM_CLASSES = 13

        private val IPN_MEAN = floatArrayOf(0.450097f, 0.422493f, 0.390098f)
        private val IPN_STD = floatArrayOf(0.152967f, 0.148480f, 0.157761f)

        private const val DETECTOR_MODEL = "detector_tsm_int8.dlc"

        private val CLASSIFIER_MODELS = listOf(
            ModelConfig("classifier_mediapipe_gru_int8.dlc", "GRU", "int8", "SNPE"),
            ModelConfig("classifier_mediapipe_tcn_int8.dlc", "TCN", "int8", "SNPE"),
            ModelConfig("classifier_mediapipe_tcn_int8.tflite", "TCN", "int8", "TFLite"),
            ModelConfig("classifier_mediapipe_gru_int8.tflite", "GRU", "int8", "TFLite"),
            ModelConfig("classifier_mediapipe_lstm_int8.tflite", "LSTM", "int8", "TFLite"),
        )

        enum class Runtime { CPU, DSP }

        private val GESTURE_CLASSES = arrayOf(
            "B0A", "B0B", "G01", "G02", "G03", "G04", "G05",
            "G06", "G07", "G08", "G09", "G10", "G11"
        )
    }

    data class ModelConfig(
        val filename: String,
        val architecture: String,
        val quantization: String,
        val framework: String
    )

    private lateinit var application: Application
    private lateinit var bitmapUtility: BitmapUtility
    private lateinit var testDataDir: File
    private lateinit var framesResizedDir: File  // 112x112 frames for detector
    private lateinit var framesFullDir: File     // Full resolution frames for MediaPipe
    private lateinit var logFile: File
    private var logWriter: PrintWriter? = null

    private var handLandmarker: HandLandmarker? = null
    private val testResults = JSONObject()
    private val landmarksCache = mutableMapOf<String, List<FloatArray>>()
    private val videoFrameCountCache = mutableMapOf<String, Int>()  // Cache frame counts
    private var testStartTime: Long = 0

    // ==================== LOGGING ====================

    private fun initLogFile() {
        try {
            logFile = File(testDataDir, "test_log.txt")
            logWriter = PrintWriter(FileWriter(logFile, false))
            logToFile("=".repeat(70))
            logToFile("GESTURE MODEL EVALUATION LOG - V8 ENHANCED DEBUG")
            logToFile("Started: ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())}")
            logToFile("Device: ${Build.MODEL}, Android ${Build.VERSION.SDK_INT}")
            logToFile("=".repeat(70))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create log file: ${e.message}")
        }
    }

    private fun logToFile(message: String) {
        val timestamp = SimpleDateFormat("HH:mm:ss.SSS", Locale.US).format(Date())
        val logMessage = "[$timestamp] $message"
        Log.d(TAG, message)
        try {
            logWriter?.println(logMessage)
            logWriter?.flush()
        } catch (e: Exception) { }
    }

    private fun logProgress(current: Int, total: Int, prefix: String) {
        val percent = (current * 100.0 / total).toInt()
        val elapsed = System.currentTimeMillis() - testStartTime
        val eta = if (current > 0) {
            val remaining = (elapsed * (total - current) / current) / 1000
            "${remaining / 60}m ${remaining % 60}s"
        } else "calculating..."
        logToFile("$prefix: $current/$total ($percent%) - ETA: $eta")
    }

    private fun closeLogFile() {
        try { logWriter?.close() } catch (e: Exception) { }
    }

    // ==================== SETUP / TEARDOWN ====================

    private fun checkAllFilesAccess(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Environment.isExternalStorageManager()
        } else true
    }

    @Before
    fun setup() {
        application = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext as Application

        if (!checkAllFilesAccess()) {
            throw RuntimeException("All-files access not granted.")
        }

        bitmapUtility = BitmapUtility()
        testDataDir = File("/sdcard/GestureTestData")
        framesResizedDir = File(testDataDir, "frames_resized")  // 112x112 for detector
        framesFullDir = File(testDataDir, "frames")             // Full resolution for MediaPipe

        initLogFile()
        testStartTime = System.currentTimeMillis()

        // Verify frame directories exist
        logToFile("Checking frame directories...")
        logToFile("  frames_resized: ${framesResizedDir.absolutePath} - exists: ${framesResizedDir.exists()}")
        logToFile("  frames: ${framesFullDir.absolutePath} - exists: ${framesFullDir.exists()}")

        if (!framesResizedDir.exists()) {
            logToFile("WARNING: frames_resized directory not found! Detector evaluation will fail.")
        }
        if (!framesFullDir.exists()) {
            logToFile("WARNING: frames directory not found! MediaPipe evaluation will fail.")
        }

        testResults.put("test_date", SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date()))
        testResults.put("device", Build.MODEL)
        testResults.put("android_version", Build.VERSION.SDK_INT)

        logToFile("Setup complete. Test data dir: ${testDataDir.absolutePath}")
    }

    @After
    fun teardown() {
        handLandmarker?.close()
        saveResults("final")

        val totalTime = (System.currentTimeMillis() - testStartTime) / 1000
        logToFile("=".repeat(70))
        logToFile("TEST COMPLETE - Total time: ${totalTime / 60}m ${totalTime % 60}s")
        logToFile("=".repeat(70))
        closeLogFile()
    }

    // ==================== MAIN TEST ====================

    @Test
    fun runFullEvaluation() {
        logToFile("========== STARTING FULL EVALUATION ==========")

        val annotations = loadAnnotations()
        if (annotations.isEmpty()) {
            logToFile("ERROR: No annotations loaded!")
            return
        }
        logToFile("Loaded ${annotations.size} test annotations")

        val uniqueVideos = annotations.map { it.videoName }.distinct()
        logToFile("Unique videos: ${uniqueVideos.size}")

        // 1. Detector evaluation
        logToFile("")
        logToFile("========== STEP 1/4: DETECTOR EVALUATION ==========")

        logToFile("--- Detector on CPU ---")
        evaluateDetector(annotations, Runtime.CPU)
        saveResults("after_detector_cpu")

        logToFile("--- Detector on DSP ---")
        evaluateDetector(annotations, Runtime.DSP)
        saveResults("after_detector_dsp")

        // 2. MediaPipe latency
        logToFile("")
        logToFile("========== STEP 2/4: MEDIAPIPE LATENCY ==========")
        evaluateMediaPipeLatency(annotations)
        saveResults("after_mediapipe")

        // 3. Pre-extract landmarks
        logToFile("")
        logToFile("========== STEP 3/4: PRE-EXTRACTING LANDMARKS ==========")
        preExtractLandmarks(annotations)
        saveResults("after_landmarks")

        // 4. Classifier evaluation
        logToFile("")
        logToFile("========== STEP 4/4: CLASSIFIER EVALUATION ==========")
        val totalClassifierRuns = CLASSIFIER_MODELS.size * 2
        var completedRuns = 0

        for (modelConfig in CLASSIFIER_MODELS) {
            for (runtime in listOf(Runtime.CPU, Runtime.DSP)) {
                completedRuns++
                logToFile("")
                logToFile("--- Classifier $completedRuns/$totalClassifierRuns: ${modelConfig.filename} on $runtime ---")
                try {
                    evaluateClassifier(modelConfig, annotations, runtime)
                } catch (e: Exception) {
                    logToFile("ERROR: Failed ${modelConfig.filename} on $runtime: ${e.message}")
                    e.printStackTrace()
                }
                saveResults("after_classifier_${modelConfig.architecture}_${runtime}")
            }
        }

        logToFile("")
        logToFile("========== EVALUATION COMPLETE ==========")
        printSummary()
    }

    // ==================== ANNOTATION LOADING ====================

    data class Annotation(
        val videoPath: String,
        val videoName: String,
        val gestureLabel: String,
        val gestureClass: Int,
        val startFrame: Int,
        val endFrame: Int,
        val isGesture: Boolean
    )

    private fun loadAnnotations(): List<Annotation> {
        val annotations = mutableListOf<Annotation>()
        try {
            val annotFile = File(testDataDir, "annotations/Annot_TestList.txt")
            if (!annotFile.exists()) {
                logToFile("ERROR: Annotation file not found")
                return annotations
            }

            annotFile.bufferedReader().useLines { lines ->
                lines.forEach { line ->
                    val parts = line.trim().split(",")
                    if (parts.size >= 5) {
                        val videoName = parts[0]
                        val gestureLabel = parts[1]
                        val startFrame = parts[3].toIntOrNull() ?: 0
                        val endFrame = parts[4].toIntOrNull() ?: 0
                        val gestureClass = parseGestureClass(gestureLabel)
                        val isGesture = !gestureLabel.startsWith("D0")

                        annotations.add(Annotation(
                            videoPath = "videos/$videoName",
                            videoName = videoName,
                            gestureLabel = gestureLabel,
                            gestureClass = gestureClass,
                            startFrame = startFrame,
                            endFrame = endFrame,
                            isGesture = isGesture
                        ))
                    }
                }
            }
        } catch (e: Exception) {
            logToFile("ERROR: Failed to load annotations: ${e.message}")
        }
        return annotations
    }

    private fun parseGestureClass(label: String): Int {
        return when {
            label.startsWith("B0A") -> 0
            label.startsWith("B0B") -> 1
            label.startsWith("G01") -> 2
            label.startsWith("G02") -> 3
            label.startsWith("G03") -> 4
            label.startsWith("G04") -> 5
            label.startsWith("G05") -> 6
            label.startsWith("G06") -> 7
            label.startsWith("G07") -> 8
            label.startsWith("G08") -> 9
            label.startsWith("G09") -> 10
            label.startsWith("G10") -> 11
            label.startsWith("G11") -> 12
            else -> -1
        }
    }

    // ==================== FRAME LOADING UTILITIES ====================

    /**
     * Get total frame count for a video by counting files in the frames directory.
     * Uses cache to avoid repeated directory scans.
     */
    private fun getVideoTotalFrames(videoName: String): Int {
        // Check cache first
        videoFrameCountCache[videoName]?.let { return it }

        // Count frames in the resized directory (should match full resolution)
        val videoFramesDir = File(framesResizedDir, videoName)
        if (!videoFramesDir.exists() || !videoFramesDir.isDirectory) {
            logToFile("  [WARN] Frame directory not found: ${videoFramesDir.absolutePath}")
            return 0
        }

        val frameCount = videoFramesDir.listFiles { file ->
            file.name.startsWith("frame_") && file.name.endsWith(".jpg")
        }?.size ?: 0

        // Cache the result
        videoFrameCountCache[videoName] = frameCount
        return frameCount
    }

    /**
     * Load a single resized frame (112x112) for detector.
     * Frame numbers are 1-indexed in filenames.
     */
    private fun loadResizedFrame(videoName: String, frameNum: Int): Bitmap? {
        // Frames are 1-indexed in filenames
        val frameFile = File(framesResizedDir, "$videoName/frame_${String.format("%05d", frameNum + 1)}.jpg")

        if (!frameFile.exists()) {
            return null
        }

        return try {
            val options = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            BitmapFactory.decodeFile(frameFile.absolutePath, options)
        } catch (e: Exception) {
            logToFile("  [ERROR] Failed to load frame: ${frameFile.absolutePath}: ${e.message}")
            null
        }
    }

    /**
     * Load a single full-resolution frame (640x480) for MediaPipe.
     * Frame numbers are 1-indexed in filenames.
     */
    private fun loadFullFrame(videoName: String, frameNum: Int): Bitmap? {
        // Frames are 1-indexed in filenames
        val frameFile = File(framesFullDir, "$videoName/frame_${String.format("%05d", frameNum + 1)}.jpg")

        if (!frameFile.exists()) {
            return null
        }

        return try {
            val options = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            BitmapFactory.decodeFile(frameFile.absolutePath, options)
        } catch (e: Exception) {
            logToFile("  [ERROR] Failed to load frame: ${frameFile.absolutePath}: ${e.message}")
            null
        }
    }

    /**
     * Load multiple resized frames for detector window.
     */
    private fun loadResizedFrames(videoName: String, startFrame: Int, count: Int): List<Bitmap> {
        val frames = mutableListOf<Bitmap>()
        for (i in 0 until count) {
            loadResizedFrame(videoName, startFrame + i)?.let { frames.add(it) }
        }
        return frames
    }

    /**
     * Load multiple full-resolution frames for MediaPipe.
     */
    private fun loadFullFrames(videoName: String, startFrame: Int, endFrame: Int): List<Bitmap> {
        val frames = mutableListOf<Bitmap>()
        for (frameNum in startFrame until endFrame) {
            loadFullFrame(videoName, frameNum)?.let { frames.add(it) }
        }
        return frames
    }

    // ==================== DETECTOR EVALUATION ====================

    private fun evaluateDetector(annotations: List<Annotation>, runtime: Runtime) {
        logToFile("  [DEBUG] Loading detector model...")
        val loadStart = System.currentTimeMillis()

        val (network, inputTensor, actualRuntime) = loadDetector(runtime) ?: run {
            logToFile("ERROR: Failed to load detector for $runtime")
            return
        }

        logToFile("  [DEBUG] Detector loaded in ${System.currentTimeMillis() - loadStart}ms")
        logToFile("Detector loaded - Requested: $runtime, Actual: $actualRuntime")

        val videoAnnotations = annotations.groupBy { it.videoName }
        val totalVideos = videoAnnotations.size

        val allLabels = mutableListOf<Int>()
        val allPreds = mutableListOf<Int>()
        val allScores = mutableListOf<Float>()
        val latencies = mutableListOf<Double>()

        val minOverlapFrames = ceil(DETECTOR_WINDOW * DETECTOR_MIN_POSITIVE_RATIO).toInt()
        var processedVideos = 0
        var totalWindows = 0

        for ((videoName, videoAnns) in videoAnnotations) {
            processedVideos++

            logToFile("  [DEBUG] Processing video $processedVideos/$totalVideos: $videoName")

            val frameCountStart = System.currentTimeMillis()
            val totalFrames = getVideoTotalFrames(videoName)
            logToFile("  [DEBUG]   Frame count: $totalFrames (took ${System.currentTimeMillis() - frameCountStart}ms)")

            if (totalFrames < DETECTOR_WINDOW) {
                logToFile("  [DEBUG]   Skipping - not enough frames")
                continue
            }

            // Get gesture intervals
            val gestureIntervals = videoAnns
                .filter { it.isGesture }
                .map { it.startFrame to it.endFrame }

            // Calculate number of windows for this video
            val numWindows = (totalFrames - DETECTOR_WINDOW) / DETECTOR_STRIDE + 1
            logToFile("  [DEBUG]   Will process $numWindows windows")

            var windowStart = 0
            var windowCount = 0

            while (windowStart + DETECTOR_WINDOW <= totalFrames) {
                windowCount++

                try {
                    val windowEnd = windowStart + DETECTOR_WINDOW - 1

                    // Calculate overlap with gesture segments
                    var gestureFrames = 0
                    for ((gestureStart, gestureEnd) in gestureIntervals) {
                        val overlapStart = max(windowStart, gestureStart)
                        val overlapEnd = min(windowEnd, gestureEnd)
                        if (overlapEnd >= overlapStart) {
                            gestureFrames += overlapEnd - overlapStart + 1
                        }
                        if (gestureFrames >= minOverlapFrames) break
                    }

                    val label = if (gestureFrames >= minOverlapFrames) 1 else 0

                    // Load 8 pre-resized frames for this window
                    val loadStart = System.currentTimeMillis()
                    val frames = loadResizedFrames(videoName, windowStart, DETECTOR_WINDOW)
                    val loadTime = System.currentTimeMillis() - loadStart

                    // Log every 10th window for debugging
                    val shouldLog = windowCount == 1 || windowCount % 10 == 0
                    if (shouldLog) {
                        logToFile("  [DEBUG]   Window $windowCount: loaded ${frames.size} frames in ${loadTime}ms (frameStart=$windowStart)")
                    }

                    if (frames.size == DETECTOR_WINDOW) {
                        // Run inference
                        val inferStart = System.currentTimeMillis()
                        val (pred, score) = runDetectorInference(network, inputTensor, frames)
                        val inferTime = System.currentTimeMillis() - inferStart

                        if (shouldLog) {
                            logToFile("  [DEBUG]   Window $windowCount: inference took ${inferTime}ms")
                        }

                        latencies.add(inferTime.toDouble())
                        allLabels.add(label)
                        allPreds.add(pred)
                        allScores.add(score)
                        totalWindows++
                    } else {
                        logToFile("  [WARN]   Window $windowCount: only got ${frames.size} frames, skipping")
                    }

                    // Clean up frames
                    frames.forEach { it.recycle() }

                } catch (e: Exception) {
                    logToFile("  [ERROR] Window $windowCount CRASHED: ${e.javaClass.simpleName}: ${e.message}")
                    logToFile("  [ERROR] Stack trace: ${e.stackTraceToString().take(500)}")
                    throw e
                } catch (e: Error) {
                    logToFile("  [FATAL] Window $windowCount ERROR: ${e.javaClass.simpleName}: ${e.message}")
                    throw e
                }

                windowStart += DETECTOR_STRIDE
            }

            logToFile("  [DEBUG]   Video complete: $windowCount windows processed")

            // Progress update
            logProgress(processedVideos, totalVideos, "Detector[$runtime] videos")
        }

        network.release()

        logToFile("Detector[$runtime]: Processed $totalWindows windows from $processedVideos videos")

        if (allLabels.isEmpty()) {
            logToFile("WARNING: No detector samples processed for $runtime")
            return
        }

        val metrics = calculateBinaryMetrics(allLabels, allPreds)
        val latencyStats = calculateLatencyStats(latencies)

        val detectorResults = JSONObject().apply {
            put("model", DETECTOR_MODEL)
            put("requested_runtime", runtime.name)
            put("actual_runtime", actualRuntime)
            put("num_samples", allLabels.size)
            put("metrics", metrics)
            put("latency", latencyStats)
        }

        val detectorArray = testResults.optJSONArray("detector") ?: JSONArray()
        detectorArray.put(detectorResults)
        testResults.put("detector", detectorArray)

        printDetectorResults(runtime.name, actualRuntime, metrics, latencyStats, allLabels.size)
    }

    private fun loadDetector(runtime: Runtime): Triple<NeuralNetwork, FloatTensor, String>? {
        return try {
            application.assets.open(DETECTOR_MODEL).use { stream ->
                val snpeRuntime = when (runtime) {
                    Runtime.CPU -> NeuralNetwork.Runtime.CPU
                    Runtime.DSP -> NeuralNetwork.Runtime.DSP
                }

                val network = SNPE.NeuralNetworkBuilder(application)
                    .setRuntimeCheckOption(NeuralNetwork.RuntimeCheckOption.UNSIGNEDPD_CHECK)
                    .setModel(stream, stream.available())
                    .setPerformanceProfile(NeuralNetwork.PerformanceProfile.HIGH_PERFORMANCE)
                    .setRuntimeOrder(snpeRuntime)
                    .setCpuFallbackEnabled(runtime == Runtime.DSP)
                    .build()

                val actualRuntime = network.runtime?.name ?: "UNKNOWN"
                val inputShape = network.inputTensorsShapes["clip_input"] ?: return null
                val inputTensor = network.createFloatTensor(*inputShape)

                Triple(network, inputTensor, actualRuntime)
            }
        } catch (e: Exception) {
            logToFile("ERROR: Failed to load detector on $runtime: ${e.message}")
            null
        }
    }

    /**
     * Run detector inference on pre-resized 112x112 frames.
     * No resize needed since frames are already 112x112.
     */
    private fun runDetectorInference(
        network: NeuralNetwork,
        inputTensor: FloatTensor,
        frames: List<Bitmap>
    ): Pair<Int, Float> {
        val inputData = FloatArray(1 * 3 * 8 * 112 * 112)
        var idx = 0

        // NCTHW format - frames are already 112x112, no resize needed
        for (c in 0 until 3) {
            for (t in 0 until 8) {
                bitmapUtility.convertBitmapToBuffer(frames[t])
                val rgbFloats = bitmapUtility.bufferToFloatsRGB()

                for (h in 0 until 112) {
                    for (w in 0 until 112) {
                        val pixelIdx = (h * 112 + w) * 3 + c
                        val normalized = (rgbFloats[pixelIdx] - IPN_MEAN[c]) / IPN_STD[c]
                        inputData[idx++] = normalized
                    }
                }
            }
        }

        inputTensor.write(inputData, 0, inputData.size)
        val outputs = network.execute(mapOf("clip_input" to inputTensor))

        val outputTensor = outputs["output_0"] ?: return 0 to 0f
        val outputData = FloatArray(2)
        outputTensor.read(outputData, 0, 2)

        val maxLogit = max(outputData[0], outputData[1])
        val exp0 = exp(outputData[0] - maxLogit)
        val exp1 = exp(outputData[1] - maxLogit)
        val gestureProb = exp1 / (exp0 + exp1)

        return (if (gestureProb >= 0.5f) 1 else 0) to gestureProb
    }

    // ==================== MEDIAPIPE EVALUATION ====================

    private fun evaluateMediaPipeLatency(annotations: List<Annotation>) {
        logToFile("  [DEBUG] Initializing MediaPipe...")
        if (!initializeMediaPipe()) {
            logToFile("ERROR: Failed to initialize MediaPipe")
            return
        }
        logToFile("  [DEBUG] MediaPipe initialized")

        val latencies = mutableListOf<Double>()
        var processedFrames = 0
        val maxFrames = 1000

        val uniqueVideos = annotations.map { it.videoName }.distinct()
        logToFile("Testing MediaPipe on ${uniqueVideos.size} videos (max $maxFrames frames)")

        for (videoName in uniqueVideos) {
            if (processedFrames >= maxFrames) break

            // Load full-resolution frames for MediaPipe
            val totalFrames = getVideoTotalFrames(videoName)
            val framesToLoad = min(50, totalFrames)
            val frames = loadFullFrames(videoName, 0, framesToLoad)

            for (frame in frames) {
                if (processedFrames >= maxFrames) break

                val startTime = System.nanoTime()
                extractLandmarks(frame)
                val endTime = System.nanoTime()

                latencies.add((endTime - startTime) / 1_000_000.0)
                processedFrames++
            }

            frames.forEach { it.recycle() }

            if (processedFrames % 200 == 0) {
                logProgress(processedFrames, maxFrames, "MediaPipe frames")
            }
        }

        val latencyStats = calculateLatencyStats(latencies)

        testResults.put("mediapipe", JSONObject().apply {
            put("component", "MediaPipe HandLandmarker")
            put("runtime", "CPU/GPU (MediaPipe managed)")
            put("num_frames", processedFrames)
            put("latency", latencyStats)
        })

        logToFile("MediaPipe: $processedFrames frames, ${String.format("%.2f", latencyStats.getDouble("mean_ms"))} ms avg, ${String.format("%.1f", latencyStats.getDouble("fps"))} FPS")
    }

    private fun initializeMediaPipe(): Boolean {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(application, options)
            true
        } catch (e: Exception) {
            logToFile("ERROR: MediaPipe init failed: ${e.message}")
            false
        }
    }

    private fun extractLandmarks(bitmap: Bitmap): FloatArray? {
        val landmarker = handLandmarker ?: return null

        val argbBitmap = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
        else bitmap.copy(Bitmap.Config.ARGB_8888, false)

        val mpImage = BitmapImageBuilder(argbBitmap).build()
        val result = landmarker.detect(mpImage)

        if (result.landmarks().isEmpty()) return null

        val landmarks = result.landmarks()[0]
        val rawLandmarks = Array(21) { FloatArray(3) }
        for (i in 0 until 21) {
            rawLandmarks[i][0] = landmarks[i].x()
            rawLandmarks[i][1] = landmarks[i].y()
            rawLandmarks[i][2] = landmarks[i].z()
        }

        val center = FloatArray(3)
        for (i in 0 until 21) {
            center[0] += rawLandmarks[i][0]
            center[1] += rawLandmarks[i][1]
            center[2] += rawLandmarks[i][2]
        }
        center[0] /= 21f
        center[1] /= 21f
        center[2] /= 21f

        return FloatArray(LANDMARK_DIM).apply {
            var idx = 0
            for (i in 0 until 21) {
                this[idx++] = rawLandmarks[i][0] - center[0]
                this[idx++] = rawLandmarks[i][1] - center[1]
                this[idx++] = rawLandmarks[i][2] - center[2]
            }
        }
    }

    private fun preExtractLandmarks(annotations: List<Annotation>) {
        logToFile("  [DEBUG] Starting landmark pre-extraction...")

        if (handLandmarker == null && !initializeMediaPipe()) {
            logToFile("ERROR: Failed to initialize MediaPipe")
            return
        }

        val gestureAnnotations = annotations.filter { it.isGesture }
        val annotationsByVideo = gestureAnnotations.groupBy { it.videoName }

        val totalSegments = gestureAnnotations.size
        var extractedSegments = 0

        logToFile("Pre-extracting landmarks for $totalSegments gesture segments across ${annotationsByVideo.size} videos...")

        for ((videoName, videoAnns) in annotationsByVideo) {
            logToFile("  [DEBUG] Processing video: $videoName (${videoAnns.size} segments)")

            val videoFramesDir = File(framesFullDir, videoName)
            if (!videoFramesDir.exists()) {
                logToFile("  [DEBUG]   Frame directory not found, skipping")
                continue
            }

            for (annotation in videoAnns) {
                val cacheKey = "${annotation.videoName}_${annotation.startFrame}_${annotation.endFrame}"
                if (landmarksCache.containsKey(cacheKey)) {
                    extractedSegments++
                    continue
                }

                // Load full-resolution frames for MediaPipe
                val frames = loadFullFrames(videoName, annotation.startFrame, annotation.endFrame + 1)

                if (frames.isNotEmpty()) {
                    val landmarks = frames.map { frame ->
                        extractLandmarks(frame) ?: FloatArray(LANDMARK_DIM)
                    }
                    landmarksCache[cacheKey] = landmarks
                }

                frames.forEach { it.recycle() }
                extractedSegments++

                if (extractedSegments % 50 == 0) {
                    logProgress(extractedSegments, totalSegments, "Landmarks extracted")
                }
            }
        }

        logToFile("Pre-extraction complete: ${landmarksCache.size} segments cached")
    }

    // ==================== CLASSIFIER EVALUATION ====================

    @Suppress("UNCHECKED_CAST")
    private fun evaluateClassifier(
        modelConfig: ModelConfig,
        annotations: List<Annotation>,
        runtime: Runtime
    ) {
        logToFile("  [DEBUG] Loading classifier ${modelConfig.filename}...")

        val (classifier, actualRuntime) = when (modelConfig.framework) {
            "SNPE" -> loadSNPEClassifier(modelConfig.filename, runtime)
            "TFLite" -> loadTFLiteClassifier(modelConfig.filename, runtime)
            else -> null to "UNKNOWN"
        } ?: run {
            logToFile("ERROR: Failed to load ${modelConfig.filename} on $runtime")
            return
        }

        logToFile("Classifier loaded - Requested: $runtime, Actual: $actualRuntime")

        val allLabels = mutableListOf<Int>()
        val allPreds = mutableListOf<Int>()
        val allScores = mutableListOf<FloatArray>()
        val latencies = mutableListOf<Double>()

        val gestureAnnotations = annotations.filter { it.isGesture && it.gestureClass >= 0 }
        val total = gestureAnnotations.size
        var processed = 0

        for (annotation in gestureAnnotations) {
            processed++

            val cacheKey = "${annotation.videoName}_${annotation.startFrame}_${annotation.endFrame}"
            val segmentLandmarks = landmarksCache[cacheKey] ?: continue

            val gestureLength = segmentLandmarks.size

            if (gestureLength < CLASSIFIER_WINDOW) {
                val paddedLandmarks = segmentLandmarks.toMutableList()
                while (paddedLandmarks.size < CLASSIFIER_WINDOW) {
                    paddedLandmarks.add(FloatArray(LANDMARK_DIM))
                }

                val startTime = System.nanoTime()
                val (pred, scores) = when (modelConfig.framework) {
                    "SNPE" -> runSNPEClassifierInference(
                        classifier as Pair<NeuralNetwork, FloatTensor>,
                        paddedLandmarks.take(CLASSIFIER_WINDOW)
                    )
                    "TFLite" -> runTFLiteClassifierInference(
                        classifier as Interpreter,
                        paddedLandmarks.take(CLASSIFIER_WINDOW)
                    )
                    else -> 0 to FloatArray(NUM_CLASSES)
                }
                val endTime = System.nanoTime()

                latencies.add((endTime - startTime) / 1_000_000.0)
                allLabels.add(annotation.gestureClass)
                allPreds.add(pred)
                allScores.add(scores)
            } else {
                var windowStart = 0
                val maxStart = gestureLength - CLASSIFIER_WINDOW

                while (windowStart <= maxStart) {
                    val windowLandmarks = segmentLandmarks.subList(windowStart, windowStart + CLASSIFIER_WINDOW)

                    val startTime = System.nanoTime()
                    val (pred, scores) = when (modelConfig.framework) {
                        "SNPE" -> runSNPEClassifierInference(
                            classifier as Pair<NeuralNetwork, FloatTensor>,
                            windowLandmarks
                        )
                        "TFLite" -> runTFLiteClassifierInference(
                            classifier as Interpreter,
                            windowLandmarks
                        )
                        else -> 0 to FloatArray(NUM_CLASSES)
                    }
                    val endTime = System.nanoTime()

                    latencies.add((endTime - startTime) / 1_000_000.0)
                    allLabels.add(annotation.gestureClass)
                    allPreds.add(pred)
                    allScores.add(scores)

                    windowStart += CLASSIFIER_STRIDE
                }
            }

            if (processed % 100 == 0) {
                logProgress(processed, total, "Classifier[${modelConfig.architecture}/$runtime]")
            }
        }

        when (modelConfig.framework) {
            "SNPE" -> (classifier as Pair<NeuralNetwork, FloatTensor>).first.release()
            "TFLite" -> (classifier as Interpreter).close()
        }

        if (allLabels.isEmpty()) {
            logToFile("WARNING: No samples processed")
            return
        }

        val metrics = calculateMulticlassMetrics(allLabels, allPreds, allScores)
        val latencyStats = calculateLatencyStats(latencies)

        val classifierResults = JSONObject().apply {
            put("model_file", modelConfig.filename)
            put("architecture", modelConfig.architecture)
            put("quantization", modelConfig.quantization)
            put("framework", modelConfig.framework)
            put("requested_runtime", runtime.name)
            put("actual_runtime", actualRuntime)
            put("num_samples", allLabels.size)
            put("metrics", metrics)
            put("latency", latencyStats)
        }

        val classifiersArray = testResults.optJSONArray("classifiers") ?: JSONArray()
        classifiersArray.put(classifierResults)
        testResults.put("classifiers", classifiersArray)

        printClassifierResults(modelConfig, runtime.name, actualRuntime, metrics, latencyStats, allLabels.size)
    }

    private fun loadSNPEClassifier(modelName: String, runtime: Runtime): Pair<Pair<NeuralNetwork, FloatTensor>, String>? {
        return try {
            application.assets.open(modelName).use { stream ->
                val snpeRuntime = when (runtime) {
                    Runtime.CPU -> NeuralNetwork.Runtime.CPU
                    Runtime.DSP -> NeuralNetwork.Runtime.DSP
                }

                val network = SNPE.NeuralNetworkBuilder(application)
                    .setRuntimeCheckOption(NeuralNetwork.RuntimeCheckOption.UNSIGNEDPD_CHECK)
                    .setModel(stream, stream.available())
                    .setPerformanceProfile(NeuralNetwork.PerformanceProfile.HIGH_PERFORMANCE)
                    .setRuntimeOrder(snpeRuntime)
                    .setCpuFallbackEnabled(runtime == Runtime.DSP)
                    .build()

                val actualRuntime = network.runtime?.name ?: "UNKNOWN"
                val inputShape = network.inputTensorsShapes["sequence_input"] ?: return null
                val inputTensor = network.createFloatTensor(*inputShape)

                (network to inputTensor) to actualRuntime
            }
        } catch (e: Exception) {
            logToFile("ERROR: Failed to load SNPE classifier $modelName: ${e.message}")
            null
        }
    }

    private fun loadTFLiteClassifier(modelName: String, runtime: Runtime): Pair<Interpreter, String>? {
        return try {
            val options = Interpreter.Options()
            var actualRuntime: String

            when (runtime) {
                Runtime.DSP -> {
                    try {
                        val qnnOptions = QnnDelegate.Options()
                        qnnOptions.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                        qnnOptions.setSkelLibraryDir(application.applicationInfo.nativeLibraryDir)
                        options.addDelegate(QnnDelegate(qnnOptions))
                        actualRuntime = "DSP (QNN HTP)"
                    } catch (e: Exception) {
                        logToFile("WARNING: QNN delegate failed: ${e.message}")
                        options.setNumThreads(4)
                        actualRuntime = "CPU (QNN failed)"
                    }
                }
                Runtime.CPU -> {
                    options.setNumThreads(4)
                    actualRuntime = "CPU"
                }
            }

            val modelBuffer = application.assets.openFd(modelName).use { assetFd ->
                assetFd.createInputStream().channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    assetFd.startOffset,
                    assetFd.declaredLength
                )
            }

            Interpreter(modelBuffer, options) to actualRuntime
        } catch (e: Exception) {
            logToFile("ERROR: Failed to load TFLite classifier $modelName: ${e.message}")
            null
        }
    }

    private fun runSNPEClassifierInference(
        classifier: Pair<NeuralNetwork, FloatTensor>,
        landmarks: List<FloatArray>
    ): Pair<Int, FloatArray> {
        val (network, inputTensor) = classifier

        val inputData = FloatArray(CLASSIFIER_WINDOW * LANDMARK_DIM)
        var offset = 0
        for (lm in landmarks) {
            System.arraycopy(lm, 0, inputData, offset, lm.size)
            offset += lm.size
        }

        inputTensor.write(inputData, 0, inputData.size)
        val outputs = network.execute(mapOf("sequence_input" to inputTensor))

        val outputTensor = outputs["output_0"] ?: return 0 to FloatArray(NUM_CLASSES)
        val outputData = FloatArray(NUM_CLASSES)
        outputTensor.read(outputData, 0, NUM_CLASSES)

        val scores = softmax(outputData)
        return (scores.indices.maxByOrNull { scores[it] } ?: 0) to scores
    }

    private fun runTFLiteClassifierInference(
        interpreter: Interpreter,
        landmarks: List<FloatArray>
    ): Pair<Int, FloatArray> {
        val inputBuffer = ByteBuffer.allocateDirect(CLASSIFIER_WINDOW * LANDMARK_DIM * 4)
            .order(ByteOrder.nativeOrder())

        for (lm in landmarks) {
            for (value in lm) {
                inputBuffer.putFloat(value)
            }
        }
        inputBuffer.rewind()

        val outputBuffer = ByteBuffer.allocateDirect(NUM_CLASSES * 4)
            .order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val outputData = FloatArray(NUM_CLASSES) { outputBuffer.float }

        val scores = softmax(outputData)
        return (scores.indices.maxByOrNull { scores[it] } ?: 0) to scores
    }

    // ==================== METRICS ====================

    private fun calculateBinaryMetrics(labels: List<Int>, preds: List<Int>): JSONObject {
        val n = labels.size
        var tp = 0; var tn = 0; var fp = 0; var fn = 0

        for (i in 0 until n) {
            when {
                labels[i] == 1 && preds[i] == 1 -> tp++
                labels[i] == 0 && preds[i] == 0 -> tn++
                labels[i] == 0 && preds[i] == 1 -> fp++
                labels[i] == 1 && preds[i] == 0 -> fn++
            }
        }

        val accuracy = (tp + tn).toFloat() / n
        val precision = if (tp + fp > 0) tp.toFloat() / (tp + fp) else 0f
        val recall = if (tp + fn > 0) tp.toFloat() / (tp + fn) else 0f
        val f1 = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0f
        val specificity = if (tn + fp > 0) tn.toFloat() / (tn + fp) else 0f

        return JSONObject().apply {
            put("accuracy", accuracy)
            put("balanced_accuracy", (recall + specificity) / 2)
            put("precision", precision)
            put("recall", recall)
            put("f1", f1)
            put("specificity", specificity)
            put("false_positive_rate", if (fp + tn > 0) fp.toFloat() / (fp + tn) else 0f)
            put("false_negative_rate", if (fn + tp > 0) fn.toFloat() / (fn + tp) else 0f)
            put("true_positives", tp)
            put("true_negatives", tn)
            put("false_positives", fp)
            put("false_negatives", fn)
            put("confusion_matrix", JSONArray().apply {
                put(JSONArray().apply { put(tn); put(fp) })
                put(JSONArray().apply { put(fn); put(tp) })
            })
        }
    }

    private fun calculateMulticlassMetrics(
        labels: List<Int>,
        preds: List<Int>,
        scores: List<FloatArray>
    ): JSONObject {
        val n = labels.size
        val confusionMatrix = Array(NUM_CLASSES) { IntArray(NUM_CLASSES) }

        for (i in 0 until n) {
            if (labels[i] in 0 until NUM_CLASSES && preds[i] in 0 until NUM_CLASSES) {
                confusionMatrix[labels[i]][preds[i]]++
            }
        }

        val perClassPrecision = FloatArray(NUM_CLASSES)
        val perClassRecall = FloatArray(NUM_CLASSES)
        val perClassF1 = FloatArray(NUM_CLASSES)
        val perClassSupport = IntArray(NUM_CLASSES)

        for (c in 0 until NUM_CLASSES) {
            val tp = confusionMatrix[c][c]
            val fp = (0 until NUM_CLASSES).sumOf { if (it != c) confusionMatrix[it][c] else 0 }
            val fn = (0 until NUM_CLASSES).sumOf { if (it != c) confusionMatrix[c][it] else 0 }

            perClassPrecision[c] = if (tp + fp > 0) tp.toFloat() / (tp + fp) else 0f
            perClassRecall[c] = if (tp + fn > 0) tp.toFloat() / (tp + fn) else 0f
            perClassF1[c] = if (perClassPrecision[c] + perClassRecall[c] > 0)
                2 * perClassPrecision[c] * perClassRecall[c] / (perClassPrecision[c] + perClassRecall[c])
            else 0f
            perClassSupport[c] = labels.count { it == c }
        }

        val accuracy = (0 until n).count { labels[it] == preds[it] }.toFloat() / n
        val totalSupport = perClassSupport.sum().toFloat()

        val top3Correct = (0 until n).count { i ->
            labels[i] in scores[i].indices.sortedByDescending { scores[i][it] }.take(3)
        }

        return JSONObject().apply {
            put("accuracy", accuracy)
            put("precision_macro", perClassPrecision.average().toFloat())
            put("recall_macro", perClassRecall.average().toFloat())
            put("f1_macro", perClassF1.average().toFloat())
            put("precision_weighted", (0 until NUM_CLASSES).sumOf { (perClassPrecision[it] * perClassSupport[it]).toDouble() }.toFloat() / totalSupport)
            put("recall_weighted", (0 until NUM_CLASSES).sumOf { (perClassRecall[it] * perClassSupport[it]).toDouble() }.toFloat() / totalSupport)
            put("f1_weighted", (0 until NUM_CLASSES).sumOf { (perClassF1[it] * perClassSupport[it]).toDouble() }.toFloat() / totalSupport)
            put("top_3_accuracy", top3Correct.toFloat() / n)
            put("confusion_matrix", JSONArray().apply {
                for (row in confusionMatrix) put(JSONArray().apply { row.forEach { put(it) } })
            })
            put("per_class", JSONObject().apply {
                put("class_names", JSONArray().apply { GESTURE_CLASSES.forEach { put(it) } })
                put("precision", JSONArray().apply { perClassPrecision.forEach { put(it) } })
                put("recall", JSONArray().apply { perClassRecall.forEach { put(it) } })
                put("f1", JSONArray().apply { perClassF1.forEach { put(it) } })
                put("support", JSONArray().apply { perClassSupport.forEach { put(it) } })
            })
        }
    }

    private fun calculateLatencyStats(latencies: List<Double>): JSONObject {
        if (latencies.isEmpty()) {
            return JSONObject().apply {
                listOf("mean_ms", "std_ms", "min_ms", "max_ms", "median_ms", "p95_ms", "p99_ms", "fps").forEach { put(it, 0.0) }
            }
        }

        val sorted = latencies.sorted()
        val mean = latencies.average()
        val std = kotlin.math.sqrt(latencies.map { (it - mean) * (it - mean) }.average())

        return JSONObject().apply {
            put("mean_ms", mean)
            put("std_ms", std)
            put("min_ms", sorted.first())
            put("max_ms", sorted.last())
            put("median_ms", sorted[sorted.size / 2])
            put("p95_ms", sorted[(sorted.size * 0.95).toInt().coerceAtMost(sorted.size - 1)])
            put("p99_ms", sorted[(sorted.size * 0.99).toInt().coerceAtMost(sorted.size - 1)])
            put("fps", if (mean > 0) 1000.0 / mean else 0.0)
            put("num_samples", latencies.size)
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { exp(it - maxLogit) }
        val sum = exps.sum()
        return exps.map { it / sum }.toFloatArray()
    }

    // ==================== OUTPUT ====================

    private fun printDetectorResults(requestedRuntime: String, actualRuntime: String, metrics: JSONObject, latency: JSONObject, numSamples: Int) {
        logToFile("")
        logToFile("Detector Results [$requestedRuntime -> $actualRuntime] ($numSamples samples):")
        logToFile("  Accuracy: ${String.format("%.4f", metrics.getDouble("accuracy"))}, F1: ${String.format("%.4f", metrics.getDouble("f1"))}")
        logToFile("  Precision: ${String.format("%.4f", metrics.getDouble("precision"))}, Recall: ${String.format("%.4f", metrics.getDouble("recall"))}")
        logToFile("  Latency: ${String.format("%.2f", latency.getDouble("mean_ms"))} ms, FPS: ${String.format("%.1f", latency.getDouble("fps"))}")
    }

    private fun printClassifierResults(modelConfig: ModelConfig, requestedRuntime: String, actualRuntime: String, metrics: JSONObject, latency: JSONObject, numSamples: Int) {
        logToFile("")
        logToFile("${modelConfig.architecture} [${modelConfig.framework}] [$requestedRuntime -> $actualRuntime] ($numSamples samples):")
        logToFile("  Accuracy: ${String.format("%.4f", metrics.getDouble("accuracy"))}, F1 (macro): ${String.format("%.4f", metrics.getDouble("f1_macro"))}")
        logToFile("  Latency: ${String.format("%.2f", latency.getDouble("mean_ms"))} ms, FPS: ${String.format("%.1f", latency.getDouble("fps"))}")
    }

    private fun printSummary() {
        logToFile("")
        logToFile("=".repeat(70))
        logToFile("EVALUATION SUMMARY")
        logToFile("=".repeat(70))

        testResults.optJSONArray("detector")?.let { detectors ->
            logToFile("\nDETECTOR ($DETECTOR_MODEL):")
            for (i in 0 until detectors.length()) {
                val d = detectors.getJSONObject(i)
                val m = d.getJSONObject("metrics")
                val l = d.getJSONObject("latency")
                logToFile("  [${d.getString("requested_runtime")} -> ${d.getString("actual_runtime")}] Acc: ${String.format("%.2f%%", m.getDouble("accuracy") * 100)}, F1: ${String.format("%.4f", m.getDouble("f1"))}, ${String.format("%.1f", l.getDouble("fps"))} FPS")
            }
        }

        testResults.optJSONArray("classifiers")?.let { classifiers ->
            logToFile("\nCLASSIFIERS:")
            for (i in 0 until classifiers.length()) {
                val c = classifiers.getJSONObject(i)
                val m = c.getJSONObject("metrics")
                val l = c.getJSONObject("latency")
                logToFile("  ${c.getString("model_file")} [${c.getString("requested_runtime")} -> ${c.getString("actual_runtime")}] Acc: ${String.format("%.2f%%", m.getDouble("accuracy") * 100)}, F1: ${String.format("%.4f", m.getDouble("f1_macro"))}, ${String.format("%.1f", l.getDouble("fps"))} FPS")
            }
        }

        logToFile("=".repeat(70))
    }

    private fun saveResults(stage: String = "intermediate") {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            File(application.getExternalFilesDir(null), "evaluation_results_$timestamp.json").writeText(testResults.toString(2))
            File(application.getExternalFilesDir(null), "evaluation_results_latest.json").writeText(testResults.toString(2))
            File(testDataDir, "evaluation_results_latest.json").writeText(testResults.toString(2))
            logToFile("Results saved ($stage)")
        } catch (e: Exception) {
            logToFile("ERROR saving results: ${e.message}")
        }
    }
}