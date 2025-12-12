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

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.util.Range
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfoUnavailableException
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import com.gn.videotech.infersnpe.R
import com.gn.videotech.infersnpe.data.DetectionResult
import com.gn.videotech.infersnpe.databinding.FragmentMainBinding
import com.gn.videotech.infersnpe.ml.TwoStageGestureHelper
import com.gn.videotech.infersnpe.utils.FPSTracker
import com.gn.videotech.infersnpe.utils.toBitmapConverter
import com.gn.videotech.infersnpe.viewmodel.CameraViewModel
import com.google.android.material.snackbar.Snackbar
import java.util.concurrent.Executors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.os.SystemClock
import android.util.Size
import android.view.Surface
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy

/**
 * A fragment to display the main screen of the app.
 *
 * MODIFIED for 2-stage gesture recognition:
 * - Changed SNPEHelper → TwoStageGestureHelper
 * - Added timestampMs for MediaPipe
 * - Removed model selection (using fixed 2-stage system)
 */
class MainFragment : Fragment() {

    /**
     * View binding is a feature that allows you to more easily write code that interacts with
     * views. Once view binding is enabled in a module, it generates a binding class for each XML
     * layout file present in that module. An instance of a binding class contains direct references
     * to all views that have an ID in the corresponding layout.
     */
    private var _binding: FragmentMainBinding? = null

    /**
     * This property is only valid between `onCreateView()` and `onDestroyView()` methods.
     */
    private val binding
        get() = requireNotNull(_binding) {
            "Cannot access binding because it is null. Is the view visible?"
        }

    /**
     * A view model to manage the data access to the database. Using lazy initialization to create
     * the view model instance when the user access the object for the first time.
     */
    private val viewModel: CameraViewModel by activityViewModels()

    /**
     * This instance manages the 2-stage gesture recognition model.
     * CHANGED: SNPEHelper → TwoStageGestureHelper
     */
    private lateinit var gestureHelper: TwoStageGestureHelper

    /**
     * The camera selector allows to select a camera or return a filtered set of cameras.
     */
    private var cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    /**
     * This instance provides `takePicture()` functions to take a picture to memory or save to a
     * file, and provides image metadata.
     */
    private var imageCapture: ImageCapture? = null

    /**
     * Add a background executor to optimize the image analysis execution.
     */
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    /**
     * This object measures the frames per second (FPS).
     */
    private var fpsTracker = FPSTracker()

    /**
     * This instance manages if the neural network is loaded.
     */
    private var isNetworkLoaded = false

    /**
     * This instance sets the confidence threshold for the detector model.
     * CHANGED: Default from 0.5 → 0.4 (matching your Python code)
     */
    private var confidenceThreshold = 0.4f

    /**
     * This instance sets the selected runtime for the model.
     */
    private var selectedRuntimeChar = 'D'

    /**
     * This instance sets the selected classifier model.
     */
    private var selectedClassifierModel = "classifier_mediapipe_tcn_fp32.dlc"

    /**
     * This object launches a new permission dialog and receives back the user's permission.
     */
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { result ->
        cameraPermissionResult(result)
    }

    /**
     * Called to have the fragment instantiate its user interface view. This is optional, and
     * non-graphical fragments can return null. This will be called between `onCreate(Bundle)` and
     * `onViewCreated(View, Bundle)`. A default `View` can be returned by calling `Fragment(int)` in
     * your constructor. Otherwise, this method returns null.
     *
     * It is recommended to <strong>only</strong> inflate the layout in this method and move logic
     * that operates on the returned View to `onViewCreated(View, Bundle)`.
     *
     * If you return a `View` from here, you will later be called in `onDestroyView()` when the view
     * is being released.
     *
     * @param inflater The LayoutInflater object that can be used to inflate any views in the
     *      fragment.
     * @param container If non-null, this is the parent view that the fragment's UI should be
     *      attached to. The fragment should not add the view itself, but this can be used to
     *      generate the `LayoutParams` of the view.
     * @param savedInstanceState If non-null, this fragment is being re-constructed from a previous
     *      saved state as given here.
     *
     * @return Return the View for the fragment's UI, or null.
     */
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View = FragmentMainBinding.inflate(inflater, container, false).also {
        _binding = it
    }.root

    /**
     * Called immediately after `onCreateView(LayoutInflater, ViewGroup, Bundle)` has returned, but
     * before any saved state has been restored in to the view. This gives subclasses a chance to
     * initialize themselves once they know their view hierarchy has been completely created. The
     * fragment's view hierarchy is not however attached to its parent at this point.
     *
     * @param view The View returned by `onCreateView(LayoutInflater, ViewGroup, Bundle)`.
     * @param savedInstanceState If non-null, this fragment is being re-constructed from a previous
     *      saved state as given here.
     */
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Request camera permissions.
        if (checkPermission()) startCamera()
        else requestPermissionLauncher.launch(Manifest.permission.CAMERA)

        // The current selected camera.
        viewModel.cameraSelector.observe(viewLifecycleOwner) {
            cameraSelector = it ?: CameraSelector.DEFAULT_BACK_CAMERA
        }

        // Set the UI components behavior.
        binding.apply {
            btnCpu.setOnClickListener {
                selectedRuntimeChar = 'C'
                loadSelectedModel()
            }

            btnGpu.setOnClickListener {
                selectedRuntimeChar = 'G'
                loadSelectedModel()
            }

            btnDsp.setOnClickListener {
                selectedRuntimeChar = 'D'
                loadSelectedModel()
            }

            buttonCameraSwitch.setOnClickListener {
                viewModel.updateCameraSelector(toggleCameraSelector(cameraSelector))
                cameraSelector = toggleCameraSelector(cameraSelector)
                startCamera()
            }

            confidenceSlider.addOnChangeListener { _, value, _ ->
                confidenceThreshold = value
                textThreshold.text = getString(R.string.threshold_text, value)
            }

            // Model selection dropdown
            val models = resources.getStringArray(R.array.model_array)
            val adapter = ArrayAdapter(requireContext(), android.R.layout.simple_dropdown_item_1line, models)
            modelSelectorItem.setAdapter(adapter)
            modelSelectorItem.setText(models[2], false)

            modelSelectorItem.setOnItemClickListener { _, _, position, _ ->
                selectedClassifierModel = when (position) {
                    0 -> "classifier_mediapipe_gru_int8.dlc"
                    1 -> "classifier_mediapipe_tcn_int8.dlc"
                    2 -> "classifier_mediapipe_tcn_fp32.dlc"
                    3 -> "classifier_mediapipe_tcn_int8.tflite"
                    4 -> "classifier_mediapipe_gru_int8.tflite"
                    5 -> "classifier_mediapipe_lstm_int8.tflite"
                    else -> "classifier_mediapipe_tcn_fp32.dlc"
                }
                loadSelectedModel()
            }
        }

        // Load the default model
        loadSelectedModel()
    }

    /**
     * Loads the 2-stage gesture recognition model.
     * CHANGED: Follows SNPEHelper.loadModel() pattern but for TwoStageGestureHelper
     */
    private fun loadSelectedModel() {
        // FP32 DLC models don't work on DSP - force CPU
        // TFLite models CAN use DSP via QNN delegate
        if (selectedClassifierModel.contains("fp32") &&
            selectedClassifierModel.endsWith(".dlc") &&
            selectedRuntimeChar == 'D') {
            selectedRuntimeChar = 'C'
            binding.processorToggleGroup.check(R.id.btn_cpu)
            showSnackBar(getString(R.string.model_selector_error))
        }

        lifecycleScope.launch {
            binding.progressBar.isVisible = true
            isNetworkLoaded = false

            val success = withContext(Dispatchers.Default) {
                gestureHelper = TwoStageGestureHelper(requireActivity().application)
                gestureHelper.loadModels(selectedRuntimeChar, selectedClassifierModel)
            }

            binding.progressBar.isVisible = false
            isNetworkLoaded = success

            if (success) {
                val runtimeName = when (selectedRuntimeChar) {
                    'G' -> "GPU"
                    'D' -> "DSP"
                    else -> "CPU"
                }
                val modelName = when {
                    selectedClassifierModel.contains("gru") && selectedClassifierModel.endsWith(".dlc") -> "GRU_INT8 (DLC)"
                    selectedClassifierModel.contains("tcn_int8") -> "TCN_INT8 (DLC)"
                    selectedClassifierModel.contains("tcn_fp32") -> "TCN_FP32 (DLC)"
                    selectedClassifierModel.contains("tcn") && selectedClassifierModel.endsWith(".tflite") -> "TCN INT 8 (TFLite)"
                    selectedClassifierModel.contains("gru") && selectedClassifierModel.endsWith(".tflite") -> "GRU INT 8 (TFLite)"
                    selectedClassifierModel.contains("lstm") && selectedClassifierModel.endsWith(".tflite") -> "LSTM INT 8 (TFLite)"
                    else -> "Unknown"
                }
                showSnackBar("2-Stage gesture system loaded on $runtimeName ($modelName)")
            } else {
                showSnackBar("Failed to load gesture recognition models")
            }
        }
    }

    /**
     * Toggles between front and back camera selectors.
     *
     * @param current The current [CameraSelector] being used.
     *
     * @return The opposite [CameraSelector] to switch the camera.
     */
    private fun toggleCameraSelector(current: CameraSelector) = if (current == CameraSelector.DEFAULT_FRONT_CAMERA)
        CameraSelector.DEFAULT_BACK_CAMERA
    else
        CameraSelector.DEFAULT_FRONT_CAMERA

    /**
     * Called when the fragment is visible to the user and actively running. This is generally tied
     * to `Activity.onResume()` of the containing Activity's lifecycle.
     */
    override fun onResume() {
        super.onResume()
        fpsTracker.reset()
    }

    /**
     * Called when the Fragment is no longer resumed. This is generally tied to `Activity.onPause()`
     * of the containing Activity's lifecycle.
     */
    override fun onPause() {
        super.onPause()
        fpsTracker.reset()
    }


    /**
     * Called when the view previously created by `onCreateView()` has been detached from the
     * fragment. The next time the fragment needs to be displayed, a new view will be created. This
     * is called after `onStop()` and before `onDestroy()`. It is called <em>regardless</em> of
     * whether `onCreateView()` returned a non-null view. Internally it is called after the view's
     * state has been saved but before it has been removed from its parent.
     */
    override fun onDestroyView() {
        super.onDestroyView()
        analysisExecutor.shutdown()
        if (isNetworkLoaded) {
            gestureHelper.dispose()
        }
        _binding = null
    }

    /**
     * Handles the result of camera permission request.
     *
     * @param isGranted A boolean indicating whether the camera permission is granted or not. If
     *      true, starts the camera. If false, finishes the activity.
     */
    private fun cameraPermissionResult(isGranted: Boolean) {
        // Use the takeIf function to conditionally execute code based on the permission result
        isGranted.takeIf { it }?.run {
            startCamera()
        } ?: requireActivity().finish()
    }

    /**
     * This method checks if the user allows the application uses the camera to take photos for this
     * application.
     *
     * @return A boolean value with the user permission agreement.
     */
    private fun checkPermission() =
        ActivityCompat.checkSelfPermission(
            requireContext(), Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

    /**
     * Enabled or disabled a button to switch cameras depending on the available cameras.
     */
    private fun updateCameraSwitchButton(provider: ProcessCameraProvider) {
        binding.buttonCameraSwitch.isVisible = try {
            hasBackCamera(provider) && hasFrontCamera(provider)
        } catch (_: CameraInfoUnavailableException) {
            false
        }
    }

    /**
     * Returns true if the device has an available back camera. False otherwise.
     */
    private fun hasBackCamera(provider: ProcessCameraProvider) =
        provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)

    /**
     * Returns true if the device has an available front camera. False otherwise.
     */
    private fun hasFrontCamera(provider: ProcessCameraProvider) =
        provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)

    /**
     * This method is used to start the video camera device stream.
     * UNCHANGED from original - follows exact InferSNPE pattern
     */
    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun startCamera() {

        // Create an instance of the `ProcessCameraProvider` to bind the lifecycle of cameras to the
        // lifecycle owner.
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        // Add a listener to the `cameraProviderFuture`.
        cameraProviderFuture.addListener({

            // Used to bind the lifecycle of cameras to the lifecycle owner.
            val cameraProvider = cameraProviderFuture.get()

            val targetResolution = Size(1920, 1080)

            val targetRotation = binding.previewView.display?.rotation ?: Surface.ROTATION_0

            val resolutionStrategy = ResolutionStrategy(
                targetResolution,
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER
            )

            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(resolutionStrategy)
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setAllowedResolutionMode(ResolutionSelector.PREFER_HIGHER_RESOLUTION_OVER_CAPTURE_RATE)
                .build()

            // Video camera streaming preview.
            val preview = Preview.Builder()
                .setTargetRotation(targetRotation)
                .setResolutionSelector(resolutionSelector)
                .build().also {
                    it.surfaceProvider = binding.previewView.surfaceProvider
                }

            // Set up the image capture by getting a reference to the `ImageCapture`.
            imageCapture = ImageCapture.Builder().build()

            // Set up the image analysis use case.
            // CRITICAL: Force 30 FPS to match training data!
            val imageAnalysisBuilder = ImageAnalysis.Builder()
                .setTargetRotation(targetRotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(resolutionSelector)

            val extender = Camera2Interop.Extender(imageAnalysisBuilder)
            extender.setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                Range(30, 30)
            )

            val imageAnalyzer = imageAnalysisBuilder.build()
                .also {
                    it.setAnalyzer(analysisExecutor, ::processImageFrame)
                }

            try {
                // Unbind use cases before rebinding.
                cameraProvider.unbindAll()

                // Bind use cases to camera.
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )

                // Update the camera switch button.
                updateCameraSwitchButton(cameraProvider)

            } catch(ex: Exception) {
                showSnackBar("Use case binding failed: $ex")
            }

        }, ContextCompat.getMainExecutor(requireContext()))
    }

    /**
     * Processes a single image frame from the camera.
     * CHANGED: Added timestampMs for MediaPipe, changed method call
     *
     * @param imageProxy The image frame to be processed.
     */
    private fun processImageFrame(imageProxy: ImageProxy) {

        // Perform inference on the Bitmap.
        if (isNetworkLoaded) {

            // Convert the image to a Bitmap.
            val bitmap = imageProxy.toBitmapConverter()

            // Update the FPS tracker.
            fpsTracker.frameProcessed()

            // Get timestamp for MediaPipe (required for video mode)
            val timestampMs = SystemClock.uptimeMillis()

            // Perform 2-stage inference
            // CHANGED: Added timestampMs parameter
            val results = gestureHelper.inference(bitmap, timestampMs, confidenceThreshold)
            updateUI(results, bitmap.width, bitmap.height, fpsTracker.fps)
        }
        else {
            fpsTracker.reset()
            updateUI(emptyList(), 480, 640, null)
        }

        // Important to avoid memory leaks.
        imageProxy.close()
    }

    /**
     * Updates the detection overlay and FPS display on the main UI thread.
     * UNCHANGED from original - OverlayView still uses label/confidence
     *
     * This method sets the overlay view's dimensions based on the first detection result, triggers
     * a redraw with the provided list of detections, and updates the FPS text view.
     *
     * @param results A list of [DetectionResult] objects to render on the overlay.
     * @param width The width of the source image used for scaling bounding boxes.
     * @param height The height of the source image used for scaling bounding boxes.
     * @param fps The current frames-per-second value to display, or `null` if not available.
     */
    private fun updateUI(results: List<DetectionResult>, width: Int, height: Int, fps: Double?) {
        requireActivity().runOnUiThread {
            binding.overlayView.apply {
                imageWidth = width
                imageHeight = height
                isFrontCamera = cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA
                updateDetections(results)
            }
            fps?.let {
                if (isNetworkLoaded) binding.fpsText.text = getString(R.string.fps_text, it)
            }
        }
    }

    /**
     * Displays a SnackBar to show a brief message about the clicked button.
     *
     * The SnackBar is created using the clicked button's information and is shown at the bottom of
     * the screen.
     *
     * @param message The message to be displayed in the SnackBar.
     */
    private fun showSnackBar(message: String) {
        Snackbar.make(
            binding.root, message, Snackbar.LENGTH_SHORT
        ).show()
    }

}