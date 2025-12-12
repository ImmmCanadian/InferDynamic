# InferDynamic App

An Android application for real-time two-stage camera-based inference system using Qualcomm's SNPE (Snapdragon Neural Processing Engine) and TFLITE QNN Delegation to take advantage of DSP. The app demonstrates on-device AI model execution with support for CPU, GPU, and DSP acceleration modes (GPU does not work for TFLITE). This is a research-oriented optimization toolkit originally created and designed for experiments and development for the tutorial `Edge AI in Action: Technologies and Applications` presented during the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2025 (CVPR 2025).

This is a heavily modified version of the original InferSNPE and InferLITE created by the original authors Fabricio Batista Narcizo, Elizabete Munzlinger, Sai Narsi Reddy Donthi Reddy, and Shan Ahmed Shaffi for my thesis work with Fabricio. All credit to them.

## Project Structure

- `app/` - Main Android application module
  - `src/main/java/` - Application source code
  - `src/main/res/` - Application resources (layouts, drawables, etc.)
  - `src/main/assets/` - App assets
  - `src/main/AndroidManifest.xml` - App manifest
- `build.gradle.kts` - Project-level Gradle build script
- `settings.gradle.kts` - Gradle settings

## Features

- Real-time camera preview and inference
- Model selection and hardware acceleration (CPU, GPU, DSP)
- FPS display and annotated output
- Modern Android architecture with Kotlin, ViewBinding, and CameraX

## Requirements

- Android device with ARM64 architecture (arm64-v8a)
- Android 8.0 (API 26) or higher
- Qualcomm SNPE SDK (see below)

## Setup

1. **Trained Models**: Place your `.dlc` model files in `app/src/main/assets/` if you wish to add any.

## Build Instructions

1. Clone this repository.
2. Open in Android Studio (Arctic Fox or newer recommended).
3. Ensure you have the required SDKs and NDK installed.
4. Build and run on a physical ARM64 device or on an emulator on Android Studio.

Alternatively, from the command line:

```sh
./gradlew assembleDebug
```

## Usage

- Launch the app on your device.
- Grant camera permissions. For Instrumented testing grant file permissions as well.
- Select the desired model and hardware mode (CPU/GPU/DSP).
- View real-time inference results and FPS overlay.

## License

This project is licensed under the terms of the LICENSE file in this repository.
