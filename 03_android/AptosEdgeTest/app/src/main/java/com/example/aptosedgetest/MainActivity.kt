package com.example.aptosedgetest

import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Debug
import android.util.Log
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private val tag = "DR_TEST"
    private val defaultModelAssetName = "aptos_fixedsplit_efficientnet_lite0_float32.tflite"
    private val defaultModelName = "efficientnet_lite0"
    private val modelIncludesImageNetNormalization = true
    private val warmUpRuns = 5
    private val tfliteThreads = 4
    private val cropThreshold = 15.0

    private val imagenetMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val imagenetStd = floatArrayOf(0.229f, 0.224f, 0.225f)

    private val classNames = listOf(
        "0_No_DR",
        "1_Mild",
        "2_Moderate",
        "3_Severe",
        "4_Proliferative_DR"
    )

    private val imageExtensions = setOf("png", "jpg", "jpeg")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { view, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        if (!OpenCVLoader.initDebug()) {
            Log.e(tag, "OpenCV initialization failed.")
            updateStatus("OpenCV initialization failed. See Logcat DR_TEST.")
            return
        }

        Log.i(tag, "OpenCV initialized successfully.")

        val modelConfig = readModelConfigFromIntent()
        Thread {
            runLatencyTest(modelConfig)
        }.start()
    }

    private fun runLatencyTest(modelConfig: ModelConfig) {
        Log.i(tag, "Starting diabetic retinopathy end-to-end OpenCV + TFLite latency test")
        Log.i(tag, "Model name: ${modelConfig.name}")
        Log.i(tag, "Model asset: ${modelConfig.assetName}")
        Log.i(tag, "Android ImageNet mean/std normalization enabled: ${!modelIncludesImageNetNormalization}")
        Log.i(tag, "OpenCV preprocessing: crop threshold=$cropThreshold, resize=224x224, CLAHE clipLimit=2.0, tileGrid=8x8")

        val memoryBeforeKb = readMemoryPssKb()
        Log.i(tag, "Approx memory before test: ${memoryBeforeKb} KB PSS")

        val interpreter = try {
            createInterpreter(modelConfig.assetName)
        } catch (exception: Exception) {
            Log.e(tag, "Could not load model from assets/${modelConfig.assetName}", exception)
            updateStatus("Model load failed. See Logcat DR_TEST.")
            return
        }

        interpreter.use { tflite ->
            val inputInfo = readInputInfo(tflite)
            val outputInfo = readOutputInfo(tflite)
            logTensorInfo(inputInfo, outputInfo)

            val datasetRoot = findDatasetRoot()
            if (datasetRoot == null) {
                Log.e(tag, "No dataset folder found. Expected assets/val100 or assets/APTOS_val100_original")
                updateStatus("Dataset folder not found. See Logcat DR_TEST.")
                return
            }

            val testImages = listValidationImages(datasetRoot)
            Log.i(tag, "Dataset root: assets/$datasetRoot")
            Log.i(tag, "Total supported raw images found: ${testImages.size}")

            if (testImages.size != 100) {
                Log.w(tag, "Expected 100 images, but found ${testImages.size}. The test will run on all found images.")
            }

            if (testImages.isEmpty()) {
                Log.e(tag, "No PNG/JPG/JPEG images found under assets/$datasetRoot")
                updateStatus("No test images found. See Logcat DR_TEST.")
                return
            }

            val debugDir = File(getExternalFilesDir(null), "debug_preprocessed")
            debugDir.mkdirs()
            debugDir.listFiles()?.forEach { file -> file.delete() }
            Log.i(tag, "Debug preprocessed images directory: ${debugDir.absolutePath}")

            runWarmUps(tflite, inputInfo, outputInfo, testImages.first())

            val outputBuffer = ByteBuffer.allocateDirect(outputInfo.numBytes).order(ByteOrder.nativeOrder())
            val results = mutableListOf<ImageResult>()

            val preprocessTimesMs = mutableListOf<Double>()
            val inferenceTimesMs = mutableListOf<Double>()
            val endToEndTimesMs = mutableListOf<Double>()
            val debugSavedPerClass = mutableMapOf<Int, Int>()

            for ((zeroBasedIndex, testImage) in testImages.withIndex()) {
                val index = zeroBasedIndex + 1
                val endToEndStartNs = System.nanoTime()

                var rawBgr: Mat? = null
                var processedBgr: Mat? = null

                try {
                    val imageLoadStartNs = System.nanoTime()
                    rawBgr = loadRawImageAsBgrMat(testImage.assetPath)
                    val imageLoadMs = nanosToMillis(System.nanoTime() - imageLoadStartNs)

                    val preprocessStartNs = System.nanoTime()
                    processedBgr = preprocessFundusBgr(rawBgr, inputInfo.width, inputInfo.height)
                    val opencvPreprocessMs = nanosToMillis(System.nanoTime() - preprocessStartNs)

                    val truthLabel = numericLabel(testImage.groundTruthFolder)
                    val debugCountForClass = debugSavedPerClass[truthLabel] ?: 0
                    if (debugCountForClass < 2) {
                        saveDebugPreprocessedImage(debugDir, index, testImage, processedBgr)
                        debugSavedPerClass[truthLabel] = debugCountForClass + 1
                    }

                    val inputConversionStartNs = System.nanoTime()
                    val inputBuffer = convertBgrMatToInputBuffer(processedBgr, inputInfo)
                    val inputConversionMs = nanosToMillis(System.nanoTime() - inputConversionStartNs)

                    val inferenceMs = runInference(tflite, inputBuffer, outputBuffer)

                    val outputDecodeStartNs = System.nanoTime()
                    val predictedClass = decodePrediction(outputBuffer, outputInfo)
                    val outputDecodeMs = nanosToMillis(System.nanoTime() - outputDecodeStartNs)

                    val endToEndMs = nanosToMillis(System.nanoTime() - endToEndStartNs)

                    preprocessTimesMs.add(opencvPreprocessMs)
                    inferenceTimesMs.add(inferenceMs)
                    endToEndTimesMs.add(endToEndMs)

                    results.add(
                        ImageResult(
                            index = index,
                            imagePath = testImage.assetPath,
                            truth = testImage.groundTruthFolder,
                            predicted = predictedClass,
                            imageLoadMs = imageLoadMs,
                            opencvPreprocessMs = opencvPreprocessMs,
                            inputConversionMs = inputConversionMs,
                            tfliteInferenceMs = inferenceMs,
                            outputDecodeMs = outputDecodeMs,
                            endToEndMs = endToEndMs
                        )
                    )

                    Log.i(
                        tag,
                        "image=$index/${testImages.size}, path=${testImage.assetPath}, " +
                            "truth=${testImage.groundTruthFolder}, predicted=$predictedClass, " +
                            "image_load_ms=${formatMs(imageLoadMs)}, " +
                            "opencv_preprocess_ms=${formatMs(opencvPreprocessMs)}, " +
                            "input_conversion_ms=${formatMs(inputConversionMs)}, " +
                            "tflite_inference_ms=${formatMs(inferenceMs)}, " +
                            "output_decode_ms=${formatMs(outputDecodeMs)}, " +
                            "end_to_end_ms=${formatMs(endToEndMs)}"
                    )
                } catch (exception: Exception) {
                    Log.w(tag, "Skipping image because processing failed: ${testImage.assetPath}", exception)
                } finally {
                    rawBgr?.release()
                    processedBgr?.release()
                }
            }

            val csvFile = writeCsv(results, modelConfig)
            Log.i(tag, "CSV results saved to: ${csvFile.absolutePath}")

            logAccuracySummary(modelConfig, results)

            logSummary("OPENCV_PREPROCESS", preprocessTimesMs)
            logSummary("TFLITE_INFERENCE", inferenceTimesMs)
            logSummary("END_TO_END", endToEndTimesMs)

            val memoryAfterKb = readMemoryPssKb()
            Log.i(tag, "Approx memory after test: ${memoryAfterKb} KB PSS")
            Log.i(tag, "Approx memory change: ${memoryAfterKb - memoryBeforeKb} KB PSS")
            updateStatus("Done. CSV saved:\n${csvFile.absolutePath}")
        }
    }

    private fun readModelConfigFromIntent(): ModelConfig {
        val assetName = intent?.getStringExtra("model_asset") ?: defaultModelAssetName
        val modelName = intent?.getStringExtra("model_name")
            ?: assetName.removeSuffix(".tflite").removePrefix("aptos_fixedsplit_").removeSuffix("_float32")
        return ModelConfig(name = modelName, assetName = assetName)
    }
    private fun createInterpreter(modelAssetName: String): Interpreter {
        val options = Interpreter.Options().apply {
            setNumThreads(tfliteThreads)
            setUseXNNPACK(true)
        }

        Log.i(tag, "TensorFlow Lite options: numThreads=$tfliteThreads, XNNPACK=true")
        return Interpreter(loadModelBuffer(modelAssetName), options)
    }

    private fun loadModelBuffer(assetName: String): ByteBuffer {
        return try {
            assets.openFd(assetName).use { descriptor ->
                FileInputStream(descriptor.fileDescriptor).use { inputStream ->
                    inputStream.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        descriptor.startOffset,
                        descriptor.declaredLength
                    )
                }
            }
        } catch (exception: IOException) {
            Log.w(tag, "Memory-map failed; falling back to direct ByteBuffer model load.", exception)
            val modelBytes = assets.open(assetName).use { it.readBytes() }
            ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder()).apply {
                put(modelBytes)
                rewind()
            }
        }
    }

    private fun readInputInfo(interpreter: Interpreter): TensorInfo {
        val tensor = interpreter.getInputTensor(0)
        val shape = tensor.shape()
        val quantization = tensor.quantizationParams()
        val layout = parseInputLayout(shape)

        return TensorInfo(
            shape = shape,
            dataType = tensor.dataType(),
            scale = quantization.scale,
            zeroPoint = quantization.zeroPoint,
            numBytes = tensor.numBytes(),
            height = layout.height,
            width = layout.width,
            channels = layout.channels,
            channelsLast = layout.channelsLast
        )
    }

    private fun readOutputInfo(interpreter: Interpreter): TensorInfo {
        val tensor = interpreter.getOutputTensor(0)
        val shape = tensor.shape()
        val quantization = tensor.quantizationParams()

        return TensorInfo(
            shape = shape,
            dataType = tensor.dataType(),
            scale = quantization.scale,
            zeroPoint = quantization.zeroPoint,
            numBytes = tensor.numBytes(),
            height = 0,
            width = 0,
            channels = shape.fold(1) { product, value -> product * value },
            channelsLast = true
        )
    }

    private fun parseInputLayout(shape: IntArray): InputLayout {
        if (shape.size != 4) {
            Log.w(tag, "Expected 4D input tensor, got ${shape.contentToString()}. Falling back to 224x224x3.")
            return InputLayout(height = 224, width = 224, channels = 3, channelsLast = true)
        }

        if (shape[0] != 1) {
            Log.w(tag, "Expected batch size 1, got input shape ${shape.contentToString()}.")
        }

        return when {
            shape[3] == 3 -> InputLayout(height = shape[1], width = shape[2], channels = shape[3], channelsLast = true)
            shape[1] == 3 -> InputLayout(height = shape[2], width = shape[3], channels = shape[1], channelsLast = false)
            else -> {
                Log.w(tag, "Could not clearly detect RGB channel dimension from ${shape.contentToString()}. Assuming NHWC.")
                InputLayout(height = shape[1], width = shape[2], channels = shape[3], channelsLast = true)
            }
        }
    }

    private fun logTensorInfo(inputInfo: TensorInfo, outputInfo: TensorInfo) {
        Log.i(tag, "Input shape: ${inputInfo.shape.contentToString()}")
        Log.i(tag, "Input data type: ${inputInfo.dataType}")
        Log.i(tag, "Input quantization scale: ${inputInfo.scale}")
        Log.i(tag, "Input quantization zero point: ${inputInfo.zeroPoint}")
        Log.i(tag, "Input target: ${inputInfo.width}x${inputInfo.height}, channels=${inputInfo.channels}")

        Log.i(tag, "Output shape: ${outputInfo.shape.contentToString()}")
        Log.i(tag, "Output data type: ${outputInfo.dataType}")
        Log.i(tag, "Output quantization scale: ${outputInfo.scale}")
        Log.i(tag, "Output quantization zero point: ${outputInfo.zeroPoint}")

        val outputElementCount = outputInfo.shape.fold(1) { product, value -> product * value }
        if (outputElementCount != classNames.size) {
            Log.w(tag, "Output tensor has $outputElementCount values, but this test expects ${classNames.size} DR classes.")
        }
    }

    private fun findDatasetRoot(): String? {
        val candidates = listOf("val100", "APTOS_val100_original")
        return candidates.firstOrNull { candidate ->
            !assets.list(candidate).isNullOrEmpty()
        }
    }

    private fun listValidationImages(datasetRoot: String): List<TestImage> {
        val images = mutableListOf<TestImage>()

        for (className in classNames) {
            val classFolder = "$datasetRoot/$className"
            val files = assets.list(classFolder)

            if (files == null) {
                Log.w(tag, "Class folder is missing or unreadable: assets/$classFolder")
                continue
            }

            val classImages = files
                .filter { fileName -> isSupportedImage(fileName) }
                .sorted()
                .map { fileName ->
                    TestImage(
                        assetPath = "$classFolder/$fileName",
                        groundTruthFolder = className
                    )
                }

            Log.i(tag, "Found ${classImages.size} supported raw images in assets/$classFolder")
            images.addAll(classImages)
        }

        return images
    }

    private fun isSupportedImage(fileName: String): Boolean {
        val extension = fileName.substringAfterLast('.', missingDelimiterValue = "").lowercase()
        return extension in imageExtensions
    }

    private fun runWarmUps(
        interpreter: Interpreter,
        inputInfo: TensorInfo,
        outputInfo: TensorInfo,
        warmUpImage: TestImage
    ) {
        Log.i(tag, "Running $warmUpRuns warm-up inferences using ${warmUpImage.assetPath}")
        val rawBgr = loadRawImageAsBgrMat(warmUpImage.assetPath)
        val processedBgr = preprocessFundusBgr(rawBgr, inputInfo.width, inputInfo.height)
        val inputBuffer = convertBgrMatToInputBuffer(processedBgr, inputInfo)
        val outputBuffer = ByteBuffer.allocateDirect(outputInfo.numBytes).order(ByteOrder.nativeOrder())

        rawBgr.release()
        processedBgr.release()

        repeat(warmUpRuns) { warmUpIndex ->
            val inferenceMs = runInference(interpreter, inputBuffer, outputBuffer)
            Log.i(tag, "Warm-up ${warmUpIndex + 1}/$warmUpRuns tflite_inference_ms=${formatMs(inferenceMs)}")
        }
    }

    private fun loadRawImageAsBgrMat(assetPath: String): Mat {
        val bitmap = assets.open(assetPath).use { inputStream ->
            BitmapFactory.decodeStream(inputStream)
        } ?: error("BitmapFactory returned null for $assetPath")

        val rgba = Mat()
        val bgr = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        bitmap.recycle()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }

    private fun preprocessFundusBgr(inputBgr: Mat, targetWidth: Int, targetHeight: Int): Mat {
        val gray = Mat()
        val threshold = Mat()
        val hierarchy = Mat()
        val contours = mutableListOf<MatOfPoint>()
        var cropped: Mat? = null

        return try {
            Imgproc.cvtColor(inputBgr, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.threshold(gray, threshold, cropThreshold, 255.0, Imgproc.THRESH_BINARY)
            Imgproc.findContours(threshold, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            cropped = if (contours.isEmpty()) {
                Log.w(tag, "No contour found during preprocessing; using full image.")
                inputBgr.clone()
            } else {
                val largestContour = contours.maxByOrNull { contour -> Imgproc.contourArea(contour) }
                val rect = Imgproc.boundingRect(largestContour)
                val centerX = rect.x + rect.width / 2
                val centerY = rect.y + rect.height / 2
                val side = maxOf(rect.width, rect.height)
                val x1 = maxOf(0, centerX - side / 2)
                val y1 = maxOf(0, centerY - side / 2)
                val x2 = minOf(inputBgr.cols(), centerX + side / 2)
                val y2 = minOf(inputBgr.rows(), centerY + side / 2)

                if (x2 > x1 && y2 > y1) {
                    Mat(inputBgr, Rect(x1, y1, x2 - x1, y2 - y1)).clone()
                } else {
                    Log.w(tag, "Invalid crop rectangle during preprocessing; using full image.")
                    inputBgr.clone()
                }
            }

            val resized = Mat()
            Imgproc.resize(
                cropped,
                resized,
                Size(targetWidth.toDouble(), targetHeight.toDouble()),
                0.0,
                0.0,
                Imgproc.INTER_AREA
            )

            val lab = Mat()
            Imgproc.cvtColor(resized, lab, Imgproc.COLOR_BGR2Lab)
            val labChannels = mutableListOf<Mat>()
            Core.split(lab, labChannels)

            val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            val enhancedL = Mat()
            clahe.apply(labChannels[0], enhancedL)
            labChannels[0].release()
            labChannels[0] = enhancedL

            val mergedLab = Mat()
            Core.merge(labChannels, mergedLab)

            val processedBgr = Mat()
            Imgproc.cvtColor(mergedLab, processedBgr, Imgproc.COLOR_Lab2BGR)

            resized.release()
            lab.release()
            mergedLab.release()
            labChannels.forEach { channel -> channel.release() }
            processedBgr
        } finally {
            gray.release()
            threshold.release()
            hierarchy.release()
            contours.forEach { contour -> contour.release() }
            cropped?.release()
        }
    }

    private fun saveDebugPreprocessedImage(debugDir: File, index: Int, testImage: TestImage, processedBgr: Mat) {
        val fileName = testImage.assetPath.substringAfterLast("/")
        val stem = fileName.substringBeforeLast(".")
        val outputFile = File(debugDir, "${index.toString().padStart(3, '0')}_${testImage.groundTruthFolder}_$stem.png")
        val saved = Imgcodecs.imwrite(outputFile.absolutePath, processedBgr)
        if (saved) {
            Log.i(tag, "Saved debug preprocessed image: ${outputFile.absolutePath}")
        } else {
            Log.w(tag, "Failed to save debug preprocessed image: ${outputFile.absolutePath}")
        }
    }

    private fun convertBgrMatToInputBuffer(processedBgr: Mat, inputInfo: TensorInfo): ByteBuffer {
        require(processedBgr.cols() == inputInfo.width && processedBgr.rows() == inputInfo.height) {
            "Preprocessed image is ${processedBgr.cols()}x${processedBgr.rows()}, expected ${inputInfo.width}x${inputInfo.height}"
        }
        require(processedBgr.channels() == 3) {
            "Preprocessed image has ${processedBgr.channels()} channels, expected 3"
        }

        val inputBuffer = ByteBuffer.allocateDirect(inputInfo.numBytes).order(ByteOrder.nativeOrder())
        val bgrBytes = ByteArray(processedBgr.rows() * processedBgr.cols() * processedBgr.channels())
        processedBgr.get(0, 0, bgrBytes)

        if (inputInfo.channelsLast) {
            var byteIndex = 0
            while (byteIndex < bgrBytes.size) {
                putChannelValue(inputBuffer, 0, bgrBytes[byteIndex + 2].toInt() and 0xFF, inputInfo)
                putChannelValue(inputBuffer, 1, bgrBytes[byteIndex + 1].toInt() and 0xFF, inputInfo)
                putChannelValue(inputBuffer, 2, bgrBytes[byteIndex].toInt() and 0xFF, inputInfo)
                byteIndex += 3
            }
        } else {
            for (channelIndex in 0 until inputInfo.channels) {
                var byteIndex = 0
                while (byteIndex < bgrBytes.size) {
                    val channelValue = when (channelIndex) {
                        0 -> bgrBytes[byteIndex + 2].toInt() and 0xFF
                        1 -> bgrBytes[byteIndex + 1].toInt() and 0xFF
                        2 -> bgrBytes[byteIndex].toInt() and 0xFF
                        else -> 0
                    }
                    putChannelValue(inputBuffer, channelIndex, channelValue, inputInfo)
                    byteIndex += 3
                }
            }
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    private fun putChannelValue(
        inputBuffer: ByteBuffer,
        channelIndex: Int,
        channelValue: Int,
        inputInfo: TensorInfo
    ) {
        val clamped = channelValue.coerceIn(0, 255)
        when (inputInfo.dataType) {
            DataType.FLOAT32 -> {
                val value01 = clamped / 255.0f
                val modelInputValue = if (modelIncludesImageNetNormalization) {
                    value01
                } else {
                    (value01 - imagenetMean[channelIndex.coerceIn(0, 2)]) / imagenetStd[channelIndex.coerceIn(0, 2)]
                }
                inputBuffer.putFloat(modelInputValue)
            }
            DataType.UINT8 -> inputBuffer.put(clamped.toByte())
            DataType.INT8 -> inputBuffer.put(quantizeInt8(clamped, inputInfo).toByte())
            else -> error("Unsupported input tensor type: ${inputInfo.dataType}")
        }
    }

    private fun quantizeInt8(channelValue: Int, inputInfo: TensorInfo): Int {
        val quantized = quantizeImageChannel(channelValue, inputInfo)
        return quantized.coerceIn(-128, 127)
    }

    private fun quantizeImageChannel(channelValue: Int, inputInfo: TensorInfo): Int {
        if (inputInfo.scale <= 0.0f) {
            return channelValue + inputInfo.zeroPoint
        }

        val realValue = channelValue / 255.0f
        return (realValue / inputInfo.scale + inputInfo.zeroPoint).roundToInt()
    }

    private fun runInference(
        interpreter: Interpreter,
        inputBuffer: ByteBuffer,
        outputBuffer: ByteBuffer
    ): Double {
        inputBuffer.rewind()
        outputBuffer.clear()

        val startNs = System.nanoTime()
        interpreter.run(inputBuffer, outputBuffer)
        val elapsedNs = System.nanoTime() - startNs

        outputBuffer.rewind()
        return nanosToMillis(elapsedNs)
    }

    private fun decodePrediction(outputBuffer: ByteBuffer, outputInfo: TensorInfo): String {
        outputBuffer.rewind()

        val outputCount = outputInfo.shape.fold(1) { product, value -> product * value }
        val scores = FloatArray(outputCount)

        for (index in 0 until outputCount) {
            scores[index] = when (outputInfo.dataType) {
                DataType.FLOAT32 -> outputBuffer.getFloat()
                DataType.UINT8 -> dequantize(outputBuffer.get().toInt() and 0xFF, outputInfo)
                DataType.INT8 -> dequantize(outputBuffer.get().toInt(), outputInfo)
                DataType.INT32 -> outputBuffer.getInt().toFloat()
                DataType.INT64 -> outputBuffer.getLong().toFloat()
                else -> error("Unsupported output tensor type: ${outputInfo.dataType}")
            }
        }

        val bestIndex = scores.indices.maxByOrNull { scores[it] } ?: -1
        return classNames.getOrElse(bestIndex) { "class_$bestIndex" }
    }

    private fun dequantize(value: Int, outputInfo: TensorInfo): Float {
        return if (outputInfo.scale > 0.0f) {
            (value - outputInfo.zeroPoint) * outputInfo.scale
        } else {
            value.toFloat()
        }
    }

    private fun writeCsv(results: List<ImageResult>, modelConfig: ModelConfig): File {
        val outputFile = File(getExternalFilesDir(null), "dr_latency_results_${modelConfig.name}.csv")
        outputFile.parentFile?.mkdirs()

        outputFile.printWriter().use { writer ->
            writer.println(
                "index,image_path,truth,predicted,image_load_ms,opencv_preprocess_ms," +
                    "input_conversion_ms,tflite_inference_ms,output_decode_ms,end_to_end_ms"
            )
            for (result in results) {
                writer.println(
                    listOf(
                        result.index.toString(),
                        csv(result.imagePath),
                        csv(result.truth),
                        csv(result.predicted),
                        formatMs(result.imageLoadMs),
                        formatMs(result.opencvPreprocessMs),
                        formatMs(result.inputConversionMs),
                        formatMs(result.tfliteInferenceMs),
                        formatMs(result.outputDecodeMs),
                        formatMs(result.endToEndMs)
                    ).joinToString(",")
                )
            }
        }

        return outputFile
    }

    private fun csv(value: String): String {
        return "\"" + value.replace("\"", "\"\"") + "\""
    }

    private fun logAccuracySummary(modelConfig: ModelConfig, results: List<ImageResult>) {
        val numImages = results.size
        val totalCorrect = results.count { result -> numericLabel(result.truth) == numericLabel(result.predicted) }
        val accuracy = if (numImages > 0) totalCorrect.toDouble() / numImages else 0.0

        Log.i(tag, "ACCURACY summary")
        Log.i(tag, "model_name=${modelConfig.name}")
        Log.i(tag, "tflite_filename=${modelConfig.assetName}")
        Log.i(tag, "num_images=$numImages")
        Log.i(tag, "overall_accuracy_on_100_images=${String.format(Locale.US, "%.4f", accuracy)}")

        for (classIndex in classNames.indices) {
            val classResults = results.filter { result -> numericLabel(result.truth) == classIndex }
            val classCorrect = classResults.count { result -> numericLabel(result.predicted) == classIndex }
            Log.i(tag, "class${classIndex}_correct=$classCorrect class${classIndex}_total=${classResults.size}")
        }
    }

    private fun numericLabel(classFolder: String): Int {
        return classFolder.substringBefore("_").toIntOrNull() ?: -1
    }
    private fun logSummary(label: String, valuesMs: List<Double>) {
        if (valuesMs.isEmpty()) {
            Log.w(tag, "$label summary: no successful timings were recorded.")
            return
        }

        val sorted = valuesMs.sorted()
        val count = sorted.size
        val mean = sorted.average()
        val median = if (count % 2 == 0) {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        }
        val p95Index = (ceil(count * 0.95).toInt() - 1).coerceIn(0, count - 1)
        val p95 = sorted[p95Index]
        val min = sorted.first()
        val max = sorted.last()
        val fps = if (mean > 0.0) 1000.0 / mean else 0.0

        Log.i(tag, "$label summary")
        Log.i(tag, "count=$count")
        Log.i(tag, "mean_ms=${formatMs(mean)}")
        Log.i(tag, "median_ms=${formatMs(median)}")
        Log.i(tag, "p95_ms=${formatMs(p95)}")
        Log.i(tag, "min_ms=${formatMs(min)}")
        Log.i(tag, "max_ms=${formatMs(max)}")
        Log.i(tag, "fps=${String.format(Locale.US, "%.2f", fps)}")
    }

    private fun readMemoryPssKb(): Int {
        val memoryInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memoryInfo)
        return memoryInfo.totalPss
    }

    private fun updateStatus(message: String) {
        runOnUiThread {
            findViewById<TextView>(R.id.statusText).text = message
        }
    }

    private fun nanosToMillis(nanos: Long): Double {
        return nanos / 1_000_000.0
    }

    private fun formatMs(value: Double): String {
        return String.format(Locale.US, "%.3f", value)
    }

    data class ModelConfig(
        val name: String,
        val assetName: String
    )

    data class TestImage(
        val assetPath: String,
        val groundTruthFolder: String
    )

    data class ImageResult(
        val index: Int,
        val imagePath: String,
        val truth: String,
        val predicted: String,
        val imageLoadMs: Double,
        val opencvPreprocessMs: Double,
        val inputConversionMs: Double,
        val tfliteInferenceMs: Double,
        val outputDecodeMs: Double,
        val endToEndMs: Double
    )

    data class TensorInfo(
        val shape: IntArray,
        val dataType: DataType,
        val scale: Float,
        val zeroPoint: Int,
        val numBytes: Int,
        val height: Int,
        val width: Int,
        val channels: Int,
        val channelsLast: Boolean
    )

    data class InputLayout(
        val height: Int,
        val width: Int,
        val channels: Int,
        val channelsLast: Boolean
    )
}

