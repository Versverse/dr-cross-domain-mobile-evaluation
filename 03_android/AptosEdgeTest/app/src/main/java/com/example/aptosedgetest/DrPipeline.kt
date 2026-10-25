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

class DrPipeline(private val context: android.content.Context) {
    private val assets get() = context.assets
    private val tag = "DR_DEMO"
    private val modelIncludesImageNetNormalization = true
    private val tfliteThreads = 4
    private val cropThreshold = 15.0
    private val imagenetMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val imagenetStd = floatArrayOf(0.229f, 0.224f, 0.225f)
    private val classNames = listOf("0_No_DR", "1_Mild", "2_Moderate", "3_Severe", "4_Proliferative_DR")
    private val imageExtensions = setOf("png", "jpg", "jpeg")
    fun createInterpreter(modelAssetName: String): Interpreter {
        val options = Interpreter.Options().apply {
            setNumThreads(tfliteThreads)
            setUseXNNPACK(true)
        }

        Log.i(tag, "TensorFlow Lite options: numThreads=$tfliteThreads, XNNPACK=true")
        return Interpreter(loadModelBuffer(modelAssetName), options)
    }

    fun loadModelBuffer(assetName: String): ByteBuffer {
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

    fun readInputInfo(interpreter: Interpreter): TensorInfo {
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

    fun readOutputInfo(interpreter: Interpreter): TensorInfo {
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

    fun parseInputLayout(shape: IntArray): InputLayout {
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

    fun logTensorInfo(inputInfo: TensorInfo, outputInfo: TensorInfo) {
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

    fun findDatasetRoot(): String? {
        val candidates = listOf("val100", "APTOS_val100_original")
        return candidates.firstOrNull { candidate ->
            !assets.list(candidate).isNullOrEmpty()
        }
    }

    fun listValidationImages(datasetRoot: String): List<TestImage> {
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

    fun isSupportedImage(fileName: String): Boolean {
        val extension = fileName.substringAfterLast('.', missingDelimiterValue = "").lowercase()
        return extension in imageExtensions
    }

    fun loadRawImageAsBgrMat(assetPath: String): Mat {
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

    fun preprocessFundusBgr(inputBgr: Mat, targetWidth: Int, targetHeight: Int): Mat {
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

    fun convertBgrMatToInputBuffer(processedBgr: Mat, inputInfo: TensorInfo): ByteBuffer {
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

    fun putChannelValue(
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

    fun quantizeInt8(channelValue: Int, inputInfo: TensorInfo): Int {
        val quantized = quantizeImageChannel(channelValue, inputInfo)
        return quantized.coerceIn(-128, 127)
    }

    fun quantizeImageChannel(channelValue: Int, inputInfo: TensorInfo): Int {
        if (inputInfo.scale <= 0.0f) {
            return channelValue + inputInfo.zeroPoint
        }

        val realValue = channelValue / 255.0f
        return (realValue / inputInfo.scale + inputInfo.zeroPoint).roundToInt()
    }

    fun runInference(
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

    fun decodePrediction(outputBuffer: ByteBuffer, outputInfo: TensorInfo): String {
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

    fun dequantize(value: Int, outputInfo: TensorInfo): Float {
        return if (outputInfo.scale > 0.0f) {
            (value - outputInfo.zeroPoint) * outputInfo.scale
        } else {
            value.toFloat()
        }
    }

    fun numericLabel(classFolder: String): Int {
        return classFolder.substringBefore("_").toIntOrNull() ?: -1
    }

    fun readMemoryPssKb(): Int {
        val memoryInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memoryInfo)
        return memoryInfo.totalPss
    }

    fun nanosToMillis(nanos: Long): Double {
        return nanos / 1_000_000.0
    }

    fun formatMs(value: Double): String {
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
