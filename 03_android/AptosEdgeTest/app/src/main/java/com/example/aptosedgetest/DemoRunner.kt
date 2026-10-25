package com.example.aptosedgetest

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import kotlin.math.ceil

data class DemoModel(val title: String, val key: String, val asset: String) {
    override fun toString() = title
}

data class DemoProgress(val phase: String, val attempted: Int, val total: Int, val succeeded: Int, val failed: Int)
data class DemoResult(
    val directory: File, val status: String, val attempted: Int, val total: Int,
    val succeeded: Int, val failed: Int, val correct: Int, val last: DrPipeline.ImageResult?,
    val preview: Bitmap?, val summary: JSONObject, val error: String?
)

class DemoRunner(private val context: Context) {
    private val pipeline = DrPipeline(context)
    companion object {
        val MODELS = listOf(
            DemoModel("MobileNetV3-Large", "mobilenetv3_large", "aptos_fixedsplit_mobilenetv3_large_float32.tflite"),
            DemoModel("EfficientNet-Lite0", "efficientnet_lite0", "aptos_fixedsplit_efficientnet_lite0_float32.tflite")
        )
        val GRADES = listOf("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")
        const val TIMING = "Pipeline: asset decode -> OpenCV -> RGB tensor -> inference -> grade. Excludes model load, 5 warm-ups, preview rendering and file saving."
        fun grade(folder: String) = GRADES.getOrNull(folder.substringBefore('_').toIntOrNull() ?: -1) ?: "Unknown"
        fun digest(context: Context, asset: String): String {
            val hash = MessageDigest.getInstance("SHA-256")
            context.assets.open(asset).use { stream ->
                val bytes = ByteArray(65536)
                while (true) {
                    val count = stream.read(bytes)
                    if (count < 0) break
                    hash.update(bytes, 0, count)
                }
            }
            return hash.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
        }
    }

    fun images(): List<DrPipeline.TestImage> {
        val root = pipeline.findDatasetRoot() ?: error("Bundled test images were not found.")
        return pipeline.listValidationImages(root).also { require(it.isNotEmpty()) { "The sample set is empty." } }
    }

    fun run(model: DemoModel, selected: List<DrPipeline.TestImage>, warmupImage: DrPipeline.TestImage,
            batch: Boolean, cancel: () -> Boolean, progress: (DemoProgress) -> Unit): DemoResult {
        val runId = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date()) + "_" + UUID.randomUUID().toString().take(8)
        val base = context.getExternalFilesDir(null) ?: context.filesDir
        val dir = File(base, "demo_runs/${runId}_${model.key}_${if (batch) "batch" else "single"}")
        check(dir.mkdirs()) { "Could not create result directory: $dir" }
        val runLog = File(dir, "session_log.txt")
        fun log(value: String) { Log.i("DR_DEMO", "$runId $value"); runLog.appendText(value + "\n") }
        val rows = mutableListOf<DrPipeline.ImageResult>()
        val manifest = JSONArray()
        val errors = JSONArray()
        val exportWarnings = JSONArray()
        var attempted = 0
        var preview: Bitmap? = null
        var status = "failed"
        var fatal: String? = null
        val beforePss = pipeline.readMemoryPssKb()
        val wallStart = System.nanoTime()
        val summary = JSONObject().put("run_id", runId).put("model", model.title).put("tflite_filename", model.asset)
            .put("mode", if (batch) "batch" else "single").put("record_type", "CD_DEMO_NEW_MEASUREMENT")
            .put("device_manufacturer", Build.MANUFACTURER).put("device_model", Build.MODEL)
            .put("android_version", Build.VERSION.RELEASE).put("sdk", Build.VERSION.SDK_INT)
            .put("hardware", Build.HARDWARE).put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
            .put("soc_model", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "unavailable")
            .put("num_threads", 4).put("xnnpack_requested", true).put("warmup_runs", 5)
            .put("normalization", "RGB FLOAT32 [0,1]; ImageNet mean/std inside model only")
            .put("preprocessing", "Legacy functions unchanged: threshold 15, largest external contour, centered square crop clipped to image bounds, INTER_AREA 224x224, LAB L CLAHE 2.0/8x8")
            .put("pipeline_timing_boundary", TIMING).put("requested_images", selected.size)
            .put("warmup_image", warmupImage.assetPath).put("app_pss_before_kb", beforePss)
            .put("pss_note", "Application PSS includes UI; delta is not isolated model memory")
        File(dir, "summary.json").writeText(summary.put("status", "running").toString(2))
        log("START model=${model.asset} requested=${selected.size} mode=${if (batch) "batch" else "single"}")
        try {
            progress(DemoProgress("Loading model", 0, selected.size, 0, 0))
            summary.put("model_sha256", digest(context, model.asset))
            pipeline.createInterpreter(model.asset).use { interpreter ->
                val input = pipeline.readInputInfo(interpreter)
                val output = pipeline.readOutputInfo(interpreter)
                require(input.shape.contentEquals(intArrayOf(1, 224, 224, 3)) && input.dataType == DataType.FLOAT32) { "Unexpected model input: ${input.shape.contentToString()} ${input.dataType}" }
                require(output.channels == 5 && output.dataType == DataType.FLOAT32) { "Expected five FLOAT32 logits." }
                summary.put("input_shape", JSONArray(input.shape.toList())).put("output_shape", JSONArray(output.shape.toList()))
                val out = ByteBuffer.allocateDirect(output.numBytes).order(ByteOrder.nativeOrder())
                val warmRaw = pipeline.loadRawImageAsBgrMat(warmupImage.assetPath)
                val warmInput = try {
                    val processed = pipeline.preprocessFundusBgr(warmRaw, 224, 224)
                    try { pipeline.convertBgrMatToInputBuffer(processed, input) } finally { processed.release() }
                } finally { warmRaw.release() }
                repeat(5) {
                    progress(DemoProgress("Warming up ${it + 1} / 5", 0, selected.size, 0, 0))
                    log("warmup=${it + 1} inference_ms=${pipeline.runInference(interpreter, warmInput, out)}")
                }
                File(dir, "predictions.csv").bufferedWriter().use { writer ->
                    writer.write("index,image_path,dataset_label,predicted_grade,image_load_ms,opencv_preprocess_ms,input_conversion_ms,tflite_inference_ms,output_decode_ms,pipeline_e2e_ms,logit_0,logit_1,logit_2,logit_3,logit_4\n")
                    for ((index, sample) in selected.withIndex()) {
                        if (cancel()) break
                        var raw: Mat? = null
                        var processed: Mat? = null
                        progress(DemoProgress("Analyzing ${sample.assetPath.substringAfterLast('/')}", attempted, selected.size, rows.size, errors.length()))
                        try {
                            manifest.put(JSONObject().put("index", index + 1).put("asset_path", sample.assetPath)
                                .put("dataset_label", sample.groundTruthFolder).put("sha256", digest(context, sample.assetPath)))
                            val start = System.nanoTime()
                            raw = pipeline.loadRawImageAsBgrMat(sample.assetPath)
                            val loaded = System.nanoTime()
                            processed = pipeline.preprocessFundusBgr(raw, 224, 224)
                            val preprocessed = System.nanoTime()
                            val buffer = pipeline.convertBgrMatToInputBuffer(processed, input)
                            val converted = System.nanoTime()
                            val inferenceMs = pipeline.runInference(interpreter, buffer, out)
                            val decodeStart = System.nanoTime()
                            val predicted = pipeline.decodePrediction(out, output)
                            val end = System.nanoTime()
                            val row = DrPipeline.ImageResult(index + 1, sample.assetPath, sample.groundTruthFolder, predicted,
                                ms(loaded - start), ms(preprocessed - loaded), ms(converted - preprocessed), inferenceMs, ms(end - decodeStart), ms(end - start))
                            out.rewind()
                            val logits = List(5) { out.float }
                            require(logits.all { it.isFinite() }) { "Model returned non-finite logits." }
                            writer.write(listOf(row.index, quote(row.imagePath), quote(row.truth), quote(row.predicted), row.imageLoadMs,
                                row.opencvPreprocessMs, row.inputConversionMs, row.tfliteInferenceMs, row.outputDecodeMs, row.endToEndMs).joinToString(",") + "," + logits.joinToString(",") + "\n")
                            writer.flush()
                            rows.add(row)
                            log("image=${index + 1}/${selected.size} path=${sample.assetPath} truth=${row.truth} predicted=${row.predicted} inference_ms=${row.tfliteInferenceMs} pipeline_e2e_ms=${row.endToEndMs}")
                            // Export the exact processed pixels only after the measured pipeline ends.
                            if (!batch || index == selected.lastIndex) {
                              try {
                                val file = File(dir, "processed_${sample.assetPath.substringAfterLast('/').substringBeforeLast('.')}.png")
                                check(Imgcodecs.imwrite(file.absolutePath, processed)) { "Could not save processed image." }
                                val rgba = Mat()
                                try {
                                    Imgproc.cvtColor(processed, rgba, Imgproc.COLOR_BGR2RGBA)
                                    preview = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888).also { Utils.matToBitmap(rgba, it) }
                                } finally { rgba.release() }
                              } catch (error: Exception) {
                                  exportWarnings.put(JSONObject().put("image_path", sample.assetPath).put("error", error.toString()))
                                  log("EXPORT_WARNING image=${sample.assetPath} $error")
                              }
                            }
                        } catch (exception: Exception) {
                            errors.put(JSONObject().put("image_path", sample.assetPath).put("error", exception.toString()))
                            log("ERROR image=${sample.assetPath} $exception")
                        } finally { raw?.release(); processed?.release() }
                        attempted++
                        progress(DemoProgress("Processed $attempted / ${selected.size}", attempted, selected.size, rows.size, errors.length()))
                    }
                }
                status = when { attempted < selected.size -> "cancelled"; errors.length() > 0 -> "completed_with_errors"; else -> "complete" }
            }
        } catch (exception: Exception) { fatal = exception.toString(); log("FATAL $exception") }
        val correct = rows.count { it.truth == it.predicted }
        summary.put("status", status).put("attempted", attempted).put("succeeded", rows.size)
            .put("failed", errors.length()).put("correct", correct)
            .put("descriptive_sample_accuracy", if (rows.isNotEmpty()) correct.toDouble() / rows.size else JSONObject.NULL)
            .put("fatal_error", fatal ?: JSONObject.NULL).put("worker_elapsed_ms", ms(System.nanoTime() - wallStart))
            .put("app_pss_after_kb", pipeline.readMemoryPssKb())
            .put("preprocessing_ms", stats(rows.map { it.opencvPreprocessMs }))
            .put("inference_ms", stats(rows.map { it.tfliteInferenceMs }))
            .put("pipeline_e2e_ms", stats(rows.map { it.endToEndMs }))
        val perClass = JSONArray()
        for (i in 0..4) {
            val group = rows.filter { it.truth.startsWith("${i}_") }
            perClass.put(JSONObject().put("class", i).put("total_successful", group.size).put("correct", group.count { it.truth == it.predicted }))
        }
        summary.put("per_class", perClass)
        summary.put("export_warnings", exportWarnings)
        File(dir, "input_manifest.json").writeText(manifest.toString(2))
        File(dir, "errors.json").writeText(errors.toString(2))
        File(dir, "summary.json").writeText(summary.toString(2))
        log("DONE status=$status succeeded=${rows.size} failed=${errors.length()} correct=$correct directory=$dir")
        return DemoResult(dir, status, attempted, selected.size, rows.size, errors.length(), correct, rows.lastOrNull(), preview, summary, fatal)
    }

    private fun ms(ns: Long) = ns / 1_000_000.0
    private fun quote(value: String) = "\"" + value.replace("\"", "\"\"") + "\""
    private fun stats(values: List<Double>): JSONObject {
        if (values.isEmpty()) return JSONObject().put("count", 0)
        val s = values.sorted()
        val n = s.size
        return JSONObject().put("count", n).put("mean", s.average())
            .put("median", if (n % 2 == 0) (s[n / 2 - 1] + s[n / 2]) / 2 else s[n / 2])
            .put("p95", s[(ceil(n * .95).toInt() - 1).coerceIn(0, n - 1)])
    }
}
