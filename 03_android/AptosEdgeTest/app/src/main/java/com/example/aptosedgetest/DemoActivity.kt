package com.example.aptosedgetest

import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class DemoActivity : AppCompatActivity() {
    private val navy = Color.rgb(20, 48, 78)
    private val ink = Color.rgb(31, 41, 51)
    private val muted = Color.rgb(81, 97, 109)
    private val teal = Color.rgb(0, 112, 116)
    private val executor = Executors.newSingleThreadExecutor()
    @Volatile private var cancelled = false
    @Volatile private var destroyed = false
    private var busy = true
    private var samples = emptyList<DrPipeline.TestImage>()
    private var selectedIndex = 0
    private var selectedModel = 0
    private var previewGeneration = 0
    private var rawBitmap: Bitmap? = null
    private var processedBitmap: Bitmap? = null
    private var lastResult: DemoResult? = null
    private var lastWasBatch = false
    private var resultModel = ""
    private var statusValue = "Loading samples"
    private var progressValue = 0
    private var progressMaximum = 100
    private var progressIndeterminate = true
    private var predictedValue = "Not analyzed"
    private var resultDetail = ""
    private var timingValue = ""
    private var savedValue = ""
    private lateinit var modelPicker: Spinner
    private lateinit var samplePicker: Spinner
    private lateinit var rawView: ImageView
    private lateinit var processedView: ImageView
    private lateinit var truthView: TextView
    private lateinit var filenameView: TextView
    private lateinit var statusView: TextView
    private lateinit var predictionView: TextView
    private lateinit var predictionLabel: TextView
    private lateinit var detailView: TextView
    private lateinit var timingsView: TextView
    private lateinit var savedView: TextView
    private lateinit var bar: ProgressBar
    private lateinit var singleButton: Button
    private lateinit var batchButton: Button
    private lateinit var stopButton: Button
    private lateinit var exportButton: Button
    private lateinit var previousButton: ImageButton
    private lateinit var nextButton: ImageButton
    private var exportDirectory: File? = null

    private val exportLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        val dir = exportDirectory
        if (uri == null || dir == null) return@registerForActivityResult
        busy = true
        statusValue = "Exporting results"
        sync()
        executor.execute {
            val message = try {
                val output = contentResolver.openOutputStream(uri) ?: error("Cannot open selected destination.")
                ZipOutputStream(output).use { zip ->
                    dir.walkTopDown().filter { it.isFile }.sortedBy { it.name }.forEach { file ->
                        zip.putNextEntry(ZipEntry(file.relativeTo(dir).invariantSeparatorsPath))
                        file.inputStream().use { it.copyTo(zip) }
                        zip.closeEntry()
                    }
                }
                "Results exported"
            } catch (error: Exception) { "Export failed: ${error.message}" }
            ui { busy = false; statusValue = message; sync() }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildPage()
        executor.execute {
            try {
                check(OpenCVLoader.initDebug()) { "OpenCV initialization failed." }
                val loaded = DemoRunner(applicationContext).images()
                ui {
                    samples = loaded
                    selectedModel = DemoRunner.MODELS.indexOfFirst { it.key == intent.getStringExtra("demo_model") }.coerceAtLeast(0)
                    selectedIndex = intent.getIntExtra("demo_sample", 0).coerceIn(0, samples.lastIndex)
                    busy = false
                    statusValue = "Ready · ${samples.size} bundled samples"
                    progressIndeterminate = false
                    buildPage()
                    loadPreview()
                    // ADB verification invokes the same action as the visible buttons.
                    when (intent.getStringExtra("demo_action")) {
                        "single" -> startAnalysis(false)
                        "batch" -> startAnalysis(true)
                    }
                }
            } catch (error: Exception) { ui { busy = false; progressIndeterminate = false; statusValue = "Setup failed: ${error.message}"; sync() } }
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        buildPage()
    }

    override fun onDestroy() {
        destroyed = true
        cancelled = true
        executor.shutdown()
        super.onDestroy()
    }

    private fun buildPage() {
        val outer = ScrollView(this).apply { setBackgroundColor(Color.rgb(246, 248, 250)); isFillViewport = true }
        val wide = resources.configuration.screenWidthDp >= 900
        val page = column().apply { setPadding(dp(24), dp(if (wide) 10 else 18), dp(24), dp(16)) }
        outer.addView(page)
        setContentView(outer)
        ViewCompat.setOnApplyWindowInsetsListener(outer) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        page.addView(text("DR On-Device Analysis", if (wide) 23f else 26f, navy, true))
        page.addView(text("Research prototype — not for clinical use.", 13f, muted))
        page.addView(text("${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}   ·   Android ${android.os.Build.VERSION.RELEASE}", 12f, muted).apply { setPadding(0, dp(5), 0, dp(12)) })
        rule(page)

        val controls = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, dp(8), 0, dp(6)) }
        val modelColumn = column()
        modelColumn.addView(text("Model", 12f, muted, true))
        modelPicker = Spinner(this).apply {
            adapter = ArrayAdapter(this@DemoActivity, android.R.layout.simple_spinner_dropdown_item, DemoRunner.MODELS)
            setSelection(selectedModel)
            contentDescription = "Model"
        }
        modelColumn.addView(modelPicker, LinearLayout.LayoutParams(-1, dp(44)))
        val sampleColumn = column()
        sampleColumn.addView(text("Sample", 12f, muted, true))
        samplePicker = Spinner(this).apply {
            adapter = ArrayAdapter(this@DemoActivity, android.R.layout.simple_spinner_dropdown_item,
                samples.mapIndexed { i, sample -> "${i + 1}. ${sample.assetPath.substringAfterLast('/')}" }.ifEmpty { listOf("Loading") })
            setSelection(selectedIndex)
            contentDescription = "Sample"
        }
        sampleColumn.addView(samplePicker, LinearLayout.LayoutParams(-1, dp(44)))
        controls.addView(modelColumn, LinearLayout.LayoutParams(0, -2, .43f))
        controls.addView(sampleColumn, LinearLayout.LayoutParams(0, -2, .57f))
        page.addView(controls)

        val content = LinearLayout(this).apply { orientation = if (resources.configuration.screenWidthDp >= 900) LinearLayout.HORIZONTAL else LinearLayout.VERTICAL }
        val images = column()
        val imageRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val imageHeight = if (wide) 168 else 210
        fun imageColumn(label: String, original: Boolean): LinearLayout {
            val holder = column()
            holder.addView(text(label, 14f, ink, true).apply { setPadding(0, dp(8), 0, dp(8)) })
            val image = ImageView(this).apply {
                scaleType = ImageView.ScaleType.FIT_CENTER
                setBackgroundColor(Color.rgb(231, 237, 241))
                contentDescription = label
            }
            holder.addView(image, LinearLayout.LayoutParams(-1, dp(imageHeight)))
            if (original) rawView = image else processedView = image
            return holder
        }
        imageRow.addView(imageColumn("Original image", true), LinearLayout.LayoutParams(0, -2, 1f).apply { marginEnd = dp(12) })
        imageRow.addView(imageColumn("Processed · 224 × 224", false), LinearLayout.LayoutParams(0, -2, 1f))
        images.addView(imageRow)
        filenameView = text("", 13f, ink).apply { setPadding(0, dp(10), 0, dp(4)); setTextIsSelectable(true) }
        truthView = text("", 14f, muted)
        images.addView(filenameView)
        images.addView(truthView)
        val nav = LinearLayout(this).apply { gravity = Gravity.END; setPadding(0, dp(6), 0, 0) }
        previousButton = icon(android.R.drawable.ic_media_previous, "Previous sample") { selectSample(-1) }
        nextButton = icon(android.R.drawable.ic_media_next, "Next sample") { selectSample(1) }
        nav.addView(previousButton, LinearLayout.LayoutParams(dp(48), dp(44)))
        nav.addView(nextButton, LinearLayout.LayoutParams(dp(48), dp(44)))
        images.addView(nav)

        val results = column().apply { setPadding(if (content.orientation == LinearLayout.HORIZONTAL) dp(24) else 0, dp(8), 0, 0) }
        predictionLabel = text("Predicted grade", 13f, muted, true)
        results.addView(predictionLabel)
        predictionView = text("", 25f, navy, true).apply { setPadding(0, dp(5), 0, dp(5)) }
        detailView = text("", 13f, muted)
        timingsView = text("", if (wide) 14f else 16f, ink).apply { setPadding(0, dp(8), 0, dp(4)); setLineSpacing(dp(3).toFloat(), 1f) }
        results.addView(predictionView)
        results.addView(detailView)
        results.addView(timingsView)
        val timingInfo = Button(this).apply {
            text = "Timing details"; isAllCaps = false; setTextColor(teal); setBackgroundColor(Color.TRANSPARENT)
            setOnClickListener { AlertDialog.Builder(this@DemoActivity).setTitle("Measurement boundaries")
                .setMessage(timingDetails())
                .setPositiveButton("Close", null).show() }
        }
        results.addView(timingInfo, LinearLayout.LayoutParams(-2, dp(44)))
        if (content.orientation == LinearLayout.HORIZONTAL) {
            content.addView(images, LinearLayout.LayoutParams(0, -2, .62f))
            content.addView(results, LinearLayout.LayoutParams(0, -2, .38f))
        } else { content.addView(images); content.addView(results) }
        page.addView(content)
        rule(page)
        statusView = text("", 15f, teal, true).apply { setPadding(0, dp(12), 0, dp(6)) }
        page.addView(statusView)
        bar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { progressTintList = android.content.res.ColorStateList.valueOf(teal) }
        page.addView(bar, LinearLayout.LayoutParams(-1, dp(8)))
        val actions = LinearLayout(this).apply { setPadding(0, dp(14), 0, dp(10)); orientation = LinearLayout.HORIZONTAL }
        singleButton = command("Run Analysis", true) { startAnalysis(false) }
        batchButton = command("Run Batch Test", false) { startAnalysis(true) }
        stopButton = command("Stop", false) { cancelled = true; statusValue = "Stopping after current image"; sync() }
        actions.addView(singleButton, LinearLayout.LayoutParams(0, dp(52), 1f).apply { marginEnd = dp(10) })
        actions.addView(batchButton, LinearLayout.LayoutParams(0, dp(52), 1f).apply { marginEnd = dp(10) })
        actions.addView(stopButton, LinearLayout.LayoutParams(dp(80), dp(52)))
        page.addView(actions)
        savedView = text("", 12f, muted).apply {
            setOnClickListener { lastResult?.let { result -> AlertDialog.Builder(this@DemoActivity).setTitle("Results saved")
                .setMessage(result.directory.absolutePath).setPositiveButton("Close", null).show() } }
        }
        page.addView(savedView)
        exportButton = command("Export Results", false) {
            lastResult?.let { exportDirectory = it.directory; exportLauncher.launch("${it.directory.name}.zip") }
        }
        page.addView(exportButton, LinearLayout.LayoutParams(-2, dp(48)).apply { topMargin = dp(8) })
        modelPicker.onItemSelectedListener = selected { position ->
            if (selectedModel != position && !busy) { selectedModel = position; clearCurrent(); sync() }
        }
        samplePicker.onItemSelectedListener = selected { position ->
            if (selectedIndex != position && !busy) { selectedIndex = position; clearCurrent(); loadPreview() }
        }
        sync()
    }

    private fun clearCurrent() {
        processedBitmap = null
        predictedValue = "Not analyzed"
        resultDetail = ""
        timingValue = ""
        lastWasBatch = false
        lastResult = null
        savedValue = ""
        statusValue = "Ready"
        progressValue = 0
    }

    private fun selectSample(delta: Int) {
        if (busy || samples.isEmpty()) return
        samplePicker.setSelection((selectedIndex + delta).coerceIn(0, samples.lastIndex))
    }

    private fun loadPreview() {
        if (samples.isEmpty()) return
        val sample = samples[selectedIndex]
        val generation = ++previewGeneration
        rawBitmap = null
        sync()
        executor.execute {
            val bitmap = try {
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                assets.open(sample.assetPath).use { BitmapFactory.decodeStream(it, null, bounds) }
                val options = BitmapFactory.Options().apply { inSampleSize = 1 }
                while (maxOf(bounds.outWidth, bounds.outHeight) / options.inSampleSize > 1200) options.inSampleSize *= 2
                assets.open(sample.assetPath).use { BitmapFactory.decodeStream(it, null, options) }
            } catch (error: Exception) { android.util.Log.e("DR_DEMO", "Preview failed for ${sample.assetPath}", error); null }
            ui {
                if (generation == previewGeneration) {
                    rawBitmap = bitmap
                    if (bitmap == null && !busy) statusValue = "Preview could not be loaded"
                    sync()
                } else bitmap?.recycle()
            }
        }
    }

    private fun startAnalysis(batch: Boolean) {
        if (busy || samples.isEmpty()) return
        busy = true
        cancelled = false
        clearCurrent()
        lastWasBatch = batch
        resultModel = DemoRunner.MODELS[selectedModel].title
        statusValue = "Loading model"
        progressIndeterminate = true
        progressMaximum = if (batch) samples.size else 1
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        sync()
        val model = DemoRunner.MODELS[selectedModel]
        val chosen = if (batch) samples.toList() else listOf(samples[selectedIndex])
        val clickStart = System.nanoTime()
        executor.execute {
            try {
                val result = DemoRunner(applicationContext).run(model, chosen, samples.first(), batch, { cancelled || destroyed }) { progress ->
                    ui {
                        statusValue = "${progress.phase} · ${progress.succeeded} successful · ${progress.failed} failed"
                        progressIndeterminate = progress.phase.startsWith("Loading") || progress.phase.startsWith("Warming")
                        progressValue = progress.attempted
                        progressMaximum = progress.total
                        sync()
                    }
                }
                ui {
                    busy = false
                    progressIndeterminate = false
                    lastResult = result
                    processedBitmap = result.preview
                    predictedValue = if (batch) "${result.correct} / ${result.succeeded} correct" else result.last?.let { DemoRunner.grade(it.predicted) } ?: "No prediction"
                    resultDetail = if (batch) "$resultModel · ${result.succeeded} successful, ${result.failed} failed\nDescriptive results on the selected sample set" else "$resultModel · actual model output"
                    val ms = result.summary
                    if (result.succeeded > 0) {
                        fun values(key: String) = ms.getJSONObject(key)
                        if (batch) {
                            fun statLine(label: String, key: String) = "$label mean  ${fmt(values(key).getDouble("mean"))} ms"
                            timingValue = statLine("Preprocessing", "preprocessing_ms") + "\n" + statLine("Inference", "inference_ms") + "\n" + statLine("Pipeline", "pipeline_e2e_ms")
                        } else {
                            timingValue = "Preprocessing  ${fmt(values("preprocessing_ms").getDouble("mean"))} ms\nModel inference  ${fmt(values("inference_ms").getDouble("mean"))} ms\nPipeline end-to-end  ${fmt(values("pipeline_e2e_ms").getDouble("mean"))} ms"
                        }
                    }
                    statusValue = when (result.status) {
                        "complete" -> "Complete · ${result.attempted} / ${result.total} processed"
                        "cancelled" -> "Stopped · ${result.attempted} / ${result.total} processed"
                        else -> "${result.status.replace('_', ' ')} · ${result.error ?: "See errors.json"}"
                    }
                    savedValue = "Results saved · ${result.directory.name}"
                    if (batch && result.last != null) {
                        selectedIndex = samples.indexOfFirst { it.assetPath == result.last.imagePath }.coerceAtLeast(0)
                        samplePicker.setSelection(selectedIndex)
                        loadPreview()
                    }
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    sync()
                    val elapsed = (System.nanoTime() - clickStart) / 1_000_000.0
                    timingValue += "\nTap to result  ${fmt(elapsed)} ms"
                    timingsView.text = timingValue
                    executor.execute {
                        File(result.directory, "ui_timing.json").writeText(JSONObject()
                            .put("tap_to_result_text_update_ms", elapsed)
                            .put("note", "Includes initialization, warm-ups, worker saving and UI dispatch; excludes the next display frame and this timing-file write.").toString(2))
                    }
                }
            } catch (error: Exception) {
                ui { busy = false; progressIndeterminate = false; statusValue = "Run failed: ${error.message}"; window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); sync() }
            }
        }
    }

    private fun sync() {
        if (!::statusView.isInitialized) return
        statusView.text = statusValue
        predictionLabel.text = if (lastWasBatch) "Batch result" else "Predicted grade"
        predictionView.text = predictedValue
        detailView.text = resultDetail
        timingsView.text = timingValue
        savedView.text = savedValue
        rawView.setImageBitmap(rawBitmap)
        processedView.setImageBitmap(processedBitmap)
        samples.getOrNull(selectedIndex)?.let {
            filenameView.text = it.assetPath.substringAfterLast('/')
            truthView.text = "Dataset label: ${DemoRunner.grade(it.groundTruthFolder)}"
        }
        bar.isIndeterminate = progressIndeterminate
        bar.max = progressMaximum
        bar.progress = progressValue
        modelPicker.isEnabled = !busy
        samplePicker.isEnabled = !busy && samples.isNotEmpty()
        singleButton.isEnabled = !busy && samples.isNotEmpty()
        batchButton.isEnabled = !busy && samples.isNotEmpty()
        stopButton.isEnabled = busy && samples.isNotEmpty() && !cancelled
        previousButton.isEnabled = !busy && selectedIndex > 0
        nextButton.isEnabled = !busy && selectedIndex < samples.lastIndex
        exportButton.isEnabled = !busy && lastResult != null
        listOf(singleButton, batchButton, stopButton, exportButton, previousButton, nextButton).forEach { it.alpha = if (it.isEnabled) 1f else .4f }
    }

    private fun timingDetails(): String {
        var details = DemoRunner.TIMING + "\n\nInference measures Interpreter.run only.\n\nTap-to-result includes setup, warm-ups, result saving and UI dispatch, ending when result text is updated. It is not a display-frame latency measurement."
        lastResult?.let { result ->
            for ((label, key) in listOf("Preprocessing" to "preprocessing_ms", "Inference" to "inference_ms", "Pipeline" to "pipeline_e2e_ms")) {
                val values = result.summary.optJSONObject(key)
                if (values != null && values.optInt("count") > 0) details += "\n\n$label (ms): mean ${fmt(values.getDouble("mean"))}; median ${fmt(values.getDouble("median"))}; p95 ${fmt(values.getDouble("p95"))}."
            }
        }
        return details + "\n\nNew demonstration measurements; these do not replace the manuscript's historical latency results."
    }

    private fun ui(action: () -> Unit) { if (!destroyed) runOnUiThread { if (!destroyed) action() } }
    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
    private fun fmt(value: Double) = String.format(Locale.US, "%.2f", value)
    private fun column() = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    private fun text(value: String, size: Float, color: Int, bold: Boolean = false) = TextView(this).apply {
        text = value; textSize = size; setTextColor(color); if (bold) setTypeface(typeface, Typeface.BOLD)
    }
    private fun rule(parent: LinearLayout) { parent.addView(View(this).apply { setBackgroundColor(Color.rgb(212, 222, 228)) }, LinearLayout.LayoutParams(-1, dp(1))) }
    private fun command(label: String, primary: Boolean, action: () -> Unit) = Button(this).apply {
        text = label; textSize = 14f; isAllCaps = false; setPadding(dp(10), 0, dp(10), 0)
        setTextColor(if (primary) Color.WHITE else navy)
        background = GradientDrawable().apply { setColor(if (primary) navy else Color.WHITE); cornerRadius = dp(5).toFloat(); if (!primary) setStroke(dp(1), Color.rgb(184, 201, 213)) }
        setOnClickListener { action() }
    }
    private fun icon(resource: Int, label: String, action: () -> Unit) = ImageButton(this).apply {
        setImageResource(resource); contentDescription = label; tooltipText = label
        setColorFilter(navy); setBackgroundColor(Color.TRANSPARENT); setOnClickListener { action() }
    }
    private fun selected(action: (Int) -> Unit) = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) { action(position) }
        override fun onNothingSelected(parent: AdapterView<*>?) = Unit
    }
}
