package com.example.libras

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.PointF
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.contract.ActivityResultContracts.StartActivityForResult
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.material3.SuggestionChipDefaults.suggestionChipColors
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.libras.ui.theme.LibrasTheme
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import androidx.compose.ui.draw.alpha
import com.example.libras.toBitmap
private const val TAG = "LIBRAS"
private enum class OverlayMode { NONE, POINTS, POINTS_LINES }

/** Conexões dos 21 pontos. */
private val HAND_CONNECTIONS: List<Pair<Int, Int>> = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4,
    0 to 5, 5 to 6, 6 to 7, 7 to 8,
    0 to 9, 9 to 10, 10 to 11, 11 to 12,
    0 to 13, 13 to 14, 14 to 15, 15 to 16,
    0 to 17, 17 to 18, 18 to 19, 19 to 20,
    5 to 9, 9 to 13, 13 to 17, 17 to 5
)
/* -------------------- Dicionário / Sugestão -------------------- */
private class SimplePredictor(
    private val words: List<String>
) {
    fun suggest(token: String): String? {
        if (token.isEmpty()) return null
        val t = token.lowercase(Locale.ROOT)
        val pref = words.firstOrNull { it.startsWith(t) }
        if (pref != null) return pref
        var best: String? = null
        var bestD = Int.MAX_VALUE
        for (w in words) {
            val d = lev(t, w)
            if (d < bestD) { bestD = d; best = w }
            if (bestD == 1) break
        }
        return if (bestD <= 2) best else null
    }
    private fun lev(a: String, b: String): Int {
        val n = a.length; val m = b.length
        if (n == 0) return m
        if (m == 0) return n
        val dp = IntArray(m + 1) { it }
        for (i in 1..n) {
            var prev = dp[0]
            dp[0] = i
            for (j in 1..m) {
                val tmp = dp[j]
                val cost = if (a[i - 1] == b[j - 1]) 0 else 1
                dp[j] = minOf(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = tmp
            }
        }
        return dp[m]
    }
    companion object {
        /** Lê `assets/dicionario.txt` (um termo por linha). */
        fun fromAssets(activity: Activity): SimplePredictor {
            val list = try {
                val inStream = activity.assets.open("dicionario.txt")
                BufferedReader(InputStreamReader(inStream)).use { br ->
                    br.lineSequence().map { it.trim() }
                        .filter { it.isNotEmpty() && it.all { ch -> ch.isLetter() } }
                        .take(20000).toList()
                }
            } catch (_: Throwable) {
                listOf("oi","ola","bom","boa","tarde","noite","dia","bem","voce","sim","nao")
            }
            val sorted = list.distinct().map { it.lowercase(Locale.ROOT) }
                .sortedWith(compareBy<String> { it.length }.thenBy { it })
            return SimplePredictor(sorted)
        }
    }
}

/* -------------------- Caixa de legenda -------------------- */

private class CaptionBox(private val maxCharsPerLine: Int) {
    var line1 by mutableStateOf(""); private set
    var line2 by mutableStateOf(""); private set
    private var liveToken: String? = null

    fun clear() { line1 = ""; line2 = ""; liveToken = null }

    fun appendWord(word: String) {
        removeLiveToken()
        val w = if (line2.isEmpty()) word else " $word"
        if (line2.length + w.length <= maxCharsPerLine) {
            line2 += w
        } else {
            line1 = line2
            line2 = word
        }
    }

    fun appendSpace() {
        removeLiveToken()
        if (line2.isEmpty()) return
        if (line2.length + 1 <= maxCharsPerLine) {
            line2 += " "
        } else {
            line1 = line2
            line2 = ""
        }
    }

    fun setLiveToken(replacement: String?) {
        if (replacement == null) { removeLiveToken(); return }
        if (liveToken == null) {
            if (line2.isEmpty()) {
                line2 = replacement.take(maxCharsPerLine)
            } else {
                val add = " $replacement"
                val txt = line2 + add
                if (txt.length <= maxCharsPerLine) line2 = txt
            }
        } else {
            val parts = if (line2.isNotEmpty()) line2.split(" ") else emptyList()
            if (parts.isNotEmpty()) {
                val rebuilt = (parts.dropLast(1) + replacement).joinToString(" ")
                line2 = rebuilt.take(maxCharsPerLine)
            } else {
                line2 = replacement.take(maxCharsPerLine)
            }
        }
        liveToken = replacement
    }

    private fun removeLiveToken() {
        if (liveToken == null) return
        val parts = if (line2.isNotEmpty()) line2.split(" ") else emptyList()
        if (parts.isNotEmpty()) {
            line2 = parts.dropLast(1).joinToString(" ")
        }
        liveToken = null
    }

    fun deleteCommittedChar(): Char? {
        removeLiveToken()
        if (line2.isNotEmpty()) {
            val ch = line2.last()
            line2 = line2.dropLast(1)
            return ch
        } else if (line1.isNotEmpty()) {
            line2 = line1
            line1 = ""
            val ch = line2.last()
            line2 = line2.dropLast(1)
            return ch
        }
        return null
    }

    fun startEditingLastWord(): String? {
        removeLiveToken()
        val t = line2.trimEnd()
        if (t.isEmpty()) return null
        val parts = t.split(" ")
        val last = parts.lastOrNull() ?: return null
        line2 = parts.dropLast(1).joinToString(" ")
        liveToken = null
        setLiveToken(last)
        return last
    }
}

/* -------------------- Activity -------------------- */

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val requestCameraPermission = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted -> Log.d(TAG, "Camera permission granted? $granted") }

        val requestAudioPermission = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted -> Log.d(TAG, "Audio permission granted? $granted") }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestAudioPermission.launch(Manifest.permission.RECORD_AUDIO)
        }

        setContent { LibrasTheme { CameraWithInference() } }
    }

    override fun onDestroy() {
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        super.onDestroy()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CameraWithInference() {
    val context = LocalContext.current
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE

    val activity = context as Activity
    val lifecycleOwner = LocalLifecycleOwner.current

    val previewView = remember(context) {
        PreviewView(context).apply { implementationMode = PreviewView.ImplementationMode.COMPATIBLE }
    }

    // ======= Modelo + tracker =======
    var status by remember { mutableStateOf("carregando modelo…") }
    val classifier = remember {
        try {
            TFLiteSignClassifier(context, preferCpuOnly = true).also {
                it.predictFromFeatures(FloatArray(43) { 0f })
                status = "modelo pronto (${it.labels.size} letras)"
            }
        } catch (t: Throwable) {
            Log.e(TAG, "init classifier", t)
            status = "falha init modelo: ${t.javaClass.simpleName}"
            null
        }
    }
    val yuv = remember { YuvToRgbConverter() }
    val tracker = remember { GestureBoxTracker(context) }
    DisposableEffect(Unit) { onDispose { tracker.close() } }

    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { analysisExecutor.shutdown() } }

    // HUD
    var head by remember { mutableStateOf("?") }
    var top3 by remember { mutableStateOf(listOf<String>()) }

    // Overlay
    var viewW by remember { mutableIntStateOf(0) }
    var viewH by remember { mutableIntStateOf(0) }
    var roiOnView by remember { mutableStateOf<android.graphics.Rect?>(null) }
    var lmk by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var overlayMode by remember { mutableStateOf(OverlayMode.POINTS) }

    // Camera
    var lensFacing by remember { mutableIntStateOf(CameraSelector.LENS_FACING_FRONT) }
    var cameraProvider by remember { mutableStateOf<ProcessCameraProvider?>(null) }

    // Smoothing
    var emaL by remember { mutableStateOf<FloatArray?>(null) }
    var emaR by remember { mutableStateOf<FloatArray?>(null) }
    val alpha = 0.6f

    // thresholds
    val CONF_THRESH = 0.65f
    val DWELL_MS = 1200L
    val NO_HAND_SPACE_MS = 1200L
    val SPACE_TOAST_MS = 700L

    // Dwell
    var stableIdx by remember { mutableStateOf(-1) }
    var stableSince by remember { mutableStateOf(0L) }

    // “Mão não detectada” -> espaço
    var lastHandSeenMs by remember { mutableStateOf(System.currentTimeMillis()) }
    var spaceToastUntil by remember { mutableStateOf(0L) }

    // Legenda & dicionário
    val caption = remember { CaptionBox(maxCharsPerLine = if (isLandscape) 36 else 28) }
    var rawToken by remember { mutableStateOf("") }
    var suggestedWord by remember { mutableStateOf<String?>(null) }
    var suggestedStampMs by remember { mutableStateOf(0L) }
    val predictor = remember { SimplePredictor.fromAssets(activity) }

    // === Speech to Text (RecognizerIntent) ===
    val speechLauncher = rememberLauncherForActivityResult(StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data = result.data
            val matches = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            val text = matches?.firstOrNull()?.trim()
            if (!text.isNullOrEmpty()) {
                caption.setLiveToken(null)
                rawToken = ""
                suggestedWord = null
                text.split(Regex("\\s+"))
                    .filter { it.isNotEmpty() }
                    .forEach { caption.appendWord(it) }
                caption.appendSpace()
            }
        }
    }
    fun startSpeechToText() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.w(TAG, "RECORD_AUDIO not granted")
            return
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Fale…")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        speechLauncher.launch(intent)
    }

    // === Text to Speech (🔊) ===
    var ttsReady by remember { mutableStateOf(false) }
    var tts by remember { mutableStateOf<TextToSpeech?>(null) }
    LaunchedEffect(Unit) {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val res = tts?.setLanguage(Locale.getDefault())
                ttsReady = true
                Log.d(TAG, "TTS ready (lang=$res)")
            } else {
                Log.e(TAG, "TTS init failed: $status")
            }
        }
    }
    DisposableEffect(Unit) { onDispose { tts?.stop(); tts?.shutdown(); tts = null } }
    fun speakCaption() {
        val text = listOf(caption.line1, caption.line2)
            .filter { it.isNotBlank() }
            .joinToString(" ")
            .trim()
        if (text.isNotEmpty() && ttsReady) {
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "CAPTION")
        } else {
            Log.d(TAG, "Nothing to speak or TTS not ready")
        }
    }

    // Toast “Espaço” auto (0.7s)
    LaunchedEffect(spaceToastUntil) {
        val now = System.currentTimeMillis()
        val wait = spaceToastUntil - now
        if (wait > 0) {
            kotlinx.coroutines.delay(wait)
            spaceToastUntil = 0L
        }
    }

    fun commitLetter(ch: Char) {
        rawToken += ch
        caption.setLiveToken(rawToken)
        predictor.suggest(rawToken)?.let {
            suggestedWord = it
            suggestedStampMs = System.currentTimeMillis()
        } ?: run { suggestedWord = null }
    }

    fun acceptSuggestion() {
        val sel = suggestedWord ?: return
        caption.setLiveToken(null)
        caption.appendWord(sel)
        rawToken = ""
        suggestedWord = null
    }

    fun clearAll() {
        caption.clear()
        rawToken = ""
        suggestedWord = null
    }

    fun backspaceOneChar() {
        if (rawToken.isNotEmpty()) {
            rawToken = rawToken.dropLast(1)
            caption.setLiveToken(if (rawToken.isNotEmpty()) rawToken else null)
            predictor.suggest(rawToken)?.let {
                suggestedWord = it
                suggestedStampMs = System.currentTimeMillis()
            } ?: run { suggestedWord = null }
        } else {
            val removed = caption.deleteCommittedChar()
            if (removed == ' ') {
                rawToken = caption.startEditingLastWord() ?: ""
                if (rawToken.isNotEmpty()) {
                    predictor.suggest(rawToken)?.let {
                        suggestedWord = it
                        suggestedStampMs = System.currentTimeMillis()
                    } ?: run { suggestedWord = null }
                }
            }
        }
    }

    fun finishWordBecauseNoHand(now: Long) {
        if (rawToken.isNotEmpty()) {
            caption.setLiveToken(null)
            caption.appendWord(rawToken)
            rawToken = ""
            suggestedWord = null
            caption.appendSpace()
            spaceToastUntil = now + SPACE_TOAST_MS
        }
    }

    fun bindCamera(provider: ProcessCameraProvider, mirrorX: Boolean) {
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val analyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(previewView.display.rotation)
            .build()
            .apply {
                setAnalyzer(analysisExecutor) { img ->
                    try {
                        val c = classifier ?: run { img.close(); return@setAnalyzer }
                        val bmp = img.toBitmap(yuv)
                        val rot = img.imageInfo.rotationDegrees
                        val now = System.currentTimeMillis()

                        val both = tracker.detectBoth(bmp, rot)
                        val fw = both.frameW; val fh = both.frameH

                        val vw = if (viewW > 0) viewW else previewView.width
                        val vh = if (viewH > 0) viewH else previewView.height

                        fun mapPtsToView(pts01: List<PointF>): List<Offset> =
                            pts01.map { p01 ->
                                val xPx = p01.x * fw
                                val yPx = p01.y * fh
                                val (xv, yv) = mapBitmapPointToView(
                                    x = xPx, y = yPx,
                                    bmpW = fw, bmpH = fh,
                                    rotationDeg = 0,
                                    viewW = vw.toFloat(), viewH = vh.toFloat(),
                                    mirrorX = mirrorX
                                )
                                Offset(xv, yv)
                            }

                        data class HandCandidate(
                            val sideLeft: Boolean,
                            val probs: FloatArray,
                            val peak: Float,
                            val roi: android.graphics.Rect,
                            val ptsView: List<Offset>
                        )

                        val candidates = mutableListOf<HandCandidate>()
                        both.left?.let { h ->
                            val feat = LandmarkFeaturizer.build43CanonicalFrom01(h.landmarks01, h.isLeft)
                            val probs = c.predictFromFeatures(feat)
                            if (probs.isNotEmpty() && !probs.any { it.isNaN() }) {
                                candidates += HandCandidate(true, probs, probs.maxOrNull() ?: 0f,
                                    mapBitmapRectToView(h.roiPx, fw, fh, 0, vw, vh, mirrorX),
                                    mapPtsToView(h.landmarks01))
                            }
                        }
                        both.right?.let { h ->
                            val feat = LandmarkFeaturizer.build43CanonicalFrom01(h.landmarks01, h.isLeft)
                            val probs = c.predictFromFeatures(feat)
                            if (probs.isNotEmpty() && !probs.any { it.isNaN() }) {
                                candidates += HandCandidate(false, probs, probs.maxOrNull() ?: 0f,
                                    mapBitmapRectToView(h.roiPx, fw, fh, 0, vw, vh, mirrorX),
                                    mapPtsToView(h.landmarks01))
                            }
                        }

                        if (candidates.isEmpty()) {
                            roiOnView = null; lmk = emptyList()
                            head = "?"
                            top3 = emptyList()
                            stableIdx = -1
                            stableSince = 0L
                            if (now - lastHandSeenMs >= NO_HAND_SPACE_MS) {
                                finishWordBecauseNoHand(now)
                            }
                        } else {
                            lastHandSeenMs = now
                            val best = candidates.maxBy { it.peak }
                            val prevEma = if (best.sideLeft) emaL else emaR
                            val smoothed = prevEma?.let { prev ->
                                FloatArray(best.probs.size) { i -> alpha * prev[i] + (1f - alpha) * best.probs[i] }
                            } ?: best.probs
                            if (best.sideLeft) emaL = smoothed else emaR = smoothed

                            roiOnView = best.roi
                            lmk = best.ptsView

                            val labels = c.labels
                            val idxs = smoothed.indices.sortedByDescending { smoothed[it] }.take(3)
                            val bestIdx = idxs[0]
                            val bestProb = smoothed[bestIdx]
                            val bestLabel = labels[bestIdx]

                            if (bestProb >= CONF_THRESH) {
                                if (stableIdx == bestIdx) {
                                    // mantém
                                } else {
                                    stableIdx = bestIdx
                                    stableSince = now
                                }
                                val elapsed = now - stableSince
                                if (elapsed >= DWELL_MS) {
                                    val ch = bestLabel.firstOrNull() ?: '?'
                                    commitLetter(ch)
                                    stableIdx = -1
                                    stableSince = 0L
                                }
                            } else {
                                stableIdx = -1
                                stableSince = 0L
                            }

                            val headText = "[${if (best.sideLeft) "L" else "R"}] $bestLabel  ${(bestProb * 100f).format1()}%"
                            val t3 = idxs.map { i -> "${labels[i]} ${(smoothed[i] * 100f).format1()}%" }
                            previewView.post { head = headText; top3 = t3; status = "" }
                        }
                    } catch (t: Throwable) {
                        Log.e(TAG, "Analyzer error", t)
                        previewView.post { status = "analyzer error: ${t.javaClass.simpleName}" }
                    } finally { try { img.close() } catch (_: Throwable) {} }
                }
            }

        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        try {
            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, selector, preview, analyzer)
        } catch (t: Throwable) {
            Log.e(TAG, "bindToLifecycle failed", t)
            status = "camera bind failed"
        }
    }

    // Bind inicial
    DisposableEffect(Unit) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            cameraProvider = future.get()
            val initialMirror = (lensFacing == CameraSelector.LENS_FACING_FRONT)
            cameraProvider?.let { bindCamera(it, initialMirror) }
        }, ContextCompat.getMainExecutor(context))
        onDispose { }
    }

    Box(Modifier.fillMaxSize()) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier
                .fillMaxSize()
                .onGloballyPositioned { coords ->
                    viewW = coords.size.width
                    viewH = coords.size.height
                }
        )

        // Overlay (caixa / pontos / linhas)
        Canvas(Modifier.fillMaxSize()) {
            roiOnView?.let { r ->
                drawRect(
                    color = Color(0xFF00FF00),
                    topLeft = Offset(r.left.toFloat(), r.top.toFloat()),
                    size = androidx.compose.ui.geometry.Size(
                        (r.right - r.left).toFloat(),
                        (r.bottom - r.top).toFloat()
                    ),
                    style = Stroke(width = 6f)
                )
            }
            when (overlayMode) {
                OverlayMode.NONE -> Unit
                OverlayMode.POINTS -> {
                    lmk.forEach { p -> drawCircle(Color.Cyan, 6f, p, style = Stroke(2f)) }
                }
                OverlayMode.POINTS_LINES -> {
                    HAND_CONNECTIONS.forEach { (a, b) ->
                        if (a in lmk.indices && b in lmk.indices) {
                            drawLine(Color.Cyan, lmk[a], lmk[b], strokeWidth = 3f)
                        }
                    }
                    lmk.forEach { p -> drawCircle(Color.Cyan, 6f, p, style = Stroke(2f)) }
                }
            }
        }

        // HUD (top-center)
        Column(
            Modifier
                .align(Alignment.TopCenter)
                .padding(top = 48.dp)
                .background(Color(0x80000000))
                .padding(10.dp)
        ) {
            Text(
                text = if (status.isNotEmpty()) status else head,
                color = if (status.isNotEmpty()) Color.Yellow else Color.Green,
                style = MaterialTheme.typography.titleMedium
            )
            if (status.isEmpty() && head != "?") {
                Spacer(Modifier.height(6.dp))
                top3.forEach { Text(text = it, color = Color.White) }
            }
        }

        // Botão "..." + menu (top-left)
        var menuOpen by remember { mutableStateOf(false) }
        Box(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(start = 12.dp, top = 48.dp)
        ) {
            FloatingActionButton(onClick = { menuOpen = !menuOpen }, modifier = Modifier.size(44.dp)) {
                Text("...")
            }
            DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                DropdownMenuItem(
                    text = { Text("Mudar câmera") },
                    onClick = {
                        lensFacing =
                            if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                                CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
                        val mirrorNow = (lensFacing == CameraSelector.LENS_FACING_FRONT)
                        cameraProvider?.let { bindCamera(it, mirrorNow) }
                        menuOpen = false
                    }
                )
                Divider()
                DropdownMenuItem(
                    text = { Text("Orientação") },
                    onClick = {
                        val cur = context.resources.configuration.orientation
                        activity.requestedOrientation =
                            if (cur == Configuration.ORIENTATION_LANDSCAPE)
                                ActivityInfo.SCREEN_ORIENTATION_SENSOR_PORTRAIT
                            else
                                ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
                        menuOpen = false
                    }
                )
                Divider()
                DropdownMenuItem(
                    text = {
                        Text(
                            when (overlayMode) {
                                OverlayMode.NONE -> "Sobreposição: Nenhuma"
                                OverlayMode.POINTS -> "Sobreposição: Pontos"
                                OverlayMode.POINTS_LINES -> "Sobreposição: Pontos + Linhas"
                            }
                        )
                    },
                    onClick = {
                        overlayMode = when (overlayMode) {
                            OverlayMode.NONE -> OverlayMode.POINTS
                            OverlayMode.POINTS -> OverlayMode.POINTS_LINES
                            OverlayMode.POINTS_LINES -> OverlayMode.NONE
                        }
                        menuOpen = false
                    }
                )
            }
        }

        // ===== Posição comum para Chip e Toast =====
        val suggestionBottom = if (isLandscape) 96.dp else 122.dp

        // Chip de sugestão
        if (suggestedWord != null) {
            Box(
                Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = suggestionBottom)
            ) {
                SuggestionChip(
                    onClick = { acceptSuggestion() },
                    label = { Text(suggestedWord!!) },
                    colors = suggestionChipColors(
                        containerColor = Color(0xC0000000),
                        labelColor = Color.White
                    )
                )
            }
        }

        // Toast “Espaço” (0.7s)
        val nowMs = System.currentTimeMillis()
        if (spaceToastUntil > nowMs) {
            Box(
                Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = suggestionBottom)
            ) {
                Text(
                    "Espaço",
                    color = Color.White,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier
                        .background(Color(0xC0000000), RoundedCornerShape(8.dp))
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                )
            }
        }

        // Legenda + botões
        val bottomPad = if (isLandscape) 10.dp else 44.dp
        val fillFrac = if (isLandscape) 0.86f else 0.96f
        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .padding(horizontal = if (isLandscape) 24.dp else 12.dp)
                .padding(bottom = bottomPad)
                .fillMaxWidth(fillFrac)
        ) {
            // barra (🗑 • 🎤 • 🔊 • ⌫)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                SmallFloatingActionButton(onClick = { clearAll() }, modifier = Modifier.size(60.dp)) {
                    Text("🗑")
                }
                SmallFloatingActionButton(onClick = { startSpeechToText() }, modifier = Modifier.size(60.dp)) {
                    Text("🎤")
                }
                SmallFloatingActionButton(
                    onClick = { if (ttsReady) speakCaption() },
                    modifier = Modifier
                        .size(60.dp)
                        .alpha(if (ttsReady) 1f else 0.4f)
                ) { Text("🔊") }
                SmallFloatingActionButton(onClick = { backspaceOneChar() }, modifier = Modifier.size(60.dp)) {
                    Text("⌫")
                }
            }

            Spacer(Modifier.height(8.dp))
            Column(
                Modifier
                    .fillMaxWidth()
                    .background(Color(0xC0000000), shape = RoundedCornerShape(12.dp))
                    .padding(horizontal = 12.dp, vertical = 10.dp)
            ) {
                Text(
                    text = caption.line1,
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = caption.line2,
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
    }
}
/* ---------- helpers ---------- */
private fun mapBitmapPointToView(
    x: Float,
    y: Float,
    bmpW: Int,
    bmpH: Int,
    rotationDeg: Int,
    viewW: Float,
    viewH: Float,
    mirrorX: Boolean
): Pair<Float, Float> {
    val rotW = if (rotationDeg % 180 == 0) bmpW else bmpH
    val rotH = if (rotationDeg % 180 == 0) bmpH else bmpW

    val scale = max(viewW / rotW.toFloat(), viewH / rotH.toFloat())
    val dispW = rotW * scale
    val dispH = rotH * scale
    val left = (viewW - dispW) / 2f
    val top = (viewH - dispH) / 2f

    val (rx, ry) = when ((rotationDeg % 360 + 360) % 360) {
        0 -> x to y
        90 -> (bmpH - y) to x
        180 -> (bmpW - x) to (bmpH - y)
        270 -> y to (bmpW - x)
        else -> x to y
    }

    var vx = left + rx * scale
    val vy = top + ry * scale
    if (mirrorX) vx = viewW - vx
    return vx to vy
}

private fun mapBitmapRectToView(
    r: android.graphics.Rect,
    bmpW: Int, bmpH: Int,
    rotationDeg: Int,
    viewW: Int, viewH: Int,
    mirrorX: Boolean
): android.graphics.Rect {
    val (x0, y0) = mapBitmapPointToView(
        r.left.toFloat(), r.top.toFloat(),
        bmpW, bmpH, rotationDeg, viewW.toFloat(), viewH.toFloat(), mirrorX
    )
    val (x1, y1) = mapBitmapPointToView(
        r.right.toFloat(), r.bottom.toFloat(),
        bmpW, bmpH, rotationDeg, viewW.toFloat(), viewH.toFloat(), mirrorX
    )
    val l = min(x0, x1).toInt()
    val t = min(y0, y1).toInt()
    val rr = max(x0, x1).toInt()
    val b = max(y0, y1).toInt()
    return android.graphics.Rect(l, t, rr, b)
}
private fun Float.format1(): String = String.format(Locale.US, "%.1f", this)