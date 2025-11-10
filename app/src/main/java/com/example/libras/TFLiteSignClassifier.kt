package com.example.libras

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil

class TFLiteSignClassifier(
    context: Context,
    preferCpuOnly: Boolean = true
) {
    private val tflite: Interpreter
    val labels: List<String>

    init {
        val model = FileUtil.loadMappedFile(context, "libras_landmarks_mlp_v2_fp32.tflite")
        val opts = Interpreter.Options().apply {
        }
        tflite = Interpreter(model, opts)
        labels = FileUtil.loadLabels(context, "labels.txt")
    }

    /**
     * Inference: input 1x43 float32 -> output 1xN float32 (softmax)
     */
    fun predictFromFeatures(feat43: FloatArray): FloatArray {
        require(feat43.size == 43)
        val input = arrayOf(feat43)
        val output = Array(1) { FloatArray(labels.size) }
        tflite.run(input, output)
        return output[0]
    }

    fun close() = runCatching { tflite.close() }
}