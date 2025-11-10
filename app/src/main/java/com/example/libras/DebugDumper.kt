package com.example.libras

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

object DebugDumper {
    private const val TAG = "LIBRAS-DUMP"

    fun dumpOnce(
        context: Context,
        bmp: Bitmap,
        handLabel: String, // "L" or "R"
        feat43: FloatArray,
        probs: FloatArray,
        labels: List<String>
    ) {
        try {
            val dir = File(context.getExternalFilesDir(null), "parity_dumps").apply { mkdirs() }
            val ts = System.currentTimeMillis()
            // 1) Save the frame
            val png = File(dir, "frame_${handLabel}_$ts.png")
            FileOutputStream(png).use { out -> bmp.compress(Bitmap.CompressFormat.PNG, 100, out) }

            // 2) Save JSON (features + probs + labels)
            val j = JSONObject().apply {
                put("timestamp", ts)
                put("hand", handLabel)
                put("feat43", JSONArray().apply { feat43.forEach { put(it) } })
                put("probs", JSONArray().apply { probs.forEach { put(it) } })
                put("labels", JSONArray().apply { labels.forEach { put(it) } })
            }
            val jsonFile = File(dir, "dump_${handLabel}_$ts.json")
            jsonFile.writeText(j.toString())

            Log.i(TAG, "Dumped to ${png.absolutePath} and ${jsonFile.absolutePath}")
        } catch (t: Throwable) {
            Log.e(TAG, "Dump failed", t)
        }
    }
}
