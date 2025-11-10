package com.example.libras

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy

/**
 * Converts an ImageProxy (YUV_420_888) to an ARGB_8888 Bitmap using the given converter.
 * IMPORTANT: This function does NOT close the ImageProxy. Close it in the analyzer's finally{}.
 */
fun ImageProxy.toBitmap(converter: YuvToRgbConverter): Bitmap {
    val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    converter.yuvToRgb(this, bmp)
    return bmp
}
