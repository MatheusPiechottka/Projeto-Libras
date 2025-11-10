package com.example.libras

import android.graphics.Bitmap
import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min

class YuvToRgbConverter {

    fun yuvToRgb(image: ImageProxy, out: Bitmap) {
        require(image.format == ImageFormat.YUV_420_888) {
            "Unsupported ImageProxy format: ${image.format}"
        }

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixStride = yPlane.pixelStride // often 1, but not guaranteed

        val uRowStride = uPlane.rowStride
        val uPixStride = uPlane.pixelStride

        val vRowStride = vPlane.rowStride
        val vPixStride = vPlane.pixelStride

        val width = image.width
        val height = image.height

        val outPixels = IntArray(width * height)
        var outIndex = 0

        for (y in 0 until height) {
            val yRow = y * yRowStride
            val uvRow = (y shr 1) * uRowStride
            val vvRow = (y shr 1) * vRowStride

            for (x in 0 until width) {
                val yIdx = yRow + x * yPixStride
                val uIdx = uvRow + (x shr 1) * uPixStride
                val vIdx = vvRow + (x shr 1) * vPixStride

                val Y = (yBuf.getSafe(yIdx).toInt() and 0xFF)
                val U = (uBuf.getSafe(uIdx).toInt() and 0xFF) - 128
                val V = (vBuf.getSafe(vIdx).toInt() and 0xFF) - 128

                // BT.601 full-range conversion
                var r = (Y + 1.402f * V).toInt()
                var g = (Y - 0.344136f * U - 0.714136f * V).toInt()
                var b = (Y + 1.772f * U).toInt()

                if (r < 0) r = 0 else if (r > 255) r = 255
                if (g < 0) g = 0 else if (g > 255) g = 255
                if (b < 0) b = 0 else if (b > 255) b = 255

                outPixels[outIndex++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        out.setPixels(outPixels, 0, width, 0, 0, width, height)
    }

    /** Safe indexed read that never moves the buffer's position and clamps indices. */
    private fun ByteBuffer.getSafe(index: Int): Byte {
        val i = max(0, min(index, this.limit() - 1))
        return this[i]
    }
}
