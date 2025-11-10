package com.example.libras

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlin.math.max

/**
 * MediaPipe HandLandmarker.
 * We rotate the bitmap ourselves (clockwise by CameraX rotation) and run MP with rotation=0.
 * We also return the rotated frame size so the overlay can map to the Preview correctly.
 */
class GestureBoxTracker(context: Context) {

    data class Hand(
        val landmarks01: List<PointF>, // 21 normalized [0..1]
        val isLeft: Boolean,
        val roiPx: Rect                // square ROI in *rotated* bitmap pixels
    )

    /** Rotated frame size included so the overlay can map with rotation=0. */
    data class BothHands(
        val left: Hand?,
        val right: Hand?,
        val frameW: Int,
        val frameH: Int
    )

    private val landmarker: HandLandmarker

    init {
        val base = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
        val opts = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.6f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setRunningMode(RunningMode.IMAGE)
            .build()
        landmarker = HandLandmarker.createFromOptions(context, opts)
    }

    /** Rotate bitmap clockwise by 0/90/180/270. */
    private fun rotateBitmapCW(src: Bitmap, deg: Int): Bitmap {
        val d = ((deg % 360) + 360) % 360
        if (d == 0) return src
        val m = Matrix().apply { postRotate(d.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    /** Detect up to 2 hands. Bitmap is pre-rotated; MP runs with rotation=0. */
    fun detectBoth(bmp: Bitmap, rotationDeg: Int): BothHands {
        val rotated = rotateBitmapCW(bmp, rotationDeg)

        val mpImg = BitmapImageBuilder(rotated).build()
        val ipo = ImageProcessingOptions.builder().setRotationDegrees(0).build()
        val res: HandLandmarkerResult = landmarker.detect(mpImg, ipo)

        val hands = mutableListOf<Hand>()
        val allLandmarks = res.landmarks()
        val allHanded = res.handednesses()

        for (i in allLandmarks.indices) {
            val lmList = allLandmarks[i]
            val pts01 = lmList.map { p -> PointF(p.x(), p.y()) }
            val handedCat = allHanded.getOrNull(i)?.firstOrNull()
            val isLeft = handedCat?.categoryName()?.equals("Left", ignoreCase = true) == true

            // ROI (normalized -> px) with square padding ~0.65
            val minX = pts01.minOf { it.x }
            val maxX = pts01.maxOf { it.x }
            val minY = pts01.minOf { it.y }
            val maxY = pts01.maxOf { it.y }

            val cx = (minX + maxX) / 2f
            val cy = (minY + maxY) / 2f
            val half = (max(maxX - minX, maxY - minY) * 0.65f)

            val leftPx   = ((cx - half) * rotated.width ).toInt().coerceIn(0, rotated.width  - 1)
            val topPx    = ((cy - half) * rotated.height).toInt().coerceIn(0, rotated.height - 1)
            val rightPx  = ((cx + half) * rotated.width ).toInt().coerceIn(leftPx + 1, rotated.width)
            val bottomPx = ((cy + half) * rotated.height).toInt().coerceIn(topPx  + 1, rotated.height)

            hands += Hand(pts01, isLeft, Rect(leftPx, topPx, rightPx, bottomPx))
        }

        val left  = hands.firstOrNull { it.isLeft }
        val right = hands.firstOrNull { !it.isLeft }
        return BothHands(left, right, frameW = rotated.width, frameH = rotated.height)
    }

    fun close() = runCatching { landmarker.close() }
}