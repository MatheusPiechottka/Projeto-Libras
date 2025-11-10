package com.example.libras

import android.graphics.PointF
import kotlin.math.*

/**
 * Mesmos passos do Python:
 * - Se esquerda: espelha X (x = 1-x) para frame Right
 * - Centraliza pela média dos pontos da palma [0,1,2,5,9]
 * - Escala por maior distância par-a-par em XY
 * - Alinha rotação: vetor (0→9) para +Y usando ang = atan2(vx, vy)
 * - Clipa [-2,2] e ACHATA INTERCALANDO (x0,y0,x1,y1,...,x20,y20) + hbit = 43-D
 */
object LandmarkFeaturizer {

    private val PALM = intArrayOf(0, 1, 2, 5, 9)

    fun build43CanonicalFrom01(pts01: List<PointF>, isLeft: Boolean): FloatArray {
        require(pts01.size == 21)

        // copia para double para ter estabilidade numérica
        val xy = Array(21) { i -> doubleArrayOf(pts01[i].x.toDouble(), pts01[i].y.toDouble()) }

        // espelha LEFT → RIGHT (x = 1 - x) — igual ao Python
        if (isLeft) {
            for (i in 0 until 21) xy[i][0] = 1.0 - xy[i][0]
        }

        // centraliza na palma
        var cx = 0.0; var cy = 0.0
        for (idx in PALM) { cx += xy[idx][0]; cy += xy[idx][1] }
        cx /= PALM.size; cy /= PALM.size
        for (i in 0 until 21) { xy[i][0] -= cx; xy[i][1] -= cy }

        // escala por max distancia par-a-par
        var dmax = 1e-6
        for (i in 0 until 21) {
            for (j in i+1 until 21) {
                val dx = xy[i][0] - xy[j][0]
                val dy = xy[i][1] - xy[j][1]
                val d = sqrt(dx*dx + dy*dy)
                if (d > dmax) dmax = d
            }
        }
        for (i in 0 until 21) { xy[i][0] /= dmax; xy[i][1] /= dmax }

        // rotação: (0→9) para +Y  —— MESMA FÓRMULA DO PYTHON: ang = atan2(vx, vy)
        val vx = xy[9][0] - xy[0][0]
        val vy = xy[9][1] - xy[0][1]
        val ang = atan2(vx, vy)         // ⚠️ ordem (vx, vy) — não (vy, vx)
        val ca = cos(-ang)
        val sa = sin(-ang)
        for (i in 0 until 21) {
            val x = xy[i][0]; val y = xy[i][1]
            xy[i][0] =  ca * x - sa * y
            xy[i][1] =  sa * x + ca * y
        }

        // ---- NOVO EMPACOTAMENTO: intercalar X,Y como no Python ----
        val feat = FloatArray(43)
        var k = 0
        for (i in 0 until 21) {
            feat[k++] = xy[i][0].coerceIn(-2.0, 2.0).toFloat()  // x_i
            feat[k++] = xy[i][1].coerceIn(-2.0, 2.0).toFloat()  // y_i
        }
        feat[42] = if (isLeft) 0f else 1f                       // handedness bit

        return feat
    }
}
