package model.weightedPairs

import model.TileGranularities
import model.entities.EntityT
import org.apache.commons.math3.stat.inference.ChiSquareTest
import utils.configuration.Constants.WeightingFunction.WeightingFunction
import utils.configuration.Constants._

import scala.util.Try


case class WeightedPairFactory(mainWF: WeightingFunction, secondaryWF: Option[WeightingFunction],
                               weightingScheme: WeightingScheme, theta: TileGranularities, totalBlocks: Double) {

    /**
     * compute the main weight of a pair of entities
     * @param s source entity
     * @param t target entity
     * @return a weight
     */
    def getMainWeight(s: EntityT, t: EntityT): Float = getWeight(s, t, mainWF)


    /**
     * compute the secondary weight of a pair of entities, if the secondary scheme was provided
     * @param s source entity
     * @param t target entity
     * @return a weight
     */
    def getSecondaryWeight(s: EntityT, t: EntityT): Float = secondaryWF match {
        case Some(wf) => getWeight(s, t, wf)
        case None => 0f
    }

    def createWeightedPair(counter: Int, s: EntityT, sIndex: Int, t:EntityT, tIndex: Int, prob: Float = 0f): WeightedPairT = {
        weightingScheme match {
            case SIMPLE =>
                val mw = WeightingFunction match {
                    case WeightingFunction.LR => prob
                    case _ => getMainWeight(s, t)
                }
                SimpleWP(counter, sIndex, tIndex, mw)
            case COMPOSITE =>
                val mw = getMainWeight(s, t)
                val sw = getSecondaryWeight(s, t)
                CompositeWP(counter, sIndex, tIndex, mw, sw)
            case HYBRID =>
                val mw = getMainWeight(s, t)
                val sw = getSecondaryWeight(s, t)
                HybridWP(counter, sIndex, tIndex, mw, sw)
            case THIN_MULTI_COMPOSITE =>
                val mw = jaccardSimilarity(s, t)
                val sw = coOccurrenceFrequency(s, t)
                val lw = minimumBoundingRectangleOverlap(s, t)
                ThinMultiCompositePair(counter, sIndex, tIndex, mw, sw, lw)
        }
    }

    def safeDivision(x: Int, y: Int): Float = Try(x/y).toOption match {
        case Some(z) => z
        case None => 0f
    }

    def safeDivision(x: Double, y: Double): Float = Try(x/y).toOption match {
        case Some(z) => z.toFloat
        case None => 0f
    }

    /**
     * Weight a pair
     * @param s Spatial entity
     * @param t Spatial entity
     * @return weight
     */
    def getWeight(s: EntityT, t: EntityT, wf: WeightingFunction): Float = {
        wf match {
            case WeightingFunction.MBRO => minimumBoundingRectangleOverlap(s, t)
            case WeightingFunction.ISP => inversePointSum(s, t)
            case WeightingFunction.JS => jaccardSimilarity(s, t)
            case WeightingFunction.PEARSON_X2 => pearsonsX2(s, t)
            case WeightingFunction.CF | _ => coOccurrenceFrequency(s, t)
        }
    }

    def getNumOfOverlappingTiles(e: EntityT): Int = e.getNumOfOverlappingTiles(theta)

    def getNumOfCommonTiles(s: EntityT, t: EntityT): Int = s.getNumOfCommonTiles(t, theta)

    def coOccurrenceFrequency(s: EntityT, t: EntityT): Float = getNumOfCommonTiles(s, t).toFloat

    def jaccardSimilarity(s: EntityT, t: EntityT): Float ={
        val sBlocks = getNumOfOverlappingTiles(s)
        val tBlocks = getNumOfOverlappingTiles(t)
        val cb = getNumOfCommonTiles(s, t)
        safeDivision(cb, sBlocks + tBlocks - cb)
    }

    def pearsonsX2(s: EntityT, t: EntityT): Float ={
        val sBlocks = getNumOfOverlappingTiles(s)
        val tBlocks = getNumOfOverlappingTiles(t)
        val cb = getNumOfCommonTiles(s, t)
        val v1: Array[Long] = Array[Long](cb, (tBlocks - cb).toLong)
        val v2: Array[Long] = Array[Long]((sBlocks - cb).toLong, (totalBlocks - (v1(0) + v1(1) + (sBlocks - cb))).toLong)
        val chiTest = new ChiSquareTest()
        chiTest.chiSquare(Array(v1, v2)).toFloat
    }

    def minimumBoundingRectangleOverlap(s: EntityT, t: EntityT): Float ={
        val intersectionArea = s.getIntersectingInterior(t).getArea
        val w = safeDivision(intersectionArea, s.approximation.getArea + t.approximation.getArea - intersectionArea)
        if (!w.isNaN) w else 0f
    }

    def inversePointSum(s: EntityT, t: EntityT): Float = safeDivision(1f, s.geometry.getNumPoints + t.geometry.getNumPoints)
}