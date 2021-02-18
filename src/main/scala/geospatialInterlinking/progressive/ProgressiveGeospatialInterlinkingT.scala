package geospatialInterlinking.progressive

import dataModel.{ComparisonPQ, Entity, IM, MBR}
import geospatialInterlinking.GeospatialInterlinkingT
import org.apache.commons.math3.stat.inference.ChiSquareTest
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import utils.Constants.Relation.Relation
import utils.Constants.WeightStrategy.WeightStrategy
import utils.Constants.{Relation, WeightStrategy}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.math.{ceil, floor, max, min}

trait ProgressiveGeospatialInterlinkingT extends GeospatialInterlinkingT{
    val budget: Int
    val ws: WeightStrategy

    val totalBlocks: Double = if (ws == WeightStrategy.PEARSON_X2){
        val globalMinX = joinedRDD.flatMap(p => p._2._1.map(_.mbr.minX/thetaXY._1)).min()
        val globalMaxX = joinedRDD.flatMap(p => p._2._1.map(_.mbr.maxX/thetaXY._1)).max()
        val globalMinY = joinedRDD.flatMap(p => p._2._1.map(_.mbr.minY/thetaXY._2)).min()
        val globalMaxY = joinedRDD.flatMap(p => p._2._1.map(_.mbr.maxY/thetaXY._2)).max()
        (globalMaxX - globalMinX + 1) * (globalMaxY - globalMinY + 1)
    } else -1

    def prioritize(source: Array[Entity], target: Array[Entity], partition: MBR, relation: Relation): ComparisonPQ[(Int, Int)]

    /**
     * Weight a comparison
     *
     * @param e1        Spatial entity
     * @param e2        Spatial entity
     * @return weight
     */
    def getWeight(e1: Entity, e2: Entity): Double = {
        val e1Blocks = (ceil(e1.mbr.maxX/thetaXY._1).toInt - floor(e1.mbr.minX/thetaXY._1).toInt + 1) * (ceil(e1.mbr.maxY/thetaXY._2).toInt - floor(e1.mbr.minY/thetaXY._2).toInt + 1).toDouble
        val e2Blocks = (ceil(e2.mbr.maxX/thetaXY._1).toInt - floor(e2.mbr.minX/thetaXY._1).toInt + 1) * (ceil(e2.mbr.maxY/thetaXY._2).toInt - floor(e2.mbr.minY/thetaXY._2).toInt + 1).toDouble
        val cb = (min(ceil(e1.mbr.maxX/thetaXY._1), ceil(e2.mbr.maxX/thetaXY._1)).toInt - max(floor(e1.mbr.minX/thetaXY._1), floor(e2.mbr.minX/thetaXY._1)).toInt + 1) *
            (min(ceil(e1.mbr.maxY/thetaXY._2), ceil(e2.mbr.maxY/thetaXY._2)).toInt - max(floor(e1.mbr.minY/thetaXY._2), floor(e2.mbr.minY/thetaXY._2)).toInt + 1)

        ws match {
            case WeightStrategy.MBR_INTERSECTION =>
                val intersectionArea = MBR(e1.geometry.getEnvelopeInternal.intersection(e2.geometry.getEnvelopeInternal)).getArea
                intersectionArea / (e1.mbr.getArea + e2.mbr.getArea - intersectionArea)

            case WeightStrategy.POINTS =>
                1d / (e1.geometry.getNumPoints + e2.geometry.getNumPoints);

            case WeightStrategy.JS =>
                cb / (e1Blocks + e2Blocks - cb)

            case WeightStrategy.PEARSON_X2 =>
                val v1: Array[Long] = Array[Long](cb, (e2Blocks - cb).toLong)
                val v2: Array[Long] = Array[Long]((e1Blocks - cb).toLong, (totalBlocks - (v1(0) + v1(1) + (e1Blocks - cb))).toLong)
                val chiTest = new ChiSquareTest()
                chiTest.chiSquare(Array(v1, v2))

            case WeightStrategy.CF | _ =>
                cb.toDouble
        }
    }

    /**
     *  Get the DE-9IM of the top most related entities based
     *  on the input budget and the Weighting strategy
     * @return an RDD of IM
     */
    def getDE9IM: RDD[IM] ={
        joinedRDD.filter(j => j._2._1.nonEmpty && j._2._2.nonEmpty)
            .flatMap{ p =>
            val pid = p._1
            val partition = partitionsZones(pid)
            val source = p._2._1.toArray
            val target = p._2._2.toArray

            val pq = prioritize(source, target, partition, Relation.DE9IM)
            if (!pq.isEmpty)
                pq.dequeueAll.map{ case (_, (i, j)) =>
                    val e1 = source(i)
                    val e2 = target(j)
                    IM(e1, e2)
                }.takeWhile(_ => !pq.isEmpty)
            else Iterator()
        }
    }


    /**
     *  Examine the Relation of the top most related entities based
     *  on the input budget and the Weighting strategy
     *  @param relation the relation to examine
     *  @return an RDD of pair of IDs
     */
    def relate(relation: Relation): RDD[(String, String)] = {
        joinedRDD.filter(j => j._2._1.nonEmpty && j._2._2.nonEmpty)
            .flatMap{ p =>
            val pid = p._1
            val partition = partitionsZones(pid)
            val source = p._2._1.toArray
            val target = p._2._2.toArray

            val pq = prioritize(source, target, partition, relation)
            if (!pq.isEmpty)
                pq.dequeueAll.map{ case (_, (i, j)) =>
                    val e1 = source(i)
                    val e2 = target(j)
                    (e1.relate(e2, relation), (e1.originalID, e2.originalID))
                }.filter(_._1).map(_._2)
            else Iterator()
        }
    }


    /**
     * Compute PGR - first weight and perform the comparisons in each partition,
     * then collect them in descending order and compute the progressive True Positives.
     *
     * @param relation the examined relation
     * @return (PGR, total interlinked Geometries (TP), total comparisons)
     */
    def evaluate(relation: Relation, n: Int = 10, totalQualifiedPairs: Double, takeBudget: Seq[Int]): Seq[(Double, Long, Long, (List[Int], List[Int]))]  ={
       // computes weighted the weighted comparisons
        val matches: RDD[(Double, Boolean)] = joinedRDD
            .filter(p => p._2._1.nonEmpty && p._2._2.nonEmpty)
            .flatMap { p =>
                val pid = p._1
                val partition = partitionsZones(pid)
                val source = p._2._1.toArray
                val target = p._2._2.toArray

                val pq = prioritize(source, target, partition, relation)
                if (!pq.isEmpty)
                    pq.dequeueAll.map{ case (w, (i, j)) =>
                        val e1 = source(i)
                        val e2 = target(j)
                        relation match {
                            case Relation.DE9IM => (w, IM(e1, e2).relate)
                            case _ => (w, e1.relate(e2, relation))
                        }
                    }.takeWhile(_ => !pq.isEmpty)
                else Iterator()
            }.persist(StorageLevel.MEMORY_AND_DISK)

        var results = mutable.ListBuffer[(Double, Long, Long, (List[Int], List[Int]))]()
        for(b <- takeBudget){
            // compute AUC prioritizing the comparisons based on their weight
            val sorted = matches.takeOrdered(b)(Ordering.by[(Double, Boolean), Double](_._1).reverse)
            val verifications = sorted.length
            val step = math.ceil(verifications/n)

            var progressiveQP: Double = 0
            var qp = 0
            val verificationSteps = ListBuffer[Int]()
            val qualifiedPairsSteps = ListBuffer[Int]()

            sorted
                .map(_._2)
                .zipWithIndex
                .foreach{
                    case (r, i) =>
                        if (r) qp += 1
                        progressiveQP += qp
                        if (i % step == 0){
                            qualifiedPairsSteps += qp
                            verificationSteps += i
                        }
                }
            qualifiedPairsSteps += qp
            verificationSteps += verifications
            val qualifiedPairsWithinBudget = if (totalQualifiedPairs < verifications) totalQualifiedPairs else verifications
            val pgr = (progressiveQP/qualifiedPairsWithinBudget)/verifications.toDouble
            results += ((pgr, qp, verifications, (verificationSteps.toList, qualifiedPairsSteps.toList)))
        }
        matches.unpersist()
        results
    }

}