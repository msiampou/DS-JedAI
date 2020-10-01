package EntityMatching.PartitionMatching

import DataStructures.{IM, MBB, SpatialEntity}
import com.google.common.collect.MinMaxPriorityQueue
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import utils.Constants.Relation
import utils.Constants.ThetaOption.ThetaOption
import utils.Constants.WeightStrategy.WeightStrategy
import utils.Readers.SpatialReader
import utils.Utils

import scala.collection.mutable
import scala.collection.JavaConverters._


case class ReciprocalTopK(joinedRDD: RDD[(Int, (Iterable[SpatialEntity], Iterable[SpatialEntity]))],
                     thetaXY: (Double, Double), ws: WeightStrategy, budget: Long, sourceCount: Long) extends ProgressiveTrait {


    def compute(source: Array[SpatialEntity], target: Array[SpatialEntity], partition: MBB): MinMaxPriorityQueue[(Double, (Int, Int))] = {
        val sourceIndex = index(source)
        val filteringFunction = (b: (Int, Int)) => sourceIndex.contains(b)

        val localBudget: Int = ((source.length * budget) / sourceCount).toInt
        val k = math.ceil(localBudget / (source.length + target.length)).toInt * 2

        val orderingInt = Ordering.by[(Double, Int), Double](_._1).reverse
        val orderingPair = Ordering.by[(Double, (Int, Int)), Double](_._1).reverse

        val sourceMinWeightPQ: Array[Double] = Array.fill(source.length)(0d)
        val sourcePQ: Array[MinMaxPriorityQueue[(Double, Int)]] = new Array(source.length)

        val targetPQ: mutable.PriorityQueue[(Double, Int)] = mutable.PriorityQueue()(Ordering.by[(Double, Int), Double](_._1).reverse)
        val targetSet: Array[mutable.HashSet[Int]] = Array.fill(target.length)(new mutable.HashSet[Int]())
        var minW = 0d

        val partitionPQ: MinMaxPriorityQueue[(Double, (Int, Int))] = MinMaxPriorityQueue.orderedBy(orderingPair).maximumSize(localBudget + 1).create()
        var partitionMinWeight = 0d
        target
            .zipWithIndex
            .foreach { case (e2, j) =>
                val frequencies = e2.index(thetaXY, filteringFunction)
                    .flatMap(c => sourceIndex.get(c))
                    .groupBy(identity)
                    .mapValues(_.length)

                frequencies
                    .filter { case (i, _) => source(i).partitionRF(e2.mbb, thetaXY, partition) && source(i).testMBB(e2, Relation.INTERSECTS, Relation.TOUCHES) }
                    .foreach { case (i, f) =>
                        val e1 = source(i)
                        val w = getWeight(f, e1, e2)

                        // set top-K for each target entity
                        if (minW < w) {
                            targetPQ.enqueue((w, i))
                            if (targetPQ.size > localBudget)
                                minW = targetPQ.dequeue()._1
                        }

                        // update source entities' top-K
                        if (sourceMinWeightPQ(i) == 0)
                            sourcePQ(i) = MinMaxPriorityQueue.orderedBy(orderingInt).maximumSize(k + 1).create()
                        if (sourceMinWeightPQ(i) < w) {
                            sourcePQ(i).add((w, j))
                            if (sourcePQ(i).size > k)
                                sourceMinWeightPQ(i) = sourcePQ(i).pollLast()._1
                        }
                    }

                while (targetPQ.nonEmpty) targetSet(j).add(targetPQ.dequeue()._2)
            }

        sourcePQ
            .zipWithIndex
            .filter(_._1 != null)
            .foreach { case (pq, i) =>
                val w = Double.MaxValue
                while (pq.size > 0 && w > partitionMinWeight) {
                    val (w, j) = pq.pollFirst()
                    if (targetSet(j).contains(i))
                        if (partitionMinWeight < w) {
                            partitionPQ.add(w, (i, j))
                            if (partitionPQ.size() > localBudget)
                                partitionMinWeight = partitionPQ.pollLast()._1

                        }
                }
            }

        partitionPQ
    }

    def getDE9IM: RDD[IM] = joinedRDD
        .filter(p => p._2._1.nonEmpty && p._2._2.nonEmpty)
        .flatMap { p =>
            val pid = p._1
            val partition = partitionsZones(pid)
            val source: Array[SpatialEntity] = p._2._1.toArray
            val target: Array[SpatialEntity] = p._2._2.toArray

            compute(source, target, partition).iterator().asScala.map{ case(_, (i, j)) =>  IM(source(i), target(j))}
        }

    def getWeightedDE9IM: RDD[(Double, IM)] = joinedRDD.filter(p => p._2._1.nonEmpty && p._2._2.nonEmpty)
        .flatMap { p =>
            val pid = p._1
            val partition = partitionsZones(pid)
            val source: Array[SpatialEntity] = p._2._1.toArray
            val target: Array[SpatialEntity] = p._2._2.toArray

            compute(source, target, partition).iterator().asScala.map{ case(w, (i, j)) =>  (w, IM(source(i), target(j)))}
        }

}

object ReciprocalTopK{

    def apply(source:RDD[SpatialEntity], target:RDD[SpatialEntity], thetaOption: ThetaOption,
              ws: WeightStrategy, budget: Long): TopKPairs ={
        val thetaXY = Utils.getTheta
        val sourceCount = Utils.getSourceCount
        val sourcePartitions = source.map(se => (TaskContext.getPartitionId(), se))
        val targetPartitions = target.map(se => (TaskContext.getPartitionId(), se))

        val joinedRDD = sourcePartitions.cogroup(targetPartitions, SpatialReader.spatialPartitioner)
        TopKPairs(joinedRDD, thetaXY, ws, budget, sourceCount)
    }
}