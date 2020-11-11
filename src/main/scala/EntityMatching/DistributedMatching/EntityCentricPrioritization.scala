package EntityMatching.DistributedMatching

import DataStructures.{IM, SpatialEntity}
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import utils.Constants.Relation
import utils.Constants.WeightStrategy.WeightStrategy
import utils.Utils

import scala.collection.mutable

case class EntityCentricPrioritization(joinedRDD: RDD[(Int, (Iterable[SpatialEntity], Iterable[SpatialEntity]))],
                                       thetaXY: (Double, Double), ws: WeightStrategy, budget: Long, sourceCount: Long)
   extends DMProgressiveTrait {


    /**
     * For each target entity we keep only the top K comparisons, according to a weighting scheme.
     * Then we assign to these top K comparisons, a common weight calculated based on the weights
     * of all the comparisons of the target entity. Based on this weight we prioritize their execution.
     *
     * @return  an RDD of Intersection Matrices
     */
    def getDE9IM: RDD[IM] ={
        joinedRDD
            .filter(p => p._2._1.nonEmpty && p._2._2.nonEmpty)
            .flatMap { p =>
                val pid = p._1
                val partition = partitionsZones(pid)
                val source: Array[SpatialEntity] = p._2._1.toArray
                val target: Array[SpatialEntity] = p._2._2.toArray
                val sourceIndex = index(source)
                val sourceSize = source.length
                val filterIndices = (b: (Int, Int)) => sourceIndex.contains(b)
                val filterRedundantComparisons = (i: Int, j: Int) => source(i).partitionRF(target(j).mbb, thetaXY, partition) &&
                    source(i).testMBB(target(j), Relation.INTERSECTS)

                val entityPQ = mutable.PriorityQueue[(Double, Int)]()(Ordering.by[(Double, Int), Double](_._1).reverse)
                val partitionPQ = mutable.PriorityQueue[(Double, (Iterator[Int], SpatialEntity))]()(Ordering.by[(Double, (Iterator[Int], SpatialEntity)), Double](_._1))

                val localBudget: Int = ((sourceSize*budget)/sourceCount).toInt
                val k = localBudget / p._2._2.size
                var minW = 10000d

                target
                    .indices
                    .foreach {j =>
                        var wSum = 0d
                        val e2 = target(j)
                        e2.index(thetaXY, filterIndices)
                            .foreach { c =>
                                sourceIndex.get(c)
                                    .filter(i => filterRedundantComparisons(i, j))
                                    .foreach { i =>
                                        val e1 = source(i)
                                        val w = getWeight(e1, e2)
                                        wSum += w
                                        // keep the top-K for each target entity
                                        if (entityPQ.size < k) {
                                            if (w < minW) minW = w
                                            entityPQ.enqueue((w, i))
                                        }
                                        else if (w > minW) {
                                            entityPQ.dequeue()
                                            entityPQ.enqueue((w, i))
                                            minW = entityPQ.head._1
                                        }
                                    }
                            }
                        if (entityPQ.nonEmpty) {
                            val weight = wSum / entityPQ.length
                            val topK = entityPQ.dequeueAll.map(_._2).reverse.toIterator
                            partitionPQ.enqueue((weight, (topK, e2)))
                            entityPQ.clear()
                        }
                    }

                partitionPQ.dequeueAll.map(_._2).flatMap{ case(sIndices, e2) => sIndices.map(i => IM(source(i), e2))}
            }
        }


    def getWeightedDE9IM: RDD[(Double, IM)] ={
        joinedRDD
            .filter(p => p._2._1.nonEmpty && p._2._2.nonEmpty)
            .flatMap { p =>
                val pid = p._1
                val partition = partitionsZones(pid)
                val source: Array[SpatialEntity] = p._2._1.toArray
                val target: Array[SpatialEntity] = p._2._2.toArray
                val sourceIndex = index(source)
                val sourceSize = source.length
                val filterIndices = (b: (Int, Int)) => sourceIndex.contains(b)
                val filterRedundantComparisons = (i: Int, j: Int) => source(i).partitionRF(target(j).mbb, thetaXY, partition) &&
                    source(i).testMBB(target(j), Relation.INTERSECTS)

                val innerPQ = mutable.PriorityQueue[(Double, Int)]()(Ordering.by[(Double, Int), Double](_._1).reverse)
                val partitionPQ = mutable.PriorityQueue[(Double, (Iterator[Int], SpatialEntity))]()(Ordering.by[(Double, (Iterator[Int], SpatialEntity)), Double](_._1))

                val localBudget: Int = ((sourceSize*budget)/sourceCount).toInt
                val k = localBudget / p._2._2.size
                var minW = 10000d
                target
                    .indices
                    .foreach {j =>
                        var wSum = 0d
                        val e2 = target(j)
                        e2.index(thetaXY, filterIndices)
                            .foreach { c =>
                                sourceIndex.get(c)
                                    .filter(i => filterRedundantComparisons(i, j))
                                    .foreach { i =>
                                        val e1 = source(i)
                                        val w = getWeight(e1, e2)
                                    wSum += w
                                    // keep the top-K for each target entity
                                    if (innerPQ.size < k) {
                                        if(w < minW) minW = w
                                        innerPQ.enqueue((w, i))
                                    }
                                    else if(w > minW) {
                                        innerPQ.dequeue()
                                        innerPQ.enqueue((w, i))
                                        minW = innerPQ.head._1
                                    }
                                }
                            }
                          if (innerPQ.nonEmpty) {
                            val weight = wSum / innerPQ.length
                            val topK = innerPQ.dequeueAll.map(_._2).reverse.toIterator
                            partitionPQ.enqueue((weight, (topK, e2)))
                            innerPQ.clear()
                        }
                    }

                partitionPQ.dequeueAll.flatMap{ case(w, (sIndices, e2)) => sIndices.map(i => (w, IM(source(i), e2)))}
            }
    }
}


object EntityCentricPrioritization{

    def apply(source:RDD[(Int, SpatialEntity)], target:RDD[(Int, SpatialEntity)], ws: WeightStrategy, budget: Long, partitioner: Partitioner)
    : EntityCentricPrioritization ={
        val thetaXY = Utils.getTheta
        val sourceCount = Utils.getSourceCount
        val joinedRDD = source.cogroup(target, partitioner)
        EntityCentricPrioritization(joinedRDD, thetaXY, ws, budget, sourceCount)
    }
}