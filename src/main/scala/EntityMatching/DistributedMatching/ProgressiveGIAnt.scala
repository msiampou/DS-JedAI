package EntityMatching.DistributedMatching


import DataStructures.{ComparisonPQ, Entity, MBB}
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import utils.Constants.Relation.Relation
import utils.Constants.WeightStrategy.WeightStrategy
import utils.Utils


case class ProgressiveGIAnt(joinedRDD: RDD[(Int, (Iterable[Entity], Iterable[Entity]))],
                            thetaXY: (Double, Double), ws: WeightStrategy, budget: Int, sourceCount: Long) extends DMProgressiveTrait {


    /**
     * First index source and then for each entity of target, find its comparisons using source's index.
     * Weight the comparisons according to the input weighting scheme and sort them using a PQ.
     *
     * @param partition the MBB of the partition
     * @param source source
     * @param target target
     * @return a PQ with the top comparisons
     */
    def prioritize(source: Array[Entity], target: Array[Entity], partition: MBB, relation: Relation): ComparisonPQ[(Int, Int)] ={
        val sourceIndex = index(source)
        val filterIndices = (b: (Int, Int)) => sourceIndex.contains(b)
        val pq: ComparisonPQ[(Int, Int)] = ComparisonPQ[(Int, Int)](budget)

        // weight and put the comparisons in a PQ
        target
            .indices
            .foreach {j =>
                val e2 = target(j)
                e2.index(thetaXY, filterIndices)
                    .foreach { block =>
                        sourceIndex.get(block)
                            .filter(i => source(i).filter(e2, relation, block, thetaXY, Some(partition)))
                            .foreach { i =>
                                val e1 = source(i)
                                val w = getWeight(e1, e2)
                                pq.enqueue(w, (i,j))
                            }
                    }
            }
        pq
    }
}



/**
 * auxiliary constructor
 */
object ProgressiveGIAnt {

    def apply(source:RDD[(Int, Entity)], target:RDD[(Int, Entity)], ws: WeightStrategy, budget: Int, partitioner: Partitioner): ProgressiveGIAnt ={
        val thetaXY = Utils.getTheta
        val sourceCount = Utils.getSourceCount
        val joinedRDD = source.cogroup(target, partitioner)
        ProgressiveGIAnt(joinedRDD, thetaXY, ws, budget, sourceCount)
    }

}
