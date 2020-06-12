package EntityMatching.PartitionMatching

import DataStructures.{IM, SpatialEntity}
import com.vividsolutions.jts.geom.IntersectionMatrix
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import utils.{Constants, Utils}
import utils.Readers.SpatialReader

case class PartitionMatching(joinedRDD: RDD[(Int, (Iterable[SpatialEntity],  Iterable[SpatialEntity]))],
                             thetaXY: (Double, Double), weightingScheme: String)  extends  PartitionMatchingTrait {

    /**
     * First index the source and then use the index to find the comparisons with target's entities.
     *
     * @param relation the examining relation
     * @return an RDD containing the matching pairs
     */
    def apply(relation: String): RDD[(String, String)] ={
        joinedRDD.flatMap { p =>
            val partitionId = p._1
            val source: Array[SpatialEntity] = p._2._1.toArray
            val target: Iterator[SpatialEntity] = p._2._2.toIterator
            val sourceIndex = index(source, partitionId)
            val filteringFunction = (b:(Int, Int)) => zoneCheck(partitionId)(b) && sourceIndex.contains(b)

            target
                .map(se => (se.index(thetaXY, filteringFunction) , se))
                .flatMap { case (coordsAr: Array[(Int, Int)], se: SpatialEntity) =>
                    coordsAr
                        .flatMap(c => sourceIndex.get(c).map(j => (source(j), se, c)))
                }
                .filter { case (e1: SpatialEntity, e2: SpatialEntity, b: (Int, Int)) =>
                    e1.mbb.testMBB(e2.mbb, relation) && e1.mbb.referencePointFiltering(e2.mbb, b, thetaXY)
                }
                .filter(c => c._1.relate(c._2, relation))
                .map(c => (c._1.originalID, c._2.originalID))
        }
    }

    def getDE9IM: RDD[IM] ={
        joinedRDD.flatMap { p =>
            val partitionId = p._1
            val source: Array[SpatialEntity] = p._2._1.toArray
            val target: Iterator[SpatialEntity] = p._2._2.toIterator
            val sourceIndex = index(source, partitionId)
            val filteringFunction = (b:(Int, Int)) => zoneCheck(partitionId)(b) && sourceIndex.contains(b)

            target
                .map(se => (se.index(thetaXY, filteringFunction) , se))
                .flatMap { case (coordsAr: Array[(Int, Int)], se: SpatialEntity) =>
                    coordsAr
                        .flatMap(c => sourceIndex.get(c).map(j => (source(j), se, c)))
                }
                .filter { case (e1: SpatialEntity, e2: SpatialEntity, b: (Int, Int)) =>
                    e1.mbb.testMBB(e2.mbb, Constants.INTERSECTS) && e1.mbb.referencePointFiltering(e2.mbb, b, thetaXY)
                }
                .map(c => IM(c._1, c._2))
        }
    }
}

/**
 * auxiliary constructor
 */
object PartitionMatching{

    def apply(source:RDD[SpatialEntity], target:RDD[SpatialEntity], thetaMsrSTR: String,
              weightingScheme: String = Constants.NO_USE): PartitionMatching ={
       val thetaXY = Utils.initTheta(source, target, thetaMsrSTR)
        val sourcePartitions = source.map(se => (TaskContext.getPartitionId(), se))
        val targetPartitions = target.map(se => (TaskContext.getPartitionId(), se))

        val joinedRDD = sourcePartitions.cogroup(targetPartitions, SpatialReader.spatialPartitioner)
        PartitionMatching(joinedRDD, thetaXY, weightingScheme)
    }
}
