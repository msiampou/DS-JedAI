package utils.Readers

import DataStructures.SpatialEntity
import org.apache.spark.rdd.RDD

/**
 * @author George Mandilaras < gmandi@di.uoa.gr > (National and Kapodistrian University of Athens)
 */
trait TReader {

  def load( filePath: String,
                    realID_field: String,
                    geometryField: String
                  ): RDD[SpatialEntity]
}