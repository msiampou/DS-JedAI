package experiments.progressive

import linkers.progressive.DistributedProgressiveInterlinking
import model.TileGranularities
import model.approximations.{GeometryApproximationT, GeometryToApproximation}
import model.entities.{EntityT, GeometryToEntity}
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.sedona.core.serde.SedonaKryoRegistrator
import org.apache.sedona.core.spatialRDD.SpatialRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.locationtech.jts.geom.Geometry
import utils.configuration.ConfigurationParser
import utils.configuration.Constants.EntityTypeENUM.EntityTypeENUM
import utils.configuration.Constants.GeometryApproximationENUM.GeometryApproximationENUM
import utils.configuration.Constants.GridType
import utils.configuration.Constants.ProgressiveAlgorithm.ProgressiveAlgorithm
import utils.readers.{GridPartitioner, Reader}

import java.util.Calendar

object SupervisedExp {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val log = LogManager.getRootLogger
    log.setLevel(Level.INFO)

    val sparkConf = new SparkConf()
      .setAppName("DS-JedAI")
      .set("spark.serializer", classOf[KryoSerializer].getName)
      .set("spark.kryo.registrator", classOf[SedonaKryoRegistrator].getName)

    val sc = new SparkContext(sparkConf)
    val spark: SparkSession = SparkSession.builder().getOrCreate()

    val parser = new ConfigurationParser()
    val configurationOpt = parser.parse(args) match {
      case Left(errors) =>
        errors.foreach(e => log.error(e.getMessage))
        System.exit(1)
        None
      case Right(configuration) => Some(configuration)
    }
    val conf = configurationOpt.get
    conf.printSupervised(log)

    val partitions: Int = conf.getPartitions
    val gridType: GridType.GridType = conf.getGridType
    val budget: Int = conf.getBudget
    val iter: Int = conf.getIterations
    val progressiveAlg: ProgressiveAlgorithm = conf.getProgressiveAlgorithm
    val entityTypeType: EntityTypeENUM = conf.getEntityType
    val approximationTypeOpt: Option[GeometryApproximationENUM] = conf.getApproximationType
    val decompositionT: Option[Double] = conf.getDecompositionThreshold

    // load datasets
    val sourceSpatialRDD: SpatialRDD[Geometry] = Reader.read(conf.source)
    val targetSpatialRDD: SpatialRDD[Geometry] = Reader.read(conf.target)

    val partitioner = GridPartitioner(sourceSpatialRDD, partitions, gridType)
    val approximateSourceCount = partitioner.approximateCount
    val theta = TileGranularities(sourceSpatialRDD.rawSpatialRDD.rdd.map(_.getEnvelopeInternal), approximateSourceCount, conf.getTheta)

    // spatial partition
    val decompositionTheta = decompositionT.map(dt => theta*dt)
    val approximationTransformerOpt: Option[Geometry => GeometryApproximationT] = GeometryToApproximation.getTransformer(approximationTypeOpt, decompositionTheta.getOrElse(theta))
    val geometry2entity: Geometry => EntityT = GeometryToEntity.getTransformer(entityTypeType, theta, decompositionTheta, None, approximationTransformerOpt)
    val sourceRDD: RDD[(Int, EntityT)] = partitioner.distributeAndTransform(sourceSpatialRDD, geometry2entity)
    val targetRDD: RDD[(Int, EntityT)] = partitioner.distributeAndTransform(targetSpatialRDD, geometry2entity)
    sourceRDD.persist(StorageLevel.MEMORY_AND_DISK)
    val sourceCount = sourceRDD.count()

    val partitionBorders = partitioner.getPartitionsBorders(theta)
    log.info(s"DS-JEDAI: Source was loaded into ${sourceRDD.getNumPartitions} partitions")

//    val matchingStartTime = Calendar.getInstance().getTimeInMillis
//    val linkers = DistributedProgressiveInterlinking.initializeProgressiveLinkers(sourceRDD, targetRDD,
//      partitionBorders, theta, partitioner, progressiveAlg, budget, sourceCount)
  }
}
