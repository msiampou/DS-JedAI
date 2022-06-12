package experiments.progressive

import linkers.progressive.DistributedProgressiveInterlinking
import model.TileGranularities
import model.approximations.{GeometryApproximationT, GeometryToApproximation}
import model.entities.{EntityT, GeometryToEntity}
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.sedona.core.serde.SedonaKryoRegistrator
import org.apache.sedona.core.spatialRDD.SpatialRDD
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.storage.StorageLevel
import org.locationtech.jts.geom.Geometry
import utils.configuration.{ConfigurationParser, Constants}
import utils.configuration.Constants.EntityTypeENUM.EntityTypeENUM
import utils.configuration.Constants.GeometryApproximationENUM.GeometryApproximationENUM
import utils.configuration.Constants.{GridType, ProgressiveAlgorithm, Relation, WeightingFunction}
import utils.configuration.Constants.ProgressiveAlgorithm.ProgressiveAlgorithm
import utils.configuration.Constants.WeightingFunction.WeightingFunction
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
    val spark: SparkSession = SparkSession.builder().appName("JEDAI-SUPERVISED_FILTERING").getOrCreate()

    val parser = new ConfigurationParser()
    val configurationOpt = parser.parse(args) match {
      case Left(errors) =>
        errors.foreach(e => log.error(e.getMessage))
        System.exit(1)
        None
      case Right(configuration) => Some(configuration)
    }
    val conf = configurationOpt.get
    conf.printProgressive(log)

    val partitions: Int = conf.getPartitions
    val gridType: GridType.GridType = conf.getGridType
    val budget: Int = conf.getBudget
    val progressiveAlg: ProgressiveAlgorithm = conf.getProgressiveAlgorithm
    val entityTypeType: EntityTypeENUM = conf.getEntityType
    val approximationTypeOpt: Option[GeometryApproximationENUM] = conf.getApproximationType
    val decompositionT: Option[Double] = conf.getDecompositionThreshold

    val weightingScheme: Constants.WeightingScheme =
      if (progressiveAlg == ProgressiveAlgorithm.EARLY_STOPPING) Constants.THIN_MULTI_COMPOSITE
      else if (progressiveAlg == ProgressiveAlgorithm.SUPERVISED) Constants.SIMPLE
      else conf.getWS

    val mainWF: WeightingFunction =
      if (progressiveAlg == ProgressiveAlgorithm.EARLY_STOPPING) WeightingFunction.JS
      else if (progressiveAlg == ProgressiveAlgorithm.SUPERVISED) WeightingFunction.LR
      else conf.getMainWF

    val secondaryWF: Option[WeightingFunction] =
      if (progressiveAlg == ProgressiveAlgorithm.EARLY_STOPPING) Some(WeightingFunction.JS)
      else if (progressiveAlg == ProgressiveAlgorithm.SUPERVISED) None
      else conf.getSecondaryWF

    val timeExp: Boolean = conf.measureStatistic
    val relation = conf.getRelation

    val batchSize = conf.getBatchSize
    val violations = conf.getViolations
    val precisionLimit = conf.getPrecisionLimit
    val startTime = Calendar.getInstance().getTimeInMillis

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

    log.info(s"DS-JEDAI: Initializing linkers..")
    val linkers = DistributedProgressiveInterlinking.initializeProgressiveLinkers(sourceRDD, targetRDD,
      partitionBorders, theta, partitioner, progressiveAlg, budget, sourceCount, weightingScheme,
      mainWF, secondaryWF, batchSize, violations, precisionLimit)
    if (timeExp) {
      // invoke load of target
      targetRDD.count()
      val expTime = DistributedProgressiveInterlinking.supervisedTime(linkers)
      val preprocessingTime = expTime._1
      val trainTime = expTime._2
      val verificationTime = expTime._3

      log.info(s"DS-JEDAI: Preprocessing time: $preprocessingTime")
      log.info(s"DS-JEDAI: Train time: $trainTime")
      log.info(s"DS-JEDAI: Verification Time: $verificationTime")

    } else {
      val trainRDD = DistributedProgressiveInterlinking.supervisedTrain(linkers)
      val (totalContains, totalCoveredBy, totalCovers, totalCrosses, totalEquals, totalIntersects,
      totalOverlaps, totalTouches, totalWithin, verifications, qp) = DistributedProgressiveInterlinking.countAllRelations(trainRDD)

      val totalRelations = totalContains + totalCoveredBy + totalCovers + totalCrosses + totalEquals +
                           totalIntersects + totalOverlaps + totalTouches + totalWithin
      log.info("DS-JEDAI: Total Verifications: " + verifications)
      log.info("DS-JEDAI: Qualifying Pairs : " + qp)

      log.info("DS-JEDAI: CONTAINS: " + totalContains)
      log.info("DS-JEDAI: COVERED BY: " + totalCoveredBy)
      log.info("DS-JEDAI: COVERS: " + totalCovers)
      log.info("DS-JEDAI: CROSSES: " + totalCrosses)
      log.info("DS-JEDAI: EQUALS: " + totalEquals)
      log.info("DS-JEDAI: INTERSECTS: " + totalIntersects)
      log.info("DS-JEDAI: OVERLAPS: " + totalOverlaps)
      log.info("DS-JEDAI: TOUCHES: " + totalTouches)
      log.info("DS-JEDAI: WITHIN: " + totalWithin)
      log.info("DS-JEDAI: Total Relations Discovered: " + totalRelations)
    }
//    // Experiments using MLib
//    val flatRDD = trainSets.collect().flatten
//    val rowRDD = spark.sparkContext.parallelize(flatRDD)
//    val PrEndTime = Calendar.getInstance().getTimeInMillis
//    log.info("DS-JEDAI: Preprocessing Time: " + (PrEndTime - PrStartTime) / 1000.0)
//
//    log.info(s"DS-JEDAI: Starting Supervised Train..")
//    val TrainStartTime = Calendar.getInstance().getTimeInMillis
//    val predictions = DistributedProgressiveInterlinking.supervisedTrain(rowRDD, spark)
//    log.info(trainSets.count() + "," + rowRDD.count())
//    val TrainEndTime = Calendar.getInstance().getTimeInMillis
//    log.info("DS-JEDAI: Training Time: " + (TrainEndTime - TrainStartTime) / 1000.0)

  }
}
