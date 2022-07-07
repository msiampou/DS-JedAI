package experiments.progressive

import linkers.DistributedInterlinking
import linkers.progressive.DistributedProgressiveInterlinking
import model.TileGranularities
import model.entities.{EntityT, GeometryToEntity}
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.sedona.core.serde.SedonaKryoRegistrator
import org.apache.sedona.core.spatialRDD.SpatialRDD
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.locationtech.jts.geom.{Envelope, Geometry}
import utils.configuration.{ConfigurationParser, Constants}
import utils.configuration.Constants.EntityTypeENUM.EntityTypeENUM
import utils.configuration.Constants.{GridType, ProgressiveAlgorithm, Relation, WeightingFunction}
import utils.configuration.Constants.ProgressiveAlgorithm.ProgressiveAlgorithm
import utils.configuration.Constants.Relation.Relation
import utils.configuration.Constants.WeightingFunction.WeightingFunction
import utils.readers.{GridPartitioner, Reader}

object SupervisedExp {

  private val log: Logger = LogManager.getRootLogger
  log.setLevel(Level.INFO)

  val defaultBudget: Int = 3000
  var takeBudget: Seq[Int] = _
  val relation: Relation = Relation.DE9IM

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

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
    val inputBudget: Int = conf.getBudget
    val progressiveAlg: ProgressiveAlgorithm = conf.getProgressiveAlgorithm
    val budget = if (inputBudget > 0) inputBudget else defaultBudget
    this.takeBudget = Seq(budget)
    val entityTypeType: EntityTypeENUM = conf.getEntityType

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
    val evalExp: Boolean = conf.evalMetrics

    // load datasets
    val sourceSpatialRDD: SpatialRDD[Geometry] = Reader.read(conf.source)
    val targetSpatialRDD: SpatialRDD[Geometry] = Reader.read(conf.target)
    val partitioner = GridPartitioner(sourceSpatialRDD, partitions, gridType)
    val approximateSourceCount = partitioner.approximateCount
    val theta = TileGranularities(sourceSpatialRDD.rawSpatialRDD.rdd.map(
      _.getEnvelopeInternal), approximateSourceCount, conf.getTheta)

    // spatial partition
    val geometry2entity: Geometry => EntityT = GeometryToEntity.getTransformer(entityTypeType, theta, None, None, None)
    val sourceRDD: RDD[(Int, EntityT)] = partitioner.distributeAndTransform(sourceSpatialRDD, geometry2entity)
    val targetRDD: RDD[(Int, EntityT)] = partitioner.distributeAndTransform(targetSpatialRDD, geometry2entity)
    sourceRDD.persist(StorageLevel.MEMORY_AND_DISK)
    val partitionBorder = partitioner.getPartitionsBorders(theta)
    log.info(s"DS-JEDAI: Source was loaded into ${sourceRDD.getNumPartitions} partitions")

    if (timeExp) {
      // initialize linkers
      val linkers = DistributedProgressiveInterlinking.initializeProgressiveLinkers(sourceRDD, targetRDD,
        partitionBorder, theta, partitioner, progressiveAlg, budget, approximateSourceCount, weightingScheme,
        mainWF, secondaryWF)
      // invoke target execution
      targetRDD.count()
      // calculate time
      val expTime = DistributedProgressiveInterlinking.supervisedTime(linkers)
      val schedulingTime = expTime._1
      val verificationTime = expTime._2
      val matchingTime = expTime._3

      log.info(s"DS-JEDAI: Preprocessing time: $schedulingTime")
      log.info(s"DS-JEDAI: Train time: $verificationTime")
      log.info(s"DS-JEDAI: Verification Time: $matchingTime")

    } else if (evalExp) {
      // to compute recall and precision we need overall results
      val (totalVerifications, totalRelatedPairs) =
        (conf.getTotalVerifications, conf.getTotalQualifyingPairs) match {
          case (Some(tv), Some(qp)) =>
            (tv, qp)
          case _ =>
            val g = DistributedInterlinking.countAllRelations(
              DistributedInterlinking.initializeLinkers(sourceRDD, targetRDD, partitionBorder, theta, partitioner))
            (g._10, g._11)
        }

      log.info("DS-JEDAI: Total Verifications: " + totalVerifications)
      log.info("DS-JEDAI: Qualifying Pairs : " + totalRelatedPairs)

      val wf: (WeightingFunction, Option[WeightingFunction]) = (mainWF, secondaryWF)
      printEvaluationResults(sourceRDD, targetRDD, theta, partitionBorder, approximateSourceCount,
        partitioner, totalRelatedPairs, budget, progressiveAlg, wf, weightingScheme)
    } else {
      // initialize linkers
      val linkers = DistributedProgressiveInterlinking.initializeProgressiveLinkers(sourceRDD, targetRDD,
        partitionBorder, theta, partitioner, progressiveAlg, budget, approximateSourceCount, weightingScheme,
        mainWF, secondaryWF)
      // preprocess & train
      val trainRDD = DistributedProgressiveInterlinking.supervisedTrain(linkers)
      // count relations
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

  }
  def printEvaluationResults(sRDD: RDD[(Int, EntityT)], tRDD: RDD[(Int, EntityT)],
                             theta: TileGranularities, partitionBorders: Array[Envelope],
                             sourceCount: Long, partitioner: GridPartitioner,
                             totalRelations: Int, budget: Int, algorithm: ProgressiveAlgorithm,
                             wf: (WeightingFunction, Option[WeightingFunction]),
                             ws: Constants.WeightingScheme, n: Int = 10): Unit = {

    val linkers = DistributedProgressiveInterlinking.initializeProgressiveLinkers(sRDD, tRDD,
      partitionBorders, theta, partitioner, algorithm, budget, sourceCount, ws, wf._1, wf._2)

    val trainRDD = DistributedProgressiveInterlinking.supervisedTrain(linkers)
    log.info(s"DS-JEDAI: Train-End")
    val evaluation = DistributedProgressiveInterlinking.evaluate(algorithm, trainRDD, relation, n, totalRelations, takeBudget)
    log.info(s"DS-JEDAI: Evaluating")
    evaluation.zip(takeBudget).foreach { case ((pgr, qp, verifications, (_, _)), b) =>
      val qualifiedPairsWithinBudget = if (totalRelations < verifications) totalRelations else verifications
      log.info(s"DS-JEDAI: ${algorithm.toString} Budget : $b")
      log.info(s"DS-JEDAI: ${algorithm.toString} Total Verifications: $verifications")
      log.info(s"DS-JEDAI: ${algorithm.toString} Qualifying Pairs within budget: $qualifiedPairsWithinBudget")
      log.info(s"DS-JEDAI: ${algorithm.toString} Detected Qualifying Pairs: $qp")
      log.info(s"DS-JEDAI: ${algorithm.toString} Recall: ${qp.toDouble / qualifiedPairsWithinBudget.toDouble}")
      log.info(s"DS-JEDAI: ${algorithm.toString} Precision: ${qp.toDouble / verifications.toDouble}")
      log.info(s"DS-JEDAI: ${algorithm.toString} PGR: $pgr")
    }
  }
}
