package experiments

import java.util.Calendar

import EntityMatching.DistributedMatching.{GIAnt, ProgressiveGIAnt, ReciprocalTopK, TopKPairs}
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator
import utils.Constants.MatchingAlgorithm.MatchingAlgorithm
import utils.Constants.{GridType, MatchingAlgorithm, WeightStrategy}
import utils.Constants.WeightStrategy.WeightStrategy
import utils.{ConfigurationParser, SpatialReader, Utils}

object RocCurveExp {

    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.ERROR)
        Logger.getLogger("akka").setLevel(Level.ERROR)
        val log = LogManager.getRootLogger
        log.setLevel(Level.INFO)

        val sparkConf = new SparkConf()
            .setAppName("DS-JedAI")
            .set("spark.serializer", classOf[KryoSerializer].getName)
            .set("spark.kryo.registrator", classOf[GeoSparkKryoRegistrator].getName)

        val sc = new SparkContext(sparkConf)
        val spark: SparkSession = SparkSession.builder().getOrCreate()

        // Parsing input arguments
        @scala.annotation.tailrec
        def nextOption(map: OptionMap, list: List[String]): OptionMap = {
            list match {
                case Nil => map
                case ("-c" | "-conf") :: value :: tail =>
                    nextOption(map ++ Map("conf" -> value), tail)
                case ("-f" | "-fraction") :: value :: tail =>
                    nextOption(map ++ Map("fraction" -> value), tail)
                case ("-s" | "-stats") :: tail =>
                    nextOption(map ++ Map("stats" -> "true"), tail)
                case "-auc" :: tail =>
                    nextOption(map ++ Map("auc" -> "true"), tail)
                case ("-p" | "-partitions") :: value :: tail =>
                    nextOption(map ++ Map("partitions" -> value), tail)
                case ("-b" | "-budget") :: value :: tail =>
                    nextOption(map ++ Map("budget" -> value), tail)
                case "-ws" :: value :: tail =>
                    nextOption(map ++ Map("ws" -> value), tail)
                case "-ma" :: value :: tail =>
                    nextOption(map ++ Map("ma" -> value), tail)
                case "-gt" :: value :: tail =>
                    nextOption(map ++ Map("gt" -> value), tail)
                case _ :: tail =>
                    log.warn("DS-JEDAI: Unrecognized argument")
                    nextOption(map, tail)
            }
        }

        val argList = args.toList
        type OptionMap = Map[String, String]
        val options = nextOption(Map(), argList)

        if (!options.contains("conf")) {
            log.error("DS-JEDAI: No configuration file!")
            System.exit(1)
        }

        val confPath = options("conf")
        val conf = ConfigurationParser.parse(confPath)
        val partitions: Int = if (options.contains("partitions")) options("partitions").toInt else conf.getPartitions
        val budget: Int = if (options.contains("budget")) options("budget").toInt else conf.getBudget
        val ws: WeightStrategy = if (options.contains("ws")) WeightStrategy.withName(options("ws")) else conf.getWeightingScheme
        val ma: MatchingAlgorithm = if (options.contains("ma")) MatchingAlgorithm.withName(options("ma")) else conf.getMatchingAlgorithm
        val gridType: GridType.GridType = if (options.contains("gt")) GridType.withName(options("gt").toString) else conf.getGridType
        val relation = conf.getRelation

        log.info("DS-JEDAI: Input Budget: " + budget)
        log.info("DS-JEDAI: Weighting Strategy: " + ws.toString)
        val startTime = Calendar.getInstance().getTimeInMillis

        val reader = SpatialReader(conf.source, partitions, gridType)
        val sourceRDD = reader.load()
        sourceRDD.persist(StorageLevel.MEMORY_AND_DISK)
        Utils(sourceRDD.map(_._2.mbb), conf.getTheta, reader.partitionsZones)
        log.info(s"DS-JEDAI: Source was loaded into ${sourceRDD.getNumPartitions} partitions")

        val targetRDD = reader.load(conf.target)
        val partitioner = reader.partitioner


        val (_, _, _, _, _, _, _, _, _, totalVerifications, totalQualifiedPairs) = GIAnt(sourceRDD, targetRDD, ws, budget, partitioner).countAllRelations

        log.info("DS-JEDAI: Total Verifications: " + totalVerifications)
        log.info("DS-JEDAI: Total Interlinked Geometries: " + totalQualifiedPairs)
        log.info("\n")

        val (aucGiant, totalInterlinkedGeometriesGiant, totalVerifiedPairsGiant, (verifiedPairsGiant, qualifiedPairsGiant)) =
            GIAnt(sourceRDD, targetRDD, ws, budget, partitioner).getAUC(relation, 20)
        log.info("DS-JEDAI: GIANT Total Verifications: " + totalVerifiedPairsGiant)
        log.info("DS-JEDAI: GIANT Interlinked Geometries: " + totalInterlinkedGeometriesGiant)
        log.info("DS-JEDAI: GIANT AUC: " + aucGiant)
        log.info("DS-JEDAI: RECIPROCAL_TOPK Qualified Pairs\tVerified Pairs: " + qualifiedPairsGiant.zip(verifiedPairsGiant)
            .map{ case (qp: Int, vp: Int) => qp.toDouble/totalQualifiedPairs.toDouble +"\t"+vp.toDouble/totalVerifiedPairsGiant.toDouble}
            .mkString("\n"))
        log.info("\n")

        val (aucPG, totalInterlinkedGeometriesPG, totalVerifiedPairsPG, (verifiedPairsPG, qualifiedPairsPG)) =
            ProgressiveGIAnt(sourceRDD, targetRDD, ws, budget, partitioner).getAUC(relation, 20)
        log.info("DS-JEDAI: PROGRESSIVE_GIANT Total Verifications: " + totalVerifiedPairsPG)
        log.info("DS-JEDAI: PROGRESSIVE_GIANT Interlinked Geometries: " + totalInterlinkedGeometriesPG)
        log.info("DS-JEDAI: PROGRESSIVE_GIANT AUC: " + aucPG)
        log.info("DS-JEDAI: RECIPROCAL_TOPK Qualified Pairs\tVerified Pairs: " + qualifiedPairsPG.zip(verifiedPairsPG)
            .map{ case (qp: Int, vp: Int) => qp.toDouble/totalQualifiedPairs.toDouble +"\t"+vp.toDouble/totalVerifiedPairsPG.toDouble}
            .mkString("\n"))

        val (aucTopK, totalInterlinkedGeometriesTopK, totalVerifiedPairsTopK, (verifiedPairsTopK, qualifiedPairsTopK)) =
            TopKPairs(sourceRDD, targetRDD, ws, budget, partitioner).getAUC(relation, 20)
        log.info("DS-JEDAI: TOPK Total Verifications: " + totalVerifiedPairsTopK)
        log.info("DS-JEDAI: TOPK Interlinked Geometries: " + totalInterlinkedGeometriesTopK)
        log.info("DS-JEDAI: TOPK AUC: " + aucTopK)
        log.info("DS-JEDAI: RECIPROCAL_TOPK Qualified Pairs\tVerified Pairs: " + qualifiedPairsTopK.zip(verifiedPairsTopK)
            .map{ case (qp: Int, vp: Int) => qp.toDouble/totalQualifiedPairs.toDouble +"\t"+vp.toDouble/totalVerifiedPairsTopK.toDouble}
            .mkString("\n"))
        log.info("\n")

        val (aucRTopK, totalInterlinkedGeometriesTRopK, totalVerifiedPairsRTopK, (verifiedPairsRTopK, qualifiedPairsRTopK)) =
            ReciprocalTopK(sourceRDD, targetRDD, ws, budget, partitioner).getAUC(relation, 20)
        log.info("DS-JEDAI: RECIPROCAL_TOPK Total Verifications: " + totalVerifiedPairsRTopK)
        log.info("DS-JEDAI: RECIPROCAL_TOPK Interlinked Geometries: " + totalInterlinkedGeometriesTRopK)
        log.info("DS-JEDAI: RECIPROCAL_TOPK AUC: " + aucRTopK)
        log.info("DS-JEDAI: RECIPROCAL_TOPK Qualified Pairs\tVerified Pairs: " + qualifiedPairsRTopK.zip(verifiedPairsRTopK)
            .map{ case (qp: Int, vp: Int) => qp.toDouble/totalQualifiedPairs.toDouble +"\t"+vp.toDouble/totalVerifiedPairsRTopK.toDouble}
            .mkString("\n"))
        log.info("\n")

        val (aucGC, totalInterlinkedGeometriesGC, totalVerifiedPairsGC, (verifiedPairsGC, qualifiedPairsGC)) =
            ReciprocalTopK(sourceRDD, targetRDD, ws, budget, partitioner).getAUC(relation, 20)
        log.info("DS-JEDAI: GeometryCentric Total Verifications: " + totalVerifiedPairsGC)
        log.info("DS-JEDAI: GeometryCentric Interlinked Geometries: " + totalInterlinkedGeometriesGC)
        log.info("DS-JEDAI: GeometryCentric AUC: " + aucGC)
        log.info("DS-JEDAI: RECIPROCAL_TOPK Qualified Pairs\tVerified Pairs: " + qualifiedPairsGC.zip(verifiedPairsGC)
            .map{ case (qp: Int, vp: Int) => qp.toDouble/totalQualifiedPairs.toDouble +"\t"+vp.toDouble/totalVerifiedPairsGC.toDouble}
            .mkString("\n"))
        log.info("\n")

    }

}
