package experiments

import java.util.Calendar

import EntityMatching.LightAlgorithms.{LightMatchingFactory, LightRADON}
import EntityMatching.LightAlgorithms.prioritization.{ComparisonCentricPrioritization, EntityCentricPrioritization}
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import utils.Readers.{CSVReader, Reader}
import utils.{ConfigurationParser, Constants}


/**
 * @author George Mandilaras < gmandi@di.uoa.gr > (National and Kapodistrian University of Athens)
 *
 *         Execution:
 *         		spark-submit --master local[*] --class experiments.LightExp target/scala-2.11/DS-JedAI-assembly-0.1.jar -conf <conf>
 *         Debug:
 *         		spark-submit --master local[*] --conf spark.driver.extraJavaOptions=-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005 --class experiments.LightExp target/scala-2.11/DS-JedAI-assembly-0.1.jar -conf <conf>
 */


object LightExp {

    def main(args: Array[String]): Unit = {
        val startTime = Calendar.getInstance().getTimeInMillis

        Logger.getLogger("org").setLevel(Level.ERROR)
        Logger.getLogger("akka").setLevel(Level.ERROR)
        val log = LogManager.getRootLogger
        log.setLevel(Level.INFO)

        val sparkConf = new SparkConf()
            .setAppName("SD-JedAI")
            .set("spark.serializer", classOf[KryoSerializer].getName)
        val sc = new SparkContext(sparkConf)
        val spark: SparkSession = SparkSession.builder().getOrCreate()

        // Parsing the input arguments
        @scala.annotation.tailrec
        def nextOption(map: OptionMap, list: List[String]): OptionMap = {
            list match {
                case Nil => map
                case ("-c" | "-conf") :: value :: tail =>
                    nextOption(map ++ Map("conf" -> value), tail)
                case _ :: tail =>
                    log.warn("DS-JEDAI: Unrecognized argument")
                    nextOption(map, tail)
            }
        }

        val arglist = args.toList
        type OptionMap = Map[String, String]
        val options = nextOption(Map(), arglist)

        if (!options.contains("conf")) {
            log.error("DS-JEDAI: No configuration file!")
            System.exit(1)
        }

        val conf_path = options("conf")
        val conf = ConfigurationParser.parse(conf_path)
        val partitions: Int = conf.configurations.getOrElse(Constants.CONF_PARTITIONS, "0").toInt
        val spatialPartition: Boolean = conf.configurations.getOrElse(Constants.CONF_SPATIAL_PARTITION, "false").toBoolean


        val sourceRDD = Reader.read(conf.source.path, conf.source.realIdField, conf.source.geometryField, spatialPartition)
        val sourceCount = sourceRDD.setName("SourceRDD").cache().count()
        log.info("DS-JEDAI: Number of ptofiles of Source: " + sourceCount)

        // Loading Target
        val targetRDD = Reader.read(conf.target.path, conf.source.realIdField, conf.source.geometryField, spatialPartition)
        val targetCount = targetRDD.setName("TargetRDD").cache().count()
        log.info("DS-JEDAI: Number of ptofiles of Target: " + targetCount)

        val matches = LightMatchingFactory.getMatchingAlgorithm(conf, sourceRDD, targetRDD).apply(0, conf.relation) // TODO fix this..there is no indexSeparator any more

        log.info("DS-JEDAI: Matches: " + matches.count)

        val endTime = Calendar.getInstance()
        log.info("DS-JEDAI: Total Execution Time: " + (endTime.getTimeInMillis - startTime) / 1000.0)
    }
}