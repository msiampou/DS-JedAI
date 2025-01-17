package utils.configuration

import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
 * @author George Mandilaras (NKUA)
 */
object Constants {

	val defaultDatePattern = "yyyy-MM-dd HH:mm:ss"
	val geometryPredicate = "<http://strdf.di.uoa.gr/ontology#hasGeometry>"

	/**
	 * Relations
	 */
	object Relation extends Enumeration {
		type Relation = Value
		val EQUALS: Relation.Value = Value("equals")
		val DISJOINT: Relation.Value = Value("disjoint")
		val INTERSECTS: Relation.Value = Value("intersects")
		val TOUCHES: Relation.Value = Value("touches")
		val CROSSES: Relation.Value = Value("crosses")
		val WITHIN: Relation.Value = Value("within")
		val CONTAINS: Relation.Value = Value("contains")
		val OVERLAPS: Relation.Value = Value("overlaps")
		val COVERS: Relation.Value = Value("covers")
		val COVEREDBY: Relation.Value = Value("coveredby")
		val DE9IM: Relation.Value = Value("DE9IM")

		def exists(s: String): Boolean = values.exists(_.toString == s)

		def swap(r: Relation): Relation = r match {
			case Relation.WITHIN => Relation.CONTAINS
			case Relation.CONTAINS => Relation.WITHIN
			case Relation.COVERS => Relation.COVEREDBY
			case Relation.COVEREDBY => Relation.COVERS;
			case _ => r
		}
	}

	object ThetaOption extends Enumeration{
		type ThetaOption = Value
		val MAX: ThetaOption.Value = Value("max")
		val MIN: ThetaOption.Value = Value("min")
		val AVG: ThetaOption.Value = Value("avg")
		val AVG_x2: ThetaOption.Value = Value("avg2")
		val NO_USE: ThetaOption.Value = Value("none")

		def exists(s: String): Boolean = values.exists(_.toString == s)
	}

	/**
	 * Supported fileTypes
	 */
	object FileTypes extends Enumeration{
		type FileTypes = Value
		val NTRIPLES: FileTypes.Value = Value("nt")
		val TURTLE: FileTypes.Value = Value("ttl")
		val RDFXML: FileTypes.Value = Value("xml")
		val RDFJSON: FileTypes.Value = Value("rj")
		val CSV: FileTypes.Value = Value("csv")
		val TSV: FileTypes.Value = Value("tsv")
		val SHP: FileTypes.Value = Value("shp")
		val GEOJSON: FileTypes.Value = Value("geojson")

		def exists(s: String): Boolean = values.exists(_.toString == s)
	}

	/**
	 * Weighting Strategies
	 */
	object WeightingFunction extends Enumeration {
		type WeightingFunction = Value

		// co-occurrence frequency
		val CF: WeightingFunction.Value = Value("CF")

		// jaccard  similarity
		val JS: WeightingFunction.Value = Value("JS")

		// Pearson's chi squared test
		val PEARSON_X2: WeightingFunction.Value = Value("PEARSON_X2")

		// minimum bounding rectangle overlap
		val MBRO: WeightingFunction.Value = Value("MBRO")

		// inverse sum of points
		val ISP: WeightingFunction.Value = Value("ISP")

		// linear regression logits
		val LR: WeightingFunction.Value = Value("LR")

		def exists(s: String): Boolean = values.exists(_.toString == s)
	}

	/**
	 * YAML/command line Configurations arguments
	 */
	object InputConfigurations extends Enumeration {
		type YamlConfiguration = String

		val CONF_CONFIGURATIONS = "configurations"
		val CONF_SOURCE = "source"
		val CONF_TARGET = "target"
		val CONF_RELATION = "relation"
		val CONF_GEOMETRY_PREDICATE = "geometryPredicate"
		val CONF_PARTITIONS = "partitions"
		val CONF_THETA_GRANULARITY = "thetaGranularity"
		val CONF_GRID_TYPE = "gridType"
		val CONF_EXPORT_PATH = "exportPath"
		val CONF_STATISTICS = "stats"
		val CONF_UNRECOGNIZED = "unrecognized"

		// ENTITY TYPES / APPROXIMATIONS
		val CONF_ENTITY_TYPE = "entityType"
		val CONF_DECOMPOSITION_THRESHOLD = "decompositionThreshold"
		val CONF_GEOMETRY_APPROXIMATION_TYPE = "geometryApproximationType"

		// PROGRESSIVE METHODS
		val CONF_PROGRESSIVE_ALG = "progressiveAlgorithm"
		val CONF_BUDGET = "budget"
		val CONF_WS = "ws"
		val CONF_MAIN_WF = "mainWF"
		val CONF_SECONDARY_WF = "secondaryWF"
		val CONF_TOTAL_VERIFICATIONS = "totalVerifications"
		val CONF_QUALIFYING_PAIRS = "qualifyingPairs"

		// EARLY STOPPING
		val CONF_BATCH_SIZE = "batchSize"
		val CONF_PRECISION_LIMIT = "precisionLimit"
		val CONF_VIOLATIONS = "violations"

		// SUPERVISED PROGRESSIVE GIANT
		val CONF_ITERATIONS = "num_iterations"
		val CONF_EVALUATION = "eval"
	}

	object GridType extends Enumeration{
		type GridType = Value
		val KDBTREE: GridType.Value = Value("KDBTREE")
		val QUADTREE: GridType.Value = Value("QUADTREE")

		def exists(s: String): Boolean = values.exists(_.toString == s)
	}

	object EntityTypeENUM extends Enumeration {
		type EntityTypeENUM = Value
		val SPATIAL_ENTITY: EntityTypeENUM.Value = Value("SPATIAL_ENTITY")
		val SPATIOTEMPORAL_ENTITY: EntityTypeENUM.Value = Value("SPATIOTEMPORAL_ENTITY")
		val PREPARED_ENTITY: EntityTypeENUM.Value = Value("PREPARED_ENTITY")
		val DECOMPOSED_ENTITY: EntityTypeENUM.Value = Value("DECOMPOSED_ENTITY")
		val INDEXED_DECOMPOSED_ENTITY: EntityTypeENUM.Value = Value("INDEXED_DECOMPOSED_ENTITY")
		def exists(s: String): Boolean = values.exists(_.toString == s)
	}

	object GeometryApproximationENUM extends Enumeration {
		type GeometryApproximationENUM = Value
		val FINEGRAINED_ENVELOPES: GeometryApproximationENUM.Value = Value("FINEGRAINED_ENVELOPES")
		val MBR: GeometryApproximationENUM.Value = Value("MBR")

		def exists(s: String): Boolean = values.exists(_.toString == s)
	}

	/**
	 * Progressive Algorithms
	 */
	object ProgressiveAlgorithm extends Enumeration {
		type ProgressiveAlgorithm = Value
		val PROGRESSIVE_GIANT: ProgressiveAlgorithm.Value = Value("PROGRESSIVE_GIANT")
		val DYNAMIC_PROGRESSIVE_GIANT: ProgressiveAlgorithm.Value = Value("DYNAMIC_PROGRESSIVE_GIANT")
		val GEOMETRY_CENTRIC: ProgressiveAlgorithm.Value = Value("GEOMETRY_CENTRIC")
		val TOPK: ProgressiveAlgorithm.Value = Value("TOPK")
		val RECIPROCAL_TOPK: ProgressiveAlgorithm.Value = Value("RECIPROCAL_TOPK")
		val RANDOM: ProgressiveAlgorithm.Value = Value("RANDOM")
		val EARLY_STOPPING: ProgressiveAlgorithm.Value = Value("EARLY_STOPPING")
		val SUPERVISED: ProgressiveAlgorithm.Value = Value("SUPERVISED")
		def exists(s: String): Boolean = values.exists(_.toString == s)
	}

	sealed trait WeightingScheme extends Serializable {val value: String }
	case object SIMPLE extends WeightingScheme {val value = "SIMPLE"}
	case object COMPOSITE extends WeightingScheme {val value = "COMPOSITE"}
	case object HYBRID extends WeightingScheme {val value = "HYBRID"}
	case object THIN_MULTI_COMPOSITE extends WeightingScheme {val value = "THIN_MULTI_COMPOSITE"}
	def WeightingSchemeFactory(ws: String): WeightingScheme ={
		ws.toLowerCase() match {
			case "simple" => SIMPLE
			case "composite" => COMPOSITE
			case "hybrid" => HYBRID
			case "ThinMultiCompositePair" => THIN_MULTI_COMPOSITE
			case _ => SIMPLE
		}
	}

	def checkWS(ws: String): Boolean ={
		ws.toLowerCase() match {
			case "single" | "composite" | "hybrid" => true
			case _ => false
		}
	}

	sealed trait FeaturePos extends Serializable {val value: Int}
	case object SourceArea extends FeaturePos {val value = 0}
	case object TargetArea extends FeaturePos {val value = 1}
	case object IntersectionArea extends FeaturePos {val value = 2}
	case object SourceTiles extends FeaturePos {val value = 3}
	case object TargetTiles extends FeaturePos {val value = 4}
	case object CommonTiles extends FeaturePos {val value = 5}
	case object SourceBoundPoints extends FeaturePos {val value = 6}
	case object TargetBoundPoints extends FeaturePos {val value = 7}
	case object SourceLen extends FeaturePos {val value = 8}
	case object TargetLen extends FeaturePos {val value = 9}
	case object SourceTotalCooccurrences extends FeaturePos {val value = 10}
	case object SourceDistCooccurrences extends FeaturePos {val value = 11}
	case object SourceRealCooccurrences extends FeaturePos {val value = 12}
	case object TargetTotalCooccurrences extends FeaturePos {val value = 13}
	case object TargetDistCooccurrences extends FeaturePos {val value = 14}
	case object TargetRealCooccurrences extends FeaturePos {val value = 15}
	case object Label extends FeaturePos {val value = 16}

	val schema = new StructType()
		.add(StructField("_1", DoubleType, false))
		.add(StructField("_2", DoubleType, false))
		.add(StructField("_3", DoubleType, false))
		.add(StructField("_4", DoubleType, false))
		.add(StructField("_5", DoubleType, false))
		.add(StructField("_6", DoubleType, false))
		.add(StructField("_7", DoubleType, false))
		.add(StructField("_8", DoubleType, false))
		.add(StructField("_9", DoubleType, false))
		.add(StructField("_10", DoubleType, false))
		.add(StructField("_11", DoubleType, false))
		.add(StructField("_12", DoubleType, false))
		.add(StructField("_13", DoubleType, false))
		.add(StructField("_14", DoubleType, false))
		.add(StructField("_15", DoubleType, false))
		.add(StructField("_16", DoubleType, false))
		.add(StructField("label", DoubleType, false))

	val featureCols = Array("_1" , "_2" , "_2" , "_3" , "_4" , "_4" , "_6" , "_7" , "_8" , "_9" , "_10" , "_11",
		"_12", "_13", "_14", "_15", "_16")
}
