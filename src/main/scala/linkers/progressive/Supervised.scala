package linkers.progressive

import model._
import model.entities.EntityT
import model.structures.{ComparisonPQ}
import model.weightedPairs.SamplePairT
import org.apache.spark.ml.linalg.Vectors
import org.locationtech.jts.geom.Envelope
import utils.configuration.Constants
import utils.configuration.Constants.Relation.Relation
import utils.configuration.Constants.WeightingFunction.WeightingFunction
import scala.collection.mutable.ListBuffer

case class Supervised (source: Array[EntityT],
                       target: Iterable[EntityT],
                       tileGranularities: TileGranularities,
                       partitionBorder: Envelope,
                       mainWF: WeightingFunction,
                       secondaryWF: Option[WeightingFunction],
                       budget: Int,
                       totalSourceEntities: Long,
                       ws: Constants.WeightingScheme,
                       totalBlocks: Double
                      )

  extends ProgressiveLinkerT {

  val sourceLen = source.length
  val maxCandidatePairs = 10 * sourceLen

  val CLASS_SIZE = 100
  val NUM_FEATURES = 16
  val SAMPLE_SIZE = if (maxCandidatePairs < 5000) maxCandidatePairs else 5000

  val featureSet : FeatureSet = FeatureSet( CLASS_SIZE, NUM_FEATURES, SAMPLE_SIZE, sourceLen, source, tileGranularities)

  def getCandidates(t: EntityT, cleanup: Boolean = false): Set[Int] = {
    if (cleanup) featureSet.frequencyMap.clear()
    var candidateMatches = Set[Int]()
    val candidates = getAllCandidatesWithIndex(t, sourceIndex, partitionBorder)
    candidates.foreach { case (i, _) =>
      featureSet.frequencyMap.increment(i, 1)
      candidateMatches += i
    }
    candidateMatches
  }

  def preprocessing : Unit = {

    if (sourceLen < 1) return

    source.foreach(s => featureSet.updateSourceStats(s, tileGranularities))

    val random = scala.util.Random
    val pairIds = scala.collection.mutable.HashSet[Integer]()
    val maxCandidatePairs = 10 * sourceLen
    while (pairIds.size < SAMPLE_SIZE) {
      pairIds.add(random.nextInt(maxCandidatePairs))
    }

    var pairID = 0
    var counter = 0
    targetAr.indices.foreach { idx =>
      val t = targetAr(idx)
      featureSet.updateTargetStats(t, tileGranularities)
      var co_occurrences = 0
      var candidates = 0
      val candidateMatches = getCandidates(t)
      candidateMatches.foreach { cID =>
        co_occurrences += featureSet.updateCandStats(cID)
        val c = source(cID)
        candidates += featureSet.updateIntersectionStats(c, t, cID)
        if(pairIds.contains(pairID)) {
          featureSet.addMatch(SamplePairT(cID, counter, c, t))
        }
        pairID+=1
      }
      featureSet.updateTargetStats(co_occurrences, candidateMatches.size, candidates)
      counter+=1
      featureSet.frequencyMap.clear
    }
    featureSet.updateSourceStats()
  }

  def train(posCands: Seq[Set[Int]], negCands: Seq[Set[Int]], pPairs: ListBuffer[SamplePairT],
            nPairs: ListBuffer[SamplePairT], sz: Int): Unit =
    featureSet.train(posCands, negCands, pPairs, nPairs, sz)

  override def buildClassifier: FeatureSet = {

    val dfRow = Array.fill[Double](NUM_FEATURES+1)(0.0)

    if (sourceLen < 1) return this.featureSet

    this.preprocessing
    val (excessVerifications, positivePairs, negativePairs) = featureSet.trainSampling

    val sz = if (positivePairs.size < negativePairs.size) positivePairs.size else negativePairs.size
    if (sz < 1) return this.featureSet

    val posCands: Seq[Set[Int]] = for (idx <- 0 until sz)
      yield getCandidates(positivePairs(idx).getTargetGeometry, true)
    val negCands: Seq[Set[Int]] = for (idx <- 0 until  sz)
      yield getCandidates(negativePairs(idx).getTargetGeometry, true)

    train(posCands, negCands, positivePairs, negativePairs, sz)
    this.featureSet
  }

  override def prioritize(relation: Relation): ComparisonPQ = ???
}