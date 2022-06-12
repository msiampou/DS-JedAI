package linkers.progressive

import model._
import model.entities.EntityT
import model.structures.{ComparisonPQ, StaticComparisonPQ}
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

  val sourceLen: Int = source.length
  val maxCandidatePairs: Int = 10 * sourceLen

  val CLASS_SIZE: Int = 100
  val NUM_FEATURES: Int = 16
  val SAMPLE_SIZE: Int = if (maxCandidatePairs < 5000) maxCandidatePairs else 5000

  val featureSet: FeatureSet = FeatureSet( CLASS_SIZE, NUM_FEATURES, SAMPLE_SIZE, sourceLen, source, tileGranularities)
  var trainSet: weka.core.Instances = _

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

  override def preprocessing: Supervised = {

    if (sourceLen < 1) return this

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
        if (pairIds.contains(pairID)) {
          featureSet.addMatch(SamplePairT(cID, counter, c, t))
        }
        pairID += 1
      }
      featureSet.updateTargetStats(co_occurrences, candidateMatches.size, candidates)
      counter += 1
      featureSet.frequencyMap.clear
    }
    featureSet.updateSourceStats()
    this
  }

  def train(posCands: Seq[Set[Int]], negCands: Seq[Set[Int]], pPairs: ListBuffer[SamplePairT],
            nPairs: ListBuffer[SamplePairT], sz: Int): weka.core.Instances =
    featureSet.train(posCands, negCands, pPairs, nPairs, sz)

  override def buildClassifier: (Int, Supervised) = {
    if (sourceLen < 1) return (0, this)

    val (excessVerifications, positivePairs, negativePairs) = featureSet.trainSampling

    val sz = if (positivePairs.size < negativePairs.size) positivePairs.size else negativePairs.size
    if (sz < 1) return (0, this)

    val posCands: Seq[Set[Int]] = for (idx <- 0 until sz)
      yield getCandidates(positivePairs(idx).getTargetGeometry, true)
    val negCands: Seq[Set[Int]] = for (idx <- 0 until  sz)
      yield getCandidates(negativePairs(idx).getTargetGeometry, true)

    this.trainSet = train(posCands, negCands, positivePairs, negativePairs, sz)
    (excessVerifications, this)
  }

  override def prioritize(relation: Relation): ComparisonPQ = {
    var posDecisions = 0
    var totalDecisions = 0
    val localBudget = math.ceil((budget*source.length.toDouble) / totalSourceEntities.toDouble).toLong
    val pq: StaticComparisonPQ = StaticComparisonPQ(localBudget)

    this.featureSet.frequencyMap.clear()

    if (sourceLen < 1 || this.trainSet == null) return pq

    var counter = 0
    targetAr.indices.foreach { idx =>
      val t = targetAr(idx)
      val candidateMatches = getCandidates(t)
      var rPairs = 0
      val dPairs = candidateMatches.size
      var tPairs = 0
      candidateMatches.foreach { cID =>
        val candidateStats = this.featureSet.isValid(cID, t)
        tPairs += candidateStats._1
        rPairs += candidateStats._2
      }
      candidateMatches.foreach { cID =>
        val lrProbs = featureSet.getProbability(this.trainSet, cID, t, idx, tPairs, dPairs, rPairs)
        if (!lrProbs.isEmpty) {
          totalDecisions += 1
          if (lrProbs(0) < lrProbs(1)) {
            posDecisions += 1
            val c = source(cID)
            val w = lrProbs(1).toFloat
            val wp = weightedPairFactory.createWeightedPair(counter, c, cID, t, idx, w)
            pq.enqueue(wp)
            counter += 1
          }
        }
      }
    }
    pq
  }
}