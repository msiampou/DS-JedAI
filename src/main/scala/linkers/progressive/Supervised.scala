package linkers.progressive

import model._
import model.entities.EntityT
import model.structures.{CandidateSet, ComparisonPQ, StaticComparisonPQ}
import model.weightedPairs.SamplePairT
import org.apache.spark.ml.linalg.Vectors
import org.locationtech.jts.geom.Envelope
import utils.configuration.Constants
import utils.configuration.Constants.Relation.Relation
import utils.configuration.Constants.WeightingFunction.WeightingFunction

import scala.collection.mutable.ListBuffer
import scala.language.postfixOps

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
  val targetLen: Int = targetAr.size
  val maxCandidatePairs: Int = 50*sourceLen

  val NUM_FEATURES: Int = 16
  val SAMPLE_SIZE: Int = (0.6*(sourceLen+targetLen)).toInt
  val CLASS_SIZE: Int = SAMPLE_SIZE/2

  val featureSet: FeatureSet = FeatureSet(CLASS_SIZE, NUM_FEATURES, SAMPLE_SIZE, sourceLen, source, tileGranularities, targetLen)
  var trainSet: weka.core.Instances = _

  val candidateMatches = for(idx <- 0 until targetLen) yield getCandidates(targetAr(idx))

  /*
    Given a target entity `t`, it returns a set of its candidates ids.
    During iteration, a `frequencyMap` -that counts the number of common tiles
    between `t` and each candidate entity- is also computed.
   */
  def getCandidates(targetEntity: EntityT): Set[Int] = {
    var candidateMatches = Set[Int]()
    val frequencyMap: CandidateSet = CandidateSet()
    // retrieve candidates
    val candidates = getAllCandidatesWithIndex(targetEntity, sourceIndex, partitionBorder)
    // get the ids of each candidate and update the number
    // of common tiles between `t` and each candidate
    candidates.foreach { case (candidateID, _) =>
      frequencyMap.increment(candidateID)
      candidateMatches += candidateID
    }
    // append `frequencyMap` to a list
    featureSet.freqArray += frequencyMap
    // return set of ids
    candidateMatches
  }

  /*
    Returns a tuple of (a, b, c), where
      a: is the number of common tiles between the candidate geometries
      b: is the number of distinct candidates
      c: is the number of total candidates with common tiles
   */
  def candidateStatistics(pair: SamplePairT): (Int, Int, Int) = {
    val targetEntity = pair.getTargetGeometry
    val targetID = pair.getTargetId
    featureSet.getcandidateStatistics(targetEntity, candidateMatches(targetID), targetID)
  }

  /*
    Iterates through source and target geometries to pre-compute
    relevant features that will be utilized during the training process.
   */
  override def preprocessing: Supervised = {
    // no entities lie in this partition
    if (sourceLen < 1) return this
    // compute features (F1)-(F4)-(F7)-(F9)
    source.foreach(sourceEntity => featureSet.updateSourceStats(sourceEntity, tileGranularities))

    // collect a number of random candidates
    assert(maxCandidatePairs > SAMPLE_SIZE)
    val random = scala.util.Random
    val pairIds = random.shuffle(0 until maxCandidatePairs toSet).take(SAMPLE_SIZE)
    var pairID = 0
    val totalCandidates = candidateMatches.size

    // iterate through target entities to
    // compute relevant features
    targetAr.indices.foreach { targetID =>
      val targetEntity = targetAr(targetID)
      // compute features (F2)-(F5)-(F8)-(F10)
      featureSet.updateTargetStats(targetEntity, tileGranularities)
      var co_occurrences = 0
      var realCandidates = 0
      val targetMatches = candidateMatches(targetID)
      // for each candidate in set
      targetMatches.foreach { candidateID =>
        // update variables for (F11) to (F16)
        co_occurrences += featureSet.updateCandStats(candidateID, targetID)
        val candidateEntity = source(candidateID)
        // if the two entities have common tiles, then they probably have a topological relation
        val intersects = featureSet.updateIntersectionStats(candidateEntity, targetEntity, candidateID, targetID)
        if (intersects) {
          realCandidates += 1
          // randomly choose a `targetEntity` - `candidateEntity`
          // pair to sample for the training process
          if (pairIds.contains(pairID)) {
            featureSet.addMatch(SamplePairT(candidateID, targetID, candidateEntity, targetEntity))
          }
          pairID += 1
        }
      }
      // update features (F14)-(F15)-(F16)
      featureSet.updateTargetStats(co_occurrences, targetMatches.size, realCandidates)
    }
    // update features (F11)-(F12)-(F13)
    featureSet.updateSourceStats()
    this
  }

  /*
    Trains a local Logistic Regression classifier
   */
  def train(posOccurencies: Seq[(Int, Int, Int)], negOccurencies: Seq[(Int, Int, Int)],
            positivePairs: ListBuffer[SamplePairT], negativePairs: ListBuffer[SamplePairT],
            sz: Int): weka.core.Instances =
    featureSet.train(posOccurencies, negOccurencies, positivePairs, negativePairs, sz)

  /*
    Creates a sampling set and trains a local Logistic Regression classifier
   */
  override def buildClassifier: Supervised = {
    // no entities lie in this partition
    if (sourceLen < 1) return this

    // create a balanced sample of related and non-related entity pairs
    val (positivePairs, negativePairs) = featureSet.trainSampling
    val sz = if (positivePairs.size < negativePairs.size) positivePairs.size else negativePairs.size
    // no positive or negative pairs exist
    if (sz < 1) return this

    // obtain a sequence of candidate-relevant statistics to use
    // during the training process
    val posOccurencies: Seq[(Int, Int, Int)] = for (idx <- 0 until sz)
      yield candidateStatistics(positivePairs(idx))
    val negOccurencies: Seq[(Int, Int, Int)] = for (idx <- 0 until  sz)
      yield candidateStatistics(negativePairs(idx))

    // train classifier
    this.trainSet = train(posOccurencies, negOccurencies, positivePairs, negativePairs, sz)
    this
  }

  /*
    Prioritizes pairs-to-be examined based on their classification probability.
   */
  override def prioritize(relation: Relation): ComparisonPQ = {
    var totalDecisions = 0
    val localBudget = math.ceil((budget * source.length.toDouble) / totalSourceEntities.toDouble).toLong
    val pq: StaticComparisonPQ = StaticComparisonPQ(localBudget)

    if (sourceLen < 1 || this.trainSet == null) return pq

    val totalCandidates = candidateMatches.size

    var counter = 0
    targetAr.indices.foreach { idx =>
      val t = targetAr(idx)
      val targetMatches = candidateMatches(idx)
      var realPairs = 0
      val distinctPairs = targetMatches.size
      var totalPairs = 0
      targetMatches.foreach { candidateID =>
        val (entityFrequency, isValid) = this.featureSet.isValid(candidateID, t, idx)
        totalPairs += entityFrequency
        realPairs += isValid
      }
      targetMatches.foreach { candidateID =>
        val lrProbs = featureSet.getProbability(this.trainSet, candidateID, t, idx, totalPairs, distinctPairs, realPairs)
        if (!lrProbs.isEmpty) {
          totalDecisions += 1
          val candidateEntity = source(candidateID)
          if (lrProbs(0) < lrProbs(1)) {
            val w = lrProbs(1).toFloat
            val wp = weightedPairFactory.createWeightedPair(counter, candidateEntity, candidateID, t, idx, w)
            pq.enqueue(wp)
            counter += 1
          }
        }
      }
    }
    pq
  }
}