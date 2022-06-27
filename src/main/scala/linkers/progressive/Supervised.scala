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
  val targetLen: Int = target.size
  val maxCandidatePairs: Int = 10 * sourceLen

  val NUM_FEATURES: Int = 16
  val SAMPLE_SIZE: Int = (0.8*(sourceLen + targetLen)).toInt
  val CLASS_SIZE: Int = SAMPLE_SIZE/2

  val featureSet: FeatureSet = FeatureSet(CLASS_SIZE, NUM_FEATURES, SAMPLE_SIZE, sourceLen, source, tileGranularities)
  var trainSet: weka.core.Instances = _

  def getCandidates(t: EntityT): Set[Int] = {
    var candidateMatches = Set[Int]()
    val frequencyMap: CandidateSet = CandidateSet()
    val candidates = getAllCandidatesWithIndex(t, sourceIndex, partitionBorder)
    candidates.foreach { case (i, _) =>
      frequencyMap.increment(i)
      candidateMatches += i
    }
    featureSet.freqArray += frequencyMap
    candidateMatches
  }

  def getCandidates(p: SamplePairT): (Int, Int, Int) = {
    val t = p.getTargetGeometry
    val tID = p.getTargetId
    var candidateMatches = Set[Int]()
    val frequencyMap: CandidateSet = CandidateSet()
    val candidates = getAllCandidatesWithIndex(t, sourceIndex, partitionBorder)
    candidates.foreach { case (i, _) =>
      frequencyMap.increment(i)
      candidateMatches += i
    }
    featureSet.freqArray(tID) = frequencyMap
    featureSet.getCandStats(t, candidateMatches, tID)
  }

  /*
   ...
   */
  override def preprocessing: Supervised = {
    // no geometries lie in this partition
    if (sourceLen < 1) return this
    // compute features (F1)-(F4)-(F7)-(F9)
    source.foreach(s => featureSet.updateSourceStats(s, tileGranularities))
    assert(maxCandidatePairs > SAMPLE_SIZE)
    val random = scala.util.Random
    val pairIds = scala.collection.mutable.HashSet[Integer]()
    // collect a number of random candidates
    while (pairIds.size < SAMPLE_SIZE) {
      pairIds.add(random.nextInt(maxCandidatePairs))
    }

    var pairID = 0
    targetAr.indices.foreach { idx =>
      val t = targetAr(idx)
      // compute features (F2)-(F5)-(F8)-(F10)
      featureSet.updateTargetStats(t, tileGranularities)
      var co_occurrences = 0
      var candidates = 0
      val candidateMatches = getCandidates(t)
      candidateMatches.foreach { cID =>
        co_occurrences += featureSet.updateCandStats(cID, idx)
        val c = source(cID)
        val intersects = featureSet.updateIntersectionStats(c, t, cID, idx)
        if (intersects) {
          candidates += 1
          if (pairIds.contains(pairID)) {
            featureSet.addMatch(SamplePairT(cID, idx, c, t))
          }
          pairID += 1
        }
      }
      featureSet.updateTargetStats(co_occurrences, candidateMatches.size, candidates)
    }
    featureSet.updateSourceStats()
    this
  }

  def train(posCands: Seq[(Int, Int, Int)], negCands: Seq[(Int, Int, Int)], pPairs: ListBuffer[SamplePairT],
            nPairs: ListBuffer[SamplePairT], sz: Int): weka.core.Instances =
    featureSet.train(posCands, negCands, pPairs, nPairs, sz)

  override def buildClassifier: Supervised = {
    if (sourceLen < 1) return this

    val (positivePairs, negativePairs) = featureSet.trainSampling
    val sz = if (positivePairs.size < negativePairs.size) positivePairs.size else negativePairs.size
    if (sz < 1) return this

    val posCands: Seq[(Int, Int, Int)] = for (idx <- 0 until sz)
      yield getCandidates(positivePairs(idx))

    val negCands: Seq[(Int, Int, Int)] = for (idx <- 0 until  sz)
      yield getCandidates(negativePairs(idx))

    this.trainSet = train(posCands, negCands, positivePairs, negativePairs, sz)
    this
  }

  override def prioritize(relation: Relation): ComparisonPQ = {
    var posDecisions = 0
    var totalDecisions = 0
    val localBudget = math.ceil((budget * source.length.toDouble) / totalSourceEntities.toDouble).toLong
    val pq: StaticComparisonPQ = StaticComparisonPQ(localBudget)

    featureSet.freqArray = scala.collection.mutable.ListBuffer[CandidateSet]()

    if (sourceLen < 1 || this.trainSet == null) return pq

    var counter = 0
    targetAr.indices.foreach { idx =>
      val t = targetAr(idx)
      val candidateMatches = getCandidates(t)
      var rPairs = 0
      val dPairs = candidateMatches.size
      var tPairs = 0
      candidateMatches.foreach { cID =>
        val candidateStats = this.featureSet.isValid(cID, t, idx)
        tPairs += candidateStats._1
        rPairs += candidateStats._2
      }
      candidateMatches.foreach { cID =>
        val lrProbs = featureSet.getProbability(this.trainSet, cID, t, idx, tPairs, dPairs, rPairs)
        if (!lrProbs.isEmpty) {
          totalDecisions += 1
          val c = source(cID)
          if (lrProbs(0) < lrProbs(1)) {
            //  posDecisions += 1
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