package model

import model.entities.EntityT
import model.structures.CandidateSet
import model.weightedPairs.{SamplePairT, VerifiedPair}
import org.apache.spark.ml.linalg.Vectors
import utils.configuration.Constants.{CommonTiles, IntersectionArea, Label, SourceArea, SourceBoundPoints, SourceDistCooccurrences, SourceLen, SourceRealCooccurrences, SourceTiles, SourceTotalCooccurrences, TargetArea, TargetBoundPoints, TargetDistCooccurrences, TargetLen, TargetRealCooccurrences, TargetTiles, TargetTotalCooccurrences}

import scala.collection.mutable.ListBuffer

/*
 * Holds helper methods for Supervised Scheduling.
 */

sealed trait FeatureStatistics {

  val NO_OF_FEATURES: Int = 16
  val POSITIVE_PAIR: Int = 1
  val NEGATIVE_PAIR: Int = 0

  val RELATED: String = "related"
  val NON_RELATED: String = "nonrelated"

  val maxFeatures = Array.fill[Double](NO_OF_FEATURES)(Double.MinValue)
  val minFeatures = Array.fill[Double](NO_OF_FEATURES)(Double.MaxValue)

  val sample: scala.collection.mutable.ListBuffer[SamplePairT] = scala.collection.mutable.ListBuffer[SamplePairT]()
  val vPairs: scala.collection.mutable.HashSet[VerifiedPair] = scala.collection.mutable.HashSet[VerifiedPair]()

  val freqArray: scala.collection.mutable.ListBuffer[CandidateSet] = scala.collection.mutable.ListBuffer[CandidateSet]()

  var distinctCooccurrences: CandidateSet = CandidateSet()
  var totalCooccurrences: CandidateSet = CandidateSet()
  var realCandidates: CandidateSet = CandidateSet()

  val source: Array[EntityT]
  val datasetDelimiter: Int
  val THETA: TileGranularities
  val CLASS_SIZE: Int

  def update(featurePos: Int, value: Double): Unit = {
    if (maxFeatures(featurePos) < value) maxFeatures(featurePos) = value
    if (value < minFeatures(featurePos)) minFeatures(featurePos) = value
  }

  def updateSourceStats(s: EntityT, tileGranularities: TileGranularities): Unit = {
    update(SourceArea.value, s.getEnvelopeInternal().getArea)
    update(SourceTiles.value, s.getNumOfOverlappingTiles(tileGranularities))
    update(SourceBoundPoints.value, s.getNumPoints)
    update(SourceLen.value, s.getLength)
  }

  def addMatch(pair: SamplePairT): Unit = sample += pair

  def addMatch(set: CandidateSet): Unit = freqArray += set

  def updateSourceStats : Unit = {
    for (i <- 0 to datasetDelimiter) {
      update(SourceTotalCooccurrences.value, totalCooccurrences.get(i))
      update(SourceDistCooccurrences.value, distinctCooccurrences.get(i))
      update(SourceRealCooccurrences.value, realCandidates.get(i))
    }
  }

  def updateTargetStats(t: EntityT, tileGranularities: TileGranularities): Unit = {
    update(TargetArea.value, t.getEnvelopeInternal().getArea)
    update(TargetTiles.value, t.getNumOfOverlappingTiles(tileGranularities))
    update(TargetBoundPoints.value, t.getNumPoints)
    update(TargetLen.value, t.getLength)
  }

  def updateTargetStats(n_co_occurrences: Int, dist_co_occurrences: Int, n_candidates: Int): Unit = {
    update(TargetTotalCooccurrences.value, n_co_occurrences)
    update(TargetDistCooccurrences.value, dist_co_occurrences)
    update(TargetRealCooccurrences.value, n_candidates)
  }

  def updateCandStats(cID: Int, idx: Int): Int = {
    val freq = freqArray(idx).get(cID)
    distinctCooccurrences.increment(cID)
    totalCooccurrences.increment(cID, freq)
    freq
  }

  def updateIntersectionStats(s: EntityT, t: EntityT, cID: Int, idx: Int): Boolean = {
    val intersect = s.getEnvelopeInternal.intersects(t.getEnvelopeInternal)
    var retVal = false
    if (intersect) {
      realCandidates.increment(cID)
      val interMBR = s.getEnvelopeInternal.intersection(t.getEnvelopeInternal).getArea
      update(IntersectionArea.value, interMBR)
      update(CommonTiles.value, freqArray(idx).get(cID))
      retVal = true
    }
    retVal
  }

  def getcandidateStatistics(t: EntityT, candSet: Seq[Int], idx: Int): (Int, Int, Int) = {
    var co_occurrences = 0
    var d_co_occurrences = 0
    var t_co_occurrences = 0
    candSet.foreach { cID =>
      val c = source(cID)
      co_occurrences += freqArray(idx).get(cID)
      d_co_occurrences += 1
      val intersects = c.getEnvelopeInternal.intersects(t.getEnvelopeInternal)
      if (intersects) t_co_occurrences += 1
    }
    (co_occurrences, d_co_occurrences, t_co_occurrences)
  }
}

case class FeatureVector (CLASS_SIZE: Int,
                          datasetDelimiter: Int,
                          source: Array[EntityT],
                          THETA: TileGranularities) extends FeatureStatistics {

  val attributes: java.util.ArrayList[weka.core.Attribute] = this.getAttributes
  val clf: weka.classifiers.Classifier = new weka.classifiers.functions.Logistic()


  def getAttributes: java.util.ArrayList[weka.core.Attribute] = {
    val attributes = new java.util.ArrayList[weka.core.Attribute]()
    for (idx <- 0 until  NO_OF_FEATURES) {
      val attr = new weka.core.Attribute("_" + idx)
      attributes.add(attr)
    }
    val classLabels = new java.util.ArrayList[String]()
    classLabels.add(NON_RELATED)
    classLabels.add(RELATED)
    val classAttribute = new weka.core.Attribute("class", classLabels)
    attributes.add(classAttribute)
    attributes
  }

  def getFeatures(s: EntityT, t: EntityT, sID: Int, tID: Int, total: Int, distinct: Int, real: Int,
                  arraySize: Int): Array[Double] = {

    val dfRow = Array.fill[Double](arraySize)(0.0)

    val interMBR = s.getEnvelopeInternal.intersection(t.getEnvelopeInternal)
    //area-based features
    dfRow(SourceArea.value) = (s.getEnvelopeInternal.getArea - minFeatures(0)) / maxFeatures(0) * 10000
    dfRow(TargetArea.value) = (t.getEnvelopeInternal.getArea - minFeatures(1)) / maxFeatures(1) * 10000
    dfRow(IntersectionArea.value) = (interMBR.getArea - minFeatures(2)) / maxFeatures(2) * 10000

    //grid-based features
    dfRow(SourceTiles.value) = (s.getNumOfOverlappingTiles(THETA) - minFeatures(3)) / maxFeatures(3) * 10000
    dfRow(TargetTiles.value) = (t.getNumOfOverlappingTiles(THETA) - minFeatures(4)) / maxFeatures(4) * 10000
    dfRow(CommonTiles.value) = (freqArray(tID).get(sID) - minFeatures(5)) / maxFeatures(5) * 10000

    //boundary-based features
    dfRow(SourceBoundPoints.value) = (s.getNumPoints - minFeatures(6)) / maxFeatures(6) * 10000
    dfRow(TargetBoundPoints.value) = (t.getNumPoints - minFeatures(7)) / maxFeatures(7) * 10000
    dfRow(SourceLen.value) = (s.getLength - minFeatures(8)) / maxFeatures(8) * 10000
    dfRow(TargetLen.value) = (t.getLength - minFeatures(9)) / maxFeatures(9) * 10000

    //candidate-based features
    dfRow(SourceTotalCooccurrences.value) = (totalCooccurrences.get(sID) - minFeatures(10)) / maxFeatures(10) * 10000
    dfRow(SourceDistCooccurrences.value) = (distinctCooccurrences.get(sID) - minFeatures(11)) / maxFeatures(11) * 10000
    dfRow(SourceRealCooccurrences.value) = (realCandidates.get(sID) - minFeatures(12)) / maxFeatures(12) * 10000
    //target geometry
    dfRow(TargetTotalCooccurrences.value) = (total - minFeatures(13)) / maxFeatures(13) * 10000
    dfRow(TargetDistCooccurrences.value) = (distinct - minFeatures(14)) / maxFeatures(14) * 10000
    dfRow(SourceRealCooccurrences.value) = (real - minFeatures(15)) / maxFeatures(15) * 10000

    dfRow
  }

  def computeRowVector(s: EntityT, t: EntityT, sID: Int, tID: Int, tPairs: Int, dPairs: Int, rPairs: Int): Array[Double] = {
    getFeatures(s, t, sID, tID, tPairs, dPairs, rPairs, NO_OF_FEATURES+1)
  }

  def computeRowVector(label: Int, p: SamplePairT, candSet: (Int, Int, Int)) : Array[Double] = {
    val sID = p.getSourceId
    val tID = p.getTargetId
    val s = p.getSourceGeometry
    val t = p.getTargetGeometry
    val (co_occurrences, d_co_occurrences, t_co_occurrences) = candSet
    val dfRow = getFeatures(s, t, sID, tID, co_occurrences, d_co_occurrences, t_co_occurrences, NO_OF_FEATURES+1)
    dfRow(Label.value) = label.toDouble
    dfRow
  }

  def build(trainClass: Int, trainPairs: scala.collection.mutable.ListBuffer[SamplePairT],
            candidates: Seq[(Int, Int, Int)], sz: Int): Seq[Array[Double]] =  {
    for (idx <- 0 until  sz) yield computeRowVector(trainClass, trainPairs(idx), candidates(idx))
  }

  def trainSampling : (ListBuffer[SamplePairT], ListBuffer[SamplePairT]) = {
    val negativePairs = scala.collection.mutable.ListBuffer[SamplePairT]()
    val positivePairs = scala.collection.mutable.ListBuffer[SamplePairT]()

    var idx = 0
    while (sample.size > 0 && idx < sample.size) {
      val pair = sample(idx)
      val s = pair.getSourceGeometry
      val t = pair.getTargetGeometry
      val rel = s.geometry.relate(t.geometry)
      val im = IM(s, t, rel)

      val vPair = VerifiedPair(pair.getSourceId, pair.getTargetId)
      vPairs += vPair
      if (im.relate) {
        if (positivePairs.size < CLASS_SIZE) positivePairs += pair
      } else {
        if (negativePairs.size < CLASS_SIZE) negativePairs += pair
      }
      idx += 1
    }
    (positivePairs, negativePairs)
  }

  def setInstance(featureVector: Array[Double], trainSet: weka.core.Instances, weight: Double): weka.core.Instance = {
    val instance = new weka.core.DenseInstance(weight, featureVector)
    instance.setDataset(trainSet)
    instance
  }

  def train(posCands: Seq[(Int, Int, Int)], negCands: Seq[(Int, Int, Int)], pPairs: ListBuffer[SamplePairT],
            nPairs: ListBuffer[SamplePairT], sz: Int): weka.core.Instances = {
    val posInstances = build(POSITIVE_PAIR, pPairs, posCands, sz)
    val negInstances = build(NEGATIVE_PAIR, nPairs, negCands, sz)

    val trainSet = new weka.core.Instances("trainingSet", attributes, 2*sz)
    trainSet.setClassIndex(NO_OF_FEATURES)
    for (i <- 0 until sz) {
      val posI = setInstance(posInstances(i), trainSet, 1.0)
      trainSet.add(posI)

      val negI = setInstance(negInstances(i), trainSet, 1.0)
      trainSet.add(negI)
    }
    this.clf.buildClassifier(trainSet)
    trainSet
  }

  def isValid(sID: Int, t: EntityT, tID: Int): (Int, Int) = {
    val s = source(sID)
    val freq = freqArray(tID).get(sID)
    val intersects = s.getEnvelopeInternal.intersects(t.getEnvelopeInternal)
    val isValid = if (intersects) 1 else 0
    (freq, isValid)
  }

  def getProbability(trainSet: weka.core.Instances, candidateID: Int, t: EntityT, tID: Int, tPairs: Int,
                     dPairs: Int, rPairs: Int): Array[Double] = {
    val c = source(candidateID)
    val intersects = c.getEnvelopeInternal.intersects(t.getEnvelopeInternal)
    val emptyArray: Array[Double] = Array[Double]()

    if (!intersects) return emptyArray

    val vPair = VerifiedPair(candidateID, tID)
    if (this.vPairs.contains(vPair)) return emptyArray

    val features = computeRowVector(c, t, candidateID, tID, tPairs, dPairs, rPairs)
    val instance = setInstance(features, trainSet, 1.0)
    val prob: Array[Double] = this.clf.distributionForInstance(instance)
    prob
  }
}