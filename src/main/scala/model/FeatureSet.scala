package model

import model.entities.EntityT
import model.structures.CandidateSet
import model.weightedPairs.SamplePairT
import org.apache.spark.ml.linalg.Vectors
import utils.configuration.Constants.{CommonTiles, IntersectionArea, Label, SourceArea, SourceBoundPoints, SourceDistCooccurrences, SourceLen, SourceRealCooccurrences, SourceTiles, SourceTotalCooccurrences, TargetArea, TargetBoundPoints, TargetDistCooccurrences, TargetLen, TargetRealCooccurrences, TargetTiles, TargetTotalCooccurrences}
import org.apache.spark.sql.Row
import utils.configuration.Constants

import scala.collection.mutable.ListBuffer

case class FeatureSet (class_size: Int,
                       n_features: Int,
                       sample_size: Int,
                       datasetDel: Int,
                       sourceAr: Array[EntityT],
                       tileGranularities: TileGranularities
                      ) {

  val CLASS_SIZE: Int = class_size
  val NO_OF_FEATURES: Int = n_features
  val SAMPLE_SIZE: Int = sample_size

  val POSITIVE_PAIR: Int = 1
  val NEGATIVE_PAIR: Int = 0

  val RELATED: String = "related"
  val NON_RELATED: String = "nonrelated"

  val source: Array[EntityT] = sourceAr
  val datasetDelimiter: Int = datasetDel
  val THETA: TileGranularities = tileGranularities

  val maxFeatures = Array.fill[Double](NO_OF_FEATURES)(Double.MinValue)
  val minFeatures = Array.fill[Double](NO_OF_FEATURES)(Double.MaxValue)

  val attributes: java.util.ArrayList[weka.core.Attribute] = this.getAttributes

  val sample: ListBuffer[SamplePairT] = ListBuffer[SamplePairT]()

  var frequencyMap: CandidateSet = CandidateSet()
  var distinctCooccurrences: CandidateSet = CandidateSet()
  var totalCooccurrences: CandidateSet = CandidateSet()
  var realCandidates: CandidateSet = CandidateSet()

  val clf: weka.classifiers.Classifier = new weka.classifiers.functions.Logistic()

  def getDatasetDelimiter: Int = datasetDelimiter

  def getSampleSize: Int = SAMPLE_SIZE

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

  def updateSourceStats(): Unit = {
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

  def updateCandStats(cID: Int): Int = {
    distinctCooccurrences.increment(cID, 1)
    totalCooccurrences.increment(cID, frequencyMap.get(cID))
    frequencyMap.get(cID)
  }

  def updateIntersectionStats(s: EntityT, t: EntityT, cID: Int): Int = {
    val intersect = s.getEnvelopeInternal.intersects(t.getEnvelopeInternal)
    var retVal = 0
    if (intersect) {
      realCandidates.increment(cID, 1)
      val interMBR = s.getEnvelopeInternal.intersection(t.getEnvelopeInternal).getArea
      update(IntersectionArea.value, interMBR)
      update(CommonTiles.value, frequencyMap.get(cID))
      retVal = 1
    }
    retVal
  }

  def computeRowVector(label: Int, p: SamplePairT, candSet: Set[Int]) : Array[Double] = {
    val dfRow = Array.fill[Double](NO_OF_FEATURES+1)(0.0)

    val sID = p.getSourceId
    val s = p.getSourceGeometry
    val t = p.getTargetGeometry

    var co_occurrences = 0
    var d_co_occurrences = 0
    var t_co_occurrences = 0
    candSet.foreach { cID =>
      val c = source(cID)
      co_occurrences += frequencyMap.get(cID)
      d_co_occurrences += 1
      val intersects = c.getEnvelopeInternal.intersects(t.getEnvelopeInternal)
      if (intersects) t_co_occurrences +=1
    }

    val interMBR = s.getEnvelopeInternal.intersection(t.getEnvelopeInternal)
    //area-based features
    dfRow(SourceArea.value) = (s.getEnvelopeInternal.getArea - minFeatures(0)) / maxFeatures(0) * 10000
    dfRow(TargetArea.value) = (t.getEnvelopeInternal.getArea - minFeatures(1)) / maxFeatures(1) * 10000
    dfRow(IntersectionArea.value) = (interMBR.getArea - minFeatures(2)) / maxFeatures(2) * 10000

    //grid-based features
    dfRow(SourceTiles.value) = (s.getNumOfOverlappingTiles(THETA) - minFeatures(3)) / maxFeatures(3) * 10000
    dfRow(TargetTiles.value) = (t.getNumOfOverlappingTiles(THETA) - minFeatures(4)) / maxFeatures(4) * 10000
    dfRow(CommonTiles.value) = (frequencyMap.get(sID) - minFeatures(5)) / maxFeatures(5) * 10000

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
    dfRow(TargetTotalCooccurrences.value) = (co_occurrences - minFeatures(13)) / maxFeatures(13) * 10000
    dfRow(TargetDistCooccurrences.value) = (d_co_occurrences - minFeatures(14)) / maxFeatures(14) * 10000
    dfRow(SourceRealCooccurrences.value) = (t_co_occurrences - minFeatures(15)) / maxFeatures(15) * 10000

    dfRow(Label.value) = label.toDouble
    dfRow
  }

  def build(trainClass: Int, trainPairs: scala.collection.mutable.ListBuffer[SamplePairT],
            candidates: Seq[Set[Int]], sz: Int): Seq[Array[Double]] =  {
    for (idx <- 0 until  sz) yield computeRowVector(trainClass, trainPairs(idx), candidates(idx))
  }

  def trainSampling : (Int, ListBuffer[SamplePairT], ListBuffer[SamplePairT]) = {
    var excessVerifications = 0
    var negativeClassFull = false
    var positiveClassFull = false
    val negativePairs = scala.collection.mutable.ListBuffer[SamplePairT]()
    val positivePairs = scala.collection.mutable.ListBuffer[SamplePairT]()
    scala.util.Random.shuffle(sample)

    var idx = 0
    while (sample.size > 0 && idx < sample.size && !negativeClassFull && !positiveClassFull) {
      val pair = sample(idx)
      val s = pair.getSourceGeometry
      val t = pair.getTargetGeometry
      val rel = s.geometry.relate(t.geometry)
      val im = IM(s, t, rel)

      if (im.relate) {
        if (positivePairs.size < CLASS_SIZE) {
          positivePairs += pair
        } else {
          excessVerifications += 1
          positiveClassFull = true
        }
      } else {
        if (negativePairs.size < CLASS_SIZE) {
          negativePairs += pair
        } else {
          excessVerifications += 1
          negativeClassFull = true
        }
      }
      idx += 1
    }
    (excessVerifications, positivePairs, negativePairs)
  }

  def setInstance(featureVector: Array[Double], trainSet: weka.core.Instances, weight: Double): weka.core.Instance = {
    val instance = new weka.core.DenseInstance(weight, featureVector)
    instance.setDataset(trainSet)
    instance
  }

  def train(posCands: Seq[Set[Int]], negCands: Seq[Set[Int]], pPairs: ListBuffer[SamplePairT],
            nPairs: ListBuffer[SamplePairT], sz: Int): Unit = {
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
  }
}
