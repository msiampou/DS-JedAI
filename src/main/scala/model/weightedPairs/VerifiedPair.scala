package model.weightedPairs

case class VerifiedPair(sourceId: Int,
                        targetId: Int)  {

    override def toString: String = s" s: $sourceId t: $targetId"

    def getSourceId: Int = sourceId

    def getTargetId: Int = targetId
}

