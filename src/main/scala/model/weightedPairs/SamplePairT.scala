package model.weightedPairs

import model.entities.EntityT

case class SamplePairT (sourceId: Int,
                        targetId: Int,
                        geometryS: EntityT,
                        geometryT: EntityT)  {

    override def toString: String = s" s: $sourceId t: $targetId"

    def getSourceId: Int = sourceId

    def getTargetId: Int = targetId

    def getSourceGeometry: EntityT = geometryS

    def getTargetGeometry: EntityT = geometryT
}

