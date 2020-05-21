package EntityMatching

import com.vividsolutions.jts.geom.Geometry
import utils.Constants

trait MatchingTrait extends Serializable{

    /**
     * check the relation between two geometries
     *
     * @param sourceGeom geometry from source set
     * @param targetGeometry geometry from target set
     * @param relation requested relation
     * @return whether the relation is true
     */
    def relate(sourceGeom: Geometry, targetGeometry: Geometry, relation: String): Boolean ={
        relation match {
            case Constants.CONTAINS => sourceGeom.contains(targetGeometry)
            case Constants.INTERSECTS => sourceGeom.intersects(targetGeometry)
            case Constants.CROSSES => sourceGeom.crosses(targetGeometry)
            case Constants.COVERS => sourceGeom.covers(targetGeometry)
            case Constants.COVEREDBY => sourceGeom.coveredBy(targetGeometry)
            case Constants.OVERLAPS => sourceGeom.overlaps(targetGeometry)
            case Constants.TOUCHES => sourceGeom.touches(targetGeometry)
            case Constants.DISJOINT => sourceGeom.disjoint(targetGeometry)
            case Constants.EQUALS => sourceGeom.equals(targetGeometry)
            case Constants.WITHIN => sourceGeom.within(targetGeometry)
            case _ => false
        }
    }

    def normalizeWeight(weight: Double, entity1: Geometry, entity2:Geometry): Double ={
        val area1 = entity1.getArea
        val area2 = entity2.getArea
        if (area1 == 0 || area2 == 0 ) weight
        else weight/(entity1.getArea * entity2.getArea)
    }


}
