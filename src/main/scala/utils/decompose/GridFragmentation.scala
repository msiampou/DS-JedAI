package utils.decompose

import model.TileGranularities
import org.locationtech.jts.geom._
import org.locationtech.jts.operation.polygonize.Polygonizer
import org.locationtech.jts.operation.union.UnaryUnionOp
import utils.decompose.GeometryUtils.flattenCollection

import scala.collection.JavaConverters._

object GridFragmentation {
    val geometryFactory = new GeometryFactory()
    val epsilon: Double = 1e-8
    val xYEpsilon: (Double, Double) =  (epsilon, 0d)


    def splitBigGeometries(geometry: Geometry, theta: TileGranularities): Seq[Geometry] = {
        geometry match {
            case polygon: Polygon => splitPolygon(polygon, theta)
            case line: LineString => splitLineString(line, theta)
            case gc: GeometryCollection => flattenCollection(gc).flatMap(g => splitBigGeometries(g,theta))
            case _ => Seq(geometry)
        }
    }


    def getVerticalBlades(geom: Geometry, thetaX: Double): Seq[LineString] = {
        val env = geom.getEnvelopeInternal
        val minX = env.getMinX
        val maxX = env.getMaxX
        val n = math.floor(minX / thetaX) + 1
        val bladeStart: Double = n * thetaX

        for (x <- BigDecimal(bladeStart) until maxX by thetaX)
            yield geometryFactory.createLineString(Array(
                    new Coordinate(x.toDouble, env.getMinY - epsilon),
                    new Coordinate(x.toDouble, env.getMaxY + epsilon)
                ))
    }

    def getHorizontalBlades(geom: Geometry, thetaY: Double): Seq[LineString] = {
        val env = geom.getEnvelopeInternal
        val minY = env.getMinY
        val maxY = env.getMaxY
        val n = math.floor(minY/thetaY) +1
        val bladeStart: Double = n*thetaY

        for (y <- BigDecimal(bladeStart) until maxY by thetaY)
            yield geometryFactory.createLineString(Array(
                    new Coordinate(env.getMinX-epsilon, y.toDouble),
                    new Coordinate(env.getMaxX+epsilon, y.toDouble)
                ))
    }


    def combineBladeWithInteriorRings(polygon: Polygon, blade: LineString, innerRings: Seq[Geometry], isHorizontal: Boolean): Seq[LineString] = {

        // epsilon is a small value to add in the segments so to slightly intersect thus not result to dangling lines
        val (xEpsilon, yEpsilon) = if (isHorizontal) xYEpsilon else xYEpsilon.swap

        // the blade is a linestring formed by two points
        val start = blade.getCoordinateN(0)
        val end = blade.getCoordinateN(1)

        // define the cross condition based on line's orientation
        val crossCondition: Envelope => Boolean = if (isHorizontal) env => start.y >= env.getMinY && start.y <= env.getMaxY
                                                     else env => start.x >= env.getMinX && start.x <= env.getMaxX

        // ordering of the coordinates, to define max and min
        implicit val ordering: Ordering[Coordinate] = Ordering.by[Coordinate, Double](c => if (isHorizontal) c.x else c.y)

        // sort inner rings envelope by y
        // find the ones that intersect with the line
        // for each intersecting inner ring find the intersection coordinates,
        //   - sort them and get the first and the last,
        //   - create line segments that do not overlap the inner ring
        var checkpoint = start

        val segments = innerRings
            .map(ir => (ir, ir.getEnvelopeInternal))
            .filter{ case (_, env) => crossCondition(env)}
            .sortBy{ case (_, env) => if (isHorizontal) env.getMinX else env.getMinY }
            .map{ case (ir, _) =>

                val intersectingCollection = ir.intersection(blade)
                val ip: Seq[Coordinate] = (0 until intersectingCollection.getNumGeometries)
                    .map(i => intersectingCollection.getGeometryN(i))
                    .flatMap(g => g.getCoordinates)

                val stopPoint = ip.min(ordering)
                stopPoint.setX(stopPoint.x+xEpsilon)
                stopPoint.setY(stopPoint.y+yEpsilon)

                val newStart = ip.max(ordering)
                newStart.setX(newStart.x-xEpsilon)
                newStart.setY(newStart.y-yEpsilon)

                val segment = geometryFactory.createLineString(Array(checkpoint, stopPoint))

                checkpoint = newStart
                segment
            }.toList

        // create the last line segment
        val segment = geometryFactory.createLineString(Array(checkpoint, end))
        segment ::  segments
    }


    def splitPolygon(polygon: Polygon, theta: TileGranularities): Seq[Polygon] = {

        def split(polygon: Polygon): Seq[Polygon] = {
            val innerRings: Seq[Geometry] = (0 until polygon.getNumInteriorRing).map(i => polygon.getInteriorRingN(i))

            val horizontalBlades = getHorizontalBlades(polygon, theta.y)
                .flatMap(b => combineBladeWithInteriorRings(polygon, b, innerRings, isHorizontal = true))
            val verticalBlades = getVerticalBlades(polygon, theta.x)
                .flatMap(b => combineBladeWithInteriorRings(polygon, b, innerRings, isHorizontal = false))

            val blades: Seq[Geometry] = verticalBlades ++ horizontalBlades ++ innerRings
            val exteriorRing = polygon.getExteriorRing

            val polygonizer = new Polygonizer()
            val union = new UnaryUnionOp(blades.asJava).union()

            polygonizer.add(exteriorRing.union(union))

            val newPolygons = polygonizer.getPolygons.asScala.map(p => p.asInstanceOf[Polygon])
            newPolygons.filter(p => polygon.contains(p.getInteriorPoint)).toSeq
        }

        val env = polygon.getEnvelopeInternal
        if (env.getWidth > theta.x || env.getHeight > theta.y) split(polygon)
        else Seq(polygon)
    }


    def splitLineString(line: LineString, theta: TileGranularities): Seq[LineString] = {

        def split(line: LineString): Seq[LineString] = {
            val horizontalBlades = getHorizontalBlades(line, theta.y)
            val verticalBlades = getVerticalBlades(line, theta.x)
            val blades = geometryFactory.createMultiLineString((verticalBlades ++ horizontalBlades).toArray)
            val lineSegments = line.difference(blades).asInstanceOf[MultiLineString]
            val results = (0 until lineSegments.getNumGeometries).map(i => lineSegments.getGeometryN(i).asInstanceOf[LineString])
            results
        }

        val env = line.getEnvelopeInternal
        if (env.getWidth > theta.x || env.getHeight > theta.y) split(line)
        else Seq(line)
    }


}
