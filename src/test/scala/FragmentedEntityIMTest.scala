

import model.TileGranularities
import model.entities.{FragmentedEntity, SpatialEntity}
import org.locationtech.jts.geom.{Geometry, GeometryFactory, Polygon}
import org.locationtech.jts.io.WKTReader
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should._
import utils.Constants.ThetaOption
import utils.decompose.RecursiveFragmentation

class FragmentedEntityIMTest extends AnyFunSuite with Matchers  {

    val wktReader = new WKTReader()
    val geometryFactory = new GeometryFactory()

    def check(g1: Geometry, g2: Geometry, fragmentationF: Geometry => Seq[Geometry]): Boolean ={

        val se1 = SpatialEntity("g1", g1)
        val se2 = SpatialEntity("g2", g2)
        val im = se1.getIntersectionMatrix(se2)

        val fse1 = FragmentedEntity("g1", g1)(fragmentationF)
        val fse2 = FragmentedEntity("g2", g2)(fragmentationF)
        val fim = fse1.getIntersectionMatrix(fse2)

        im == fim
    }


    test("FragmentationIntersectionMatrixTest - Polygon Intersection") {
        val polygon1WKT = "Polygon ((21290.57879912905627862 -11654.62375026853987947, 21126.76324528959958116 9641.39824886050701025, 42218.01580211932741804 10747.1532372768233472, 25672.64486433445563307 8494.68937198432831792, 26205.04541431267716689 1286.80500304834276903, 42177.06191365946870064 4030.71552985920061474, 26942.21540659022139153 -842.79719686456155614, 43241.86301361591904424 -10016.46821187399837072, 30955.69647565685590962 -6207.75658510669018142, 21290.57879912905627862 -11654.62375026853987947))"
        val polygon2WKT = "Polygon ((54135.59734393961844034 -4323.87771595296362648, 48074.42185187981522176 23688.58199059370599571, 34232.00755244592437521 7143.21105280883057276, 45617.18854428800113965 10173.79879883873218205, 36361.6097523588250624 2556.37554530411580345, 50040.20849795326648746 9027.08992196255348972, 33822.46866784729354549 -11367.94653104949975386, 47746.79074420090182684 1327.75889150820876239, 54135.59734393961844034 -4323.87771595296362648))"

        val polygon1 = wktReader.read(polygon1WKT)
        val polygon2 = wktReader.read(polygon2WKT)

//        val lineT = 12000
//        val polygonT = 12000
        val theta = TileGranularities(Seq(polygon1.getEnvelopeInternal, polygon2.getEnvelopeInternal), 2, ThetaOption.AVG_x2)
        val fragmentationF: Geometry => Seq[Geometry] = RecursiveFragmentation.splitBigGeometries(theta)

        check(polygon1, polygon2, fragmentationF) shouldBe true
    }

//    test("FragmentationIntersectionMatrixTest - Complex Polygon Contain") {
//
//        val polygon1WKT = "Polygon ((2263.78355421686819682 1364.81240963855475457, 2276.29246987951910342 -2681.8218072289155316, 6648.15849397590500303 -2619.2772289156628176, 6479.28813253012231144 1352.30349397590407534, 6479.28813253012231144 1352.30349397590407534, 2263.78355421686819682 1364.81240963855475457))"
//        val polygon2WKT = "Polygon ((3683.54548192771153481 126.42975903614478739, 3677.29102409638608151 -1368.3856626506021712, 5584.90066265060340811 -1362.13120481927717265, 5491.08379518072433711 120.17530120481956146, 3683.54548192771153481 126.42975903614478739))"
//
//        val polygon1 = wktReader.read(polygon1WKT)
//        val polygon2 = wktReader.read(polygon2WKT)
//
//        val lineT = 4000
//        val polygonT = 4000
//        val fragmentationF: Geometry => Seq[Geometry] = GeometryUtils.splitBigGeometries(lineT, polygonT)
//
//        check(polygon1, polygon2, fragmentationF) shouldBe true
//    }


    test("FragmentationIntersectionMatrixTest - Intersecting LineStrings") {
        val polygon1WKT = "LineString (-21792.91186064741850714 34582.31632091741630575, -33751.44729092757916078 24180.02865211206881213, -9424.83754576862338581 19838.91647536653181305, -15649.82859166788330185 10091.8910219190074713, -15486.01303782843024237 2228.74443762520240853, -12127.79418411961887614 -4569.60104671214503469, -16223.18303010597446701 -11695.57763872840587283, -26789.28625275077138213 -16528.13647699230205035, -19990.94076841342030093 -22015.95753061402501771)"
        val polygon2WKT = "LineString (-31458.02953717521813815 37039.54962850922311191, -36372.4961523588426644 29749.75748265351285227, -21711.00408372769015841 26637.26195970388289425, -36372.4961523588426644 20330.36313688489462947, -36372.4961523588426644 16726.42095241690185503, -20646.20298377123981481 14023.4643140659063647, -36290.58837543911795365 9682.35213732036936563, -36208.6805985193932429 4276.43886061837838497, -33587.63173708812246332 -228.48886996661167359, -20318.57187609233005787 -4078.15438519378221827, -32277.10730637249071151 -7272.55768506314052502, -34406.70950628539139871 -12350.83985408622538671, -19335.67855305560442503 -17429.12202310930297244, -29164.61178342285711551 -20705.43309989838598995)"

        val polygon1 = wktReader.read(polygon1WKT)
        val polygon2 = wktReader.read(polygon2WKT)

        val theta = TileGranularities(Seq(polygon1.getEnvelopeInternal, polygon2.getEnvelopeInternal), 2, ThetaOption.AVG_x2)
        val fragmentationF: Geometry => Seq[Geometry] = RecursiveFragmentation.splitBigGeometries(theta)
        check(polygon1, polygon2, fragmentationF) shouldBe true
    }


    test("FragmentationIntersectionMatrixTest - polygon contains multipolygon") {
        val polygon1WKT = "Polygon ((-1568.64202055530768121 34930.38736630058701849, -524.16983872591663385 18591.85823625512421131, 33048.15029150448390283 17845.80667780556541402, 27228.94813559787871782 31946.18113250233000144, -1568.64202055530768121 34930.38736630058701849))"
        val polygon2aWKT = "Polygon ((3434.68343062853728043 21599.93367914066402591, 3598.4989844679912494 19757.00869844680346432, 4765.68480557410293841 20985.62535224271050538, 3434.68343062853728043 21599.93367914066402591))"
        val polygon2bWKT = "Polygon ((26512.19957776165392715 30753.12774992016784381, 23911.62766056031250628 29872.61914803310355637, 23809.24293941065843683 30957.89719221948689665, 26512.19957776165392715 30753.12774992016784381))"

        val polygon1: Polygon = wktReader.read(polygon1WKT).asInstanceOf[Polygon]
        val polygon2a: Polygon = wktReader.read(polygon2aWKT).asInstanceOf[Polygon]
        val polygon2b: Polygon = wktReader.read(polygon2bWKT).asInstanceOf[Polygon]
        val polygon2 = geometryFactory.createMultiPolygon(Array(polygon2a, polygon2b))

        val theta = TileGranularities(Seq(polygon1.getEnvelopeInternal, polygon2.getEnvelopeInternal), 2, ThetaOption.AVG_x2)
        val fragmentationF: Geometry => Seq[Geometry] = RecursiveFragmentation.splitBigGeometries(theta)

        check(polygon1, polygon2, fragmentationF) shouldBe true
    }


    test("FragmentationIntersectionMatrixTest - polygon contains/disjoint multipolygon") {
        val polygon1WKT = "Polygon ((4269.74209901952235668 9571.06420368903127383, 4344.79559299541324435 4167.21263742474729952, 12925.91173757249453047 4017.1056494729618862, 12400.53727974124376487 9496.01070971313674818, 4269.74209901952235668 9571.06420368903127383))"
        val polygon2aWKT = "Polygon ((3244.01101468231900071 11372.3480591104580526, 1542.79848456208310381 10146.47432417087475187, 942.37053275493781257 11097.15191453218540119, 3244.01101468231900071 11372.3480591104580526))"
        val polygon2bWKT = "Polygon ((10524.19993034392427944 5843.4073362196868402, 10048.86113516326986428 4917.74757718367527559, 11800.10932793410211161 4842.69408320778256893, 10524.19993034392427944 5843.4073362196868402))"

        val polygon1: Polygon = wktReader.read(polygon1WKT).asInstanceOf[Polygon]
        val polygon2a: Polygon = wktReader.read(polygon2aWKT).asInstanceOf[Polygon]
        val polygon2b: Polygon = wktReader.read(polygon2bWKT).asInstanceOf[Polygon]
        val polygon2 = geometryFactory.createMultiPolygon(Array(polygon2a, polygon2b))

        val theta = TileGranularities(Seq(polygon1.getEnvelopeInternal, polygon2.getEnvelopeInternal), 2, ThetaOption.AVG_x2)
        val fragmentationF: Geometry => Seq[Geometry] = RecursiveFragmentation.splitBigGeometries(theta)

        check(polygon1, polygon2, fragmentationF) shouldBe true
    }


    test("FragmentationIntersectionMatrixTest - polygon contains/touches multipolygon") {
        val polygon1WKT = "Polygon ((7192.61657727138481278 11218.02512061091147189, 7607.81505329928404535 41665.91336265712743625, 9532.88725579241145169 44250.11354752561601344, 67203.05159755019121803 29523.42440762255864684, 18413.86772947440476855 13784.97799856585334055, 7192.61657727138481278 11218.02512061091147189))"
        val polygon2aWKT = "Polygon ((7607.81505329928404535 41665.91336265712743625, 7192.61657727138481278 11218.02512061091147189, 1213.99415386244072579 9850.366396301673376, -11151.92802468211448286 16482.99738297556905309, 7607.81505329928404535 41665.91336265712743625))"
        val polygon2bWKT = "Polygon ((27794.37596947213751264 36660.35842501542356331, 27794.37596947213751264 36660.35842501542356331, 29817.89051201731490437 43461.61563745894818567, 35045.30308025903650559 36660.35842501542356331, 27794.37596947213751264 36660.35842501542356331))"

        val polygon1: Polygon = wktReader.read(polygon1WKT).asInstanceOf[Polygon]
        val polygon2a: Polygon = wktReader.read(polygon2aWKT).asInstanceOf[Polygon]
        val polygon2b: Polygon = wktReader.read(polygon2bWKT).asInstanceOf[Polygon]
        val polygon2 = geometryFactory.createMultiPolygon(Array(polygon2a, polygon2b))

        val theta = TileGranularities(Seq(polygon1.getEnvelopeInternal, polygon2.getEnvelopeInternal), 2, ThetaOption.AVG_x2)
        val fragmentationF: Geometry => Seq[Geometry] = RecursiveFragmentation.splitBigGeometries(theta)
        check(polygon1, polygon2, fragmentationF) shouldBe true
    }



}
