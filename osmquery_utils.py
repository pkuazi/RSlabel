import logging
import time
import overpass
import ogr, osr, json

class Tag:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __str__(self):
        return self.key + "=" + self.value
class Bbox:
    def __init__(self, left=None, bottom=None, right=None, top=None):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top

    def __str__(self):
        return str(self.bottom) + ',' + str(self.left) + ',' + str(self.top) + ',' + str(self.right)

class OverpassApi:
    def __init__(self):
        self.overpass = overpass.API(timeout=60)
        self.logger = logging.getLogger(__name__)

    def get(self, bbox, tags):
        query = self._get_query(bbox, tags)
        return self._try_overpass_download(query)

    @staticmethod
    def _get_query(bbox, tags, node=True, way=True, relation=True):
        bbox_string = '(' + str(bbox) + ');'
        query = '('
        for tag in tags:
            if node:
                node = 'node["' + tag.key + '"="' + tag.value + '"]' + bbox_string
                query += node
            if way:
                way = 'way["' + tag.key + '"="' + tag.value + '"]' + bbox_string
                query += way
            if relation:
                relation = 'relation["' + tag.key + '"="' + tag.value + '"]' + bbox_string
                query += relation
            else:
                print("the osm element type must be assigned one")
        query += ');'
        return query

    def _try_overpass_download(self, query):
        for i in range(4):
            try:
                json_data = self.overpass.Get(query)
                return json_data
            except Exception as e:
                self.logger.warning("Download from overpass failed " + str(i) + " wait " + str(i * 10) + ".  " + str(e))
                time.sleep(i * 10)
        error_message = "Download from overpass failed 4 times."
        self.logger.error(error_message)
        raise Exception(error_message)

def storeinshp(osm_results, dst_shp):
    # create the new shp
    dr = ogr.GetDriverByName("ESRI Shapefile")
    ds = dr.CreateDataSource(dst_shp)
    sr = osr.SpatialReference()
    sr.SetFromUserInput('EPSG:4326')
    lyr = ds.CreateLayer("polygon", sr, ogr.wkbPolygon)

    for geojson in osm_results['features']:
        id = geojson['id']
        geometry = geojson['geometry']

        if geometry['type'] == 'LineString':
            geometry = {'type': 'Polygon', 'coordinates': [geometry['coordinates']]}

        corner_points = json.dumps(geometry)

        # geom_json = json.dumps(geom_json)
        geom = ogr.CreateGeometryFromJson(corner_points)

        ffd = ogr.FeatureDefn()
        fgd = ogr.GeomFieldDefn()
        fgd.name = id
        fgd.type = ogr.wkbPolygon
        ffd.AddGeomFieldDefn(fgd)

        fpd = ogr.FieldDefn()
        fpd.name = geojson['properties']
        fpd.type = ogr.OFTString
        ffd.AddFieldDefn(fpd)

        feat = ogr.Feature(ffd)
        feat.SetGeometry(geom)
        lyr.CreateFeature(feat)
# def osm_query_by_overpass(bbox, tags, node=True, way=True, relation=True):
#     logger = logging.getLogger(__name__)
#     bbox_string = '(' + str(bbox) + ');'
#     query = '('
#     for tag in tags:
#         if node:
#             node = 'node["' + tag.key + '"="' + tag.value + '"]' + bbox_string
#             query += node
#         if way:
#             way = 'way["' + tag.key + '"="' + tag.value + '"]' + bbox_string
#             query +=  way
#         if relation:
#             relation = 'relation["' + tag.key + '"="' + tag.value + '"]' + bbox_string
#             query += relation
#         else:
#             print("the osm element type must be assigned one")
#         # query += node + way + relation
#     query += ');'
#
#     print(query)
#
#     for i in range(4):
#         try:
#             json_data = overpass.Get(query)
#             return json_data
#         except Exception as e:
#             logger.warning("Download from overpass failed " + str(i) + " wait " + str(i * 10) + ".  " + str(e))
#             time.sleep(i * 10)
#     error_message = "Download from overpass failed 4 times."
#     logger.error(error_message)
#     raise Exception(error_message)

from shapely.geometry import mapping, Polygon
import fiona
from fiona.crs import from_epsg
def osm2shp(osm_results, dst_shp):

    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int','tags':'str:25'},
    }

    # Write a new Shapefile
    with fiona.open(dst_shp, 'w', crs=from_epsg(4326), driver='ESRI Shapefile', schema=schema) as c:
        ## If there are multiple geometries, put the "for" loop here
        for geojson in osm_results['features']:
            id = geojson['id']
            geometry = geojson['geometry']

            # Here's an example Shapely geometry
            poly = Polygon(geometry['coordinates'])

            c.write({
                'geometry': mapping(poly),
                'properties': {'id': id, 'tags':geojson['properties']},
            })


if __name__ == '__main__':
    # define query conditions
    bbox = Bbox(116.234, 39.89575,116.28955,39.93655)
    tag = {Tag('landuse','residential'), Tag('landuse','industrial'), Tag('building','yes'), Tag('building', 'house'), Tag('type','building'), Tag('building','apartments')} #https://taginfo.openstreetmap.org/keys
    # tag = {Tag('highway','road'), Tag('route','road'), Tag('highway','primary'), Tag('highway','secondary')}

    # Example of using the Class OverpassApi
    overpassapi = OverpassApi()
    query = overpassapi._get_query(bbox, tag, False, True, True)
    results = overpassapi._try_overpass_download(query)

    num = len(results['features'])
    print(num, "polygons for this tag", tag)
    for i in range(num):
        polygon = results['features'][i]['properties']
        print(polygon)
    # Example of using the function osm_query_by_overpass
    # overpass = overpass.API(timeout=60)
    # results = osm_query_by_overpass(bbox, tag, False, True, True)

    # print("there are %s geometry in geojson"%len(results['features']))
    # print(results['features'][0].keys())
    # print(results['features'][0]['id'])
    # print(results['features'][0]['properties'])
    osm2shp(results, '/tmp/building_results.shp' )

