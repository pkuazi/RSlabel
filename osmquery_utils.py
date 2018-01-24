import logging
import time
import overpass

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
        return str(self.left)+','+str(self.bottom)+','+str(self.right)+','+str(self.top)

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


if __name__ == '__main__':
    # define query conditions
    bbox = Bbox(39.685,116.085, 40.18,116.71)
    tag = {Tag('landuse','residential'), Tag('landuse','industrial')} #https://taginfo.openstreetmap.org/keys

    # Example of using the Class OverpassApi
    overpassapi = OverpassApi()
    query = overpassapi._get_query(bbox, tag, False, True, True)
    results = overpassapi._try_overpass_download(query)

    # Example of using the function osm_query_by_overpass
    # overpass = overpass.API(timeout=60)
    # results = osm_query_by_overpass(bbox, tag, False, True, True)

    print("there are %s geometry in geojson"%len(results['features']))
    print(results['features'][0].keys())
    print(results['features'][0]['id'])
    print(results['features'][0]['properties'])

