import osmium as osm
import os, json
import pandas as pd

#region.json
# region = \
# {
#     'extracts': [{
#         'output':'beijing-inner.osh.pbf',
#         'output_format':'osh.pbf',
#         'description':'extract OSM history for Beijing (China)',
#         'bbox':{
#             'left':116.085,
#             'right':116.71,
#             'top':40.18,
#             'bottom':39.685
#         }
#     }],
#     'directory': '/mnt/win/RSlabel/osmdata/'
#
# }
# osmfile = 'latest-planet.osh.pbf' # need to be downloaded before
# region_json = json.dumps(region)
# dowdload_cmd = 'osmium extract --with-history --config=%s %s'% region_json, osmfile
# os.system(dowdload_cmd)
/root/workspace/dataservice/labelRS/osm_parse.py
class OSMHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.osm_data = []

    def tag_inventory(self, elem, elem_type):
        # if elem_type=='way' and len(elem.tags)>0:
        #     print(elem_type, elem)
        for tag in elem.tags:
            # print(elem)
            self.osm_data.append([elem_type, elem.id, elem.version, elem.visible, pd.Timestamp(elem.timestamp), elem.uid, elem.user, elem.changeset, len(elem.tags), tag.k, tag.v])

    def node(self, n):
        self.tag_inventory(n,'node')

    def way(self, w):
        self.tag_inventory(w, 'way')

    def relation(self, r):
        self.tag_inventory(r, 'relation')

osmhandler = OSMHandler()
# scan the input file and fills the handler list accordingly
osm_file = '/mnt/win/RSlabel/osmdata/bj.osm'
osmhandler.apply_file(osm_file)


# transform the list into a pandas dataframe
data_colnames = ['type', 'id', 'version', 'visible', 'ts', 'uid', 'user', 'chgset', 'ntags', 'tagkey', 'tagvalue']
df_osm = pd.DataFrame(osmhandler.osm_data, columns=data_colnames)
df_osm = df_osm.sort_values(by=['type','id','ts'])

print(df_osm.head())
# metadata = {}
#         for i in range(1, len(metadata_content)):
#             att = metadata_content[i]
#             key = att.find_all('td')[0].text.strip('\n').strip()
#             value = att.find_all('td')[1].text
#             metadata[key.lower()] = value
#         print(metadata)
# # metadata_insertDB_Landsat8(metadata)
#         db_metadata = {"entityid": "", "day_night": "", "lines": "", "samples": "", "station_id": "", "path": "",
#                        "row": "", "date_acquired": "", "start_time": "", "stop_time": "", "image_quality": "",
#                        "cloud_cover": "", "sun_elevation": "", "sun_azimuth": "", "file_size": "",
#                        "scene_center_lon": "", "scene_center_lat": "", "corner_ul_lon": "", "corner_ul_lat": "",
#                        "corner_ur_lon": "", "corner_ur_lat": "", "corner_lr_lon": "", "corner_lr_lat": "",
#                        "corner_ll_lon": "", "corner_ll_lat": ""}
#         db_metadata["entityid"] = metadata['landsat scene identifier']
#         db_metadata["day_night"] = metadata['day/night indicator']
#         db_metadata["lines"] = float(metadata["reflective lines"])
#         '''Panchromatic Lines
#         Reflective Lines
#         Thermal Lines'''
#         db_metadata["samples"] = float(metadata["reflective samples"])
#         '''Panchromatic Samples
#         Reflective Samples
#         Thermal Samples'''
#         db_metadata["station_id"] = metadata["station identifier"]
#         db_metadata["path"] = int(metadata["wrs path"])
#         db_metadata["row"] = int(metadata["wrs row"])
#         db_metadata["date_acquired"] = metadata["acquisition date"]
#         db_metadata["start_time"] = metadata["start time"]
#         db_metadata["stop_time"] = metadata["stop time"]
#         db_metadata["image_quality"] = metadata["image quality"]
#         db_metadata["cloud_cover"] = float(metadata["scene cloud cover"])
#         db_metadata["sun_elevation"] = float(metadata["sun elevation"])
#         db_metadata["sun_azimuth"] = float(metadata["sun azimuth"])
#         db_metadata["file_size"] = 0
#
#         db_metadata["scene_center_lon"] = float(metadata["center longitude dec"])
#         db_metadata["scene_center_lat"] = float(metadata["center latitude dec"])
#         db_metadata["corner_ul_lon"] = float(metadata["nw corner long dec"])
#         db_metadata["corner_ul_lat"] = float(metadata["nw corner lat dec"])
#
#         db_metadata["corner_ur_lon"] = float(metadata["ne corner long dec"])
#         db_metadata["corner_ur_lat"] = float(metadata["ne corner lat dec"])
#         db_metadata["corner_lr_lon"] = float(metadata["se corner long dec"])
#         db_metadata["corner_lr_lat"] = float(metadata["se corner lat dec"])
#
#         db_metadata["corner_ll_lon"] = float(metadata["sw corner long dec"])
#         db_metadata["corner_ll_lat"] = float(metadata["sw corner lat dec"])

