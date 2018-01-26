#!/usr/bin/env python
# -*-coding:utf-8 -*-
# created by 'zjh' on 18-1-25

from osmquery_utils import Tag, Bbox, OverpassApi
from maskimage import mask_image_by_geojson_polygon
import rasterio
import os, ogr
import numpy as np
from geojson2shp import geojson2shp

data_path = os.getcwd() + '/cache'

def osm_cut_raster(bbox, tag, image):
    '''

    :param bbox:  the bbox of the boundary of the study area
    :param tag: the tag of the polygons interested
    :param image: the image need to extract AOIs
    :return: the polygons from osm are stored as shapefiles; the images cut by polygon are stored as tiles in Geotiff format
    '''

    overpassapi = OverpassApi()
    urban_query = overpassapi._get_query(bbox, tag, False, True, True)
    urban_results = overpassapi._try_overpass_download(urban_query)

    dst_path = os.getcwd() + '/cache'

    geoproj = 'EPSG:4326'
    raster  = rasterio.open(image, 'r')
    for geojson in urban_results['features']:
        id = geojson['id']

        geometry = geojson['geometry']

        if geometry['type'] == 'LineString':
            geometry = {'type': 'Polygon', 'coordinates': [geometry['coordinates']]}

        result = mask_image_by_geojson_polygon(geometry, geoproj, raster)
        if result is None:
            # print("the polygon is not within the raster boundary")
            continue

        shpdst = "%s/%s.shp" % (dst_path, geojson['id'])
        geojson2shp(geometry, shpdst, id)

        # the data cut by the polygon, and its geotransform
        data = result[0]
        shifted_affine = result[1]

        print(data.shape)

        with rasterio.open("%s/%s.tif" % (dst_path, geojson['id']), 'w', driver='GTiff', width=data.shape[2],
                           height=data.shape[1],
                           crs=raster.crs, transform=shifted_affine, dtype=np.uint16, nodata=256,
                           count=data.shape[0],
                           indexes=raster.indexes) as dst:
            dst.write(data.astype(rasterio.uint16))

if __name__ == '__main__':
    # # define query conditions
    # bbox = Bbox(39.685,116.085, 40.18,116.71)
    # tag = {Tag('landuse','residential'), Tag('landuse','industrial')} #https://taginfo.openstreetmap.org/keys
    #
    # # Example of using the Class OverpassApi
    # overpassapi = OverpassApi()
    # query = overpassapi._get_query(bbox, tag, False, True, True)
    # results = overpassapi._try_overpass_download(query)

    tag = {Tag('landuse', 'residential')}

    # for inner beijing
    # bbox_bjurban = Bbox(39.946, 116.348, 40.006, 116.425)
    # urban_image = os.path.join(data_path, 'urban.tif')
    # # urban_image = '/mnt/win/image/GF1/add_crs/GF1_PMS2_E116.5_N39.9_20151012_L1A0001093940-MSS2.tiff'
    #
    # osm_cut_raster(bbox_bjurban, tag, urban_image)

    # for northwest rurual of beijing
    bbox_bjrural = Bbox(40.095, 116.142, 40.155, 116.224)
    rural_image = os.path.join(data_path, 'rural.tif')
    osm_cut_raster(bbox_bjrural, tag, rural_image)





    # overpassapi = OverpassApi()
    # rural_query = overpassapi._get_query(bbox_bjrural, tag, False, True, True)
    # rural_results = overpassapi._try_overpass_download(rural_query)
    #
    # rural_image = rasterio.open(rural_image, 'r')
    # for geojson in urban_results['features']:
    #     geometry = geojson['geometry']
    #     maskrs = MaskImage(geojson, geoproj, rural_image)
    #     shifted_affine = maskrs._get_transform()
    #     data = maskrs.get_data()
    #     print(data.shape)
    #
    #     with rasterio.open("%s/%s.tif" % (dst_path,geojson['id']), 'w', driver='GTiff', width=data.shape[2], height=data.shape[1],
    #                        crs=rural_image.crs, transform=shifted_affine, dtype=np.uint16, nodata=256, count=data.shape[0],
    #                        indexes=rural_image.indexes) as dst:
    #         dst.write(data.astype(rasterio.uint16))


