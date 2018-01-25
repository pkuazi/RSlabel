#!/usr/bin/env python
# -*-coding:utf-8 -*-
# created by 'root' on 18-1-25

from osmquery_utils import Tag, Bbox, OverpassApi
from maskimage import MaskImage
import rasterio
import os
import numpy as np

if __name__ == '__main__':
    # # define query conditions
    # bbox = Bbox(39.685,116.085, 40.18,116.71)
    # tag = {Tag('landuse','residential'), Tag('landuse','industrial')} #https://taginfo.openstreetmap.org/keys
    #
    # # Example of using the Class OverpassApi
    # overpassapi = OverpassApi()
    # query = overpassapi._get_query(bbox, tag, False, True, True)
    # results = overpassapi._try_overpass_download(query)

    data_path = os.getcwd() + '/data'
    # for inner beijing
    bbox_bjurban = Bbox(39.946, 116.348, 40.006, 116.425)
    urban_image = os.path.join(data_path, 'GF1_PMS2_E116.5_N39.9_20151012_L1A0001093940-MSS2.tiff')

    bbox_bjrural = Bbox(40.095, 116.142, 40.155, 116.224)
    rural_image = os.path.join(data_path, 'GF1_PMS1_E116.1_N40.0_20151012_L1A0001094174-MSS1.tiff')

    tag = {Tag('landuse','residential')}

    overpassapi = OverpassApi()
    urban_query = overpassapi._get_query(bbox_bjurban, tag, False, True, True)
    urban_results = overpassapi._try_overpass_download(urban_query)

    dst_path = os.getcwd() + '/cache'

    geoproj = 'EPSG:4326'
    urban_image = rasterio.open(urban_image, 'r')
    for geojson in urban_results['features']:
        geometry= geojson['geometry']
        maskrs = MaskImage(geometry, geoproj, urban_image)
        shifted_affine = maskrs._get_transform()
        data = maskrs.get_data()
        print(data.shape)

        with rasterio.open("%s/%s.tif" % (dst_path,geojson['id']), 'w', driver='GTiff', width=data.shape[2], height=data.shape[1],
                           crs=urban_image.crs, transform=shifted_affine, dtype=np.uint16, nodata=256, count=data.shape[0],
                           indexes=urban_image.indexes) as dst:
            dst.write(data.astype(rasterio.uint16))

    overpassapi = OverpassApi()
    rural_query = overpassapi._get_query(bbox_bjrural, tag, False, True, True)
    rural_results = overpassapi._try_overpass_download(rural_query)

    rural_image = rasterio.open(rural_image, 'r')
    for geojson in urban_results['features']:
        geometry = geojson['geometry']
        maskrs = MaskImage(geojson, geoproj, rural_image)
        shifted_affine = maskrs._get_transform()
        data = maskrs.get_data()
        print(data.shape)

        with rasterio.open("%s/%s.tif" % (dst_path,geojson['id']), 'w', driver='GTiff', width=data.shape[2], height=data.shape[1],
                           crs=rural_image.crs, transform=shifted_affine, dtype=np.uint16, nodata=256, count=data.shape[0],
                           indexes=rural_image.indexes) as dst:
            dst.write(data.astype(rasterio.uint16))


