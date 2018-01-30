#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-01-30 14:20
#

# label size is 512*512
from osmquery_utils import Tag, Bbox, OverpassApi
import numpy as np
import fiona, shapely
import rasterio
import rasterio.features
from affine import Affine
from shapely.geometry import shape
# from shapely.ops import transform
from geomtrans import GeomTrans
from shp2json import Shp2Json
import os

def mask_image_by_geometry(geomjson, geomproj, raster, tag, name):
    print('the %s geometry' % name)
    transform = GeomTrans(str(geomproj), str(raster.crs.wkt))
    geojson_crs_transformed = transform.transform_points(geomjson['coordinates'][0])
    geometry = shape({'coordinates': [geojson_crs_transformed], 'type': 'Polygon'})

    bbox = raster.bounds
    extent = [[bbox.left, bbox.top], [bbox.left, bbox.bottom], [bbox.right, bbox.bottom], [bbox.right, bbox.top]]
    raster_boundary = shape({'coordinates': [extent], 'type': 'Polygon'})

    if not geometry.intersects(raster_boundary):
        return

    # get pixel coordinates of the geometry's bounding box,
    ll = raster.index(*geometry.bounds[0:2])  # lowerleft bounds[0:2] xmin, ymin
    ur = raster.index(*geometry.bounds[2:4])  # upperright bounds[2:4] xmax, ymax

    # create an affine transform for the subset data
    t = raster.transform
    shifted_affine = Affine(t.a, t.b, t.c + ll[1] * t.a, t.d, t.e, t.f + ur[0] * t.e)

    # read the subset of the data into a numpy array
    window = ((ur[0], ll[0] + 1), (ll[1], ur[1] + 1))
    bands_num = raster.count
    multi_bands_data = []
    for i in range(bands_num):
        # band_name = raster.indexes[i]
        data = raster.read(i + 1, window=window)
        # rasterize the geometry
        mask = rasterio.features.rasterize([(geometry, 0)], out_shape=data.shape, transform=shifted_affine, fill=1,
                                           all_touched=True, dtype=np.uint8)

        # create a masked numpy array
        masked_data = np.ma.array(data=data, mask=mask.astype(bool), dtype=np.float32)
        masked_data.data[masked_data.mask] = np.nan  # raster.profile nodata is 256
        out_image = masked_data.data
        multi_bands_data.append(out_image)
    out_data = np.array(multi_bands_data)
    with rasterio.open("/tmp/mask_%s.tif" % name, 'w', driver='GTiff', width=out_data.shape[2],
                       height=out_data.shape[1], crs=raster.crs, transform=shifted_affine, dtype=np.uint16, nodata=256,
                       count=bands_num, indexes=raster.indexes) as dst:
        dst.write(out_image.astype(rasterio.uint16), 1)

    # create a masked label numpy array
    label_array = np.zeros_like(data, dtype=np.float32)
    label_array[mask == 0] = tag
    with rasterio.open("/tmp/label_%s.tif" % name, 'w', driver='GTiff', width=out_image.shape[1],
                       height=out_image.shape[0], crs=raster.crs, transform=shifted_affine, dtype=np.uint16, nodata=256,
                       count=1) as dst:
        dst.write(label_array.astype(rasterio.uint16), 1)


if __name__ == '__main__':
    tag = {Tag('landuse', 'residential'), Tag('building','residential')}

    # for inner beijing
    bbox = Bbox(39.946, 116.348, 40.006, 116.425)
    # urban_image = os.path.join(data_path, 'urban.tif')
    image = '/mnt/win/image/GF1/add_crs/GF1_PMS2_E116.5_N39.9_20151012_L1A0001093940-MSS2.tiff'

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

        result = mask_image_by_geometry(geometry, geoproj, raster, tag=1, name = 'building')
        if result is None:
            # print("the polygon is not within the raster boundary")
            continue

