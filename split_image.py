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
from geomtrans import GeomTrans
import os


def mask_image(geomjson, geomproj, raster, tag, name, size=(128,128)):
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
    window = ((ur[0], ll[0] + 1), (ll[1], ur[1] + 1))  # rows, cols of ((xmax, xmin+1),(ymin, ymax+1))
    window_shape = ((window[0][1] - window[0][0]),
                    (window[1][1] - window[1][0]))  # or data = raster.read(1,window=window); out_shape = data.shape;


    # to generate 512*512 size mask
    if window_shape[0] <= size[0] and window_shape[1] <= size[1]:
        mask = rasterio.features.rasterize([(geometry, 0)], out_shape=window_shape, transform=shifted_affine, fill=1,
                                           all_touched=True, dtype=np.uint8)
        out_shape = np.zeros((size[0], size[1]))
        # expand the window of rasterized polygon into specified size
        mask_expanded = np.pad(mask, ((0, size[0] - window_shape[0]), (0, size[1] - window_shape[1])), mode='constant',
                               constant_values=1)

        # generate label array
        label_array = np.empty_like(mask_expanded,dtype=np.float)
        label_array[mask_expanded == 0] = tag
        with rasterio.open("/tmp/label_%s.tif" % name, 'w', driver='GTiff', width=size[1], height=size[0],
                           crs=raster.crs, transform=shifted_affine, dtype=np.uint16, nodata=256, count=1) as dst:
            dst.write(label_array.astype(rasterio.uint16), 1)

        # read the original rs image into specified size
        window_expanded = ((window[0][0], window[0][0] + size[0]), (window[1][0], window[1][0] + size[1]))
        out_data = raster.read(window= window_expanded)
        with rasterio.open("/tmp/mask_%s2.tif" % name, 'w', driver='GTiff', width=size[1],height=size[0], crs=raster.crs, transform=shifted_affine, dtype=np.uint16,nodata=256,count=raster.count, indexes=raster.indexes) as dst:
            # Write the src array into indexed bands of the dataset. If `indexes` is a list, the src must be a 3D array of matching shape. If an int, the src must be a 2D array.
            dst.write(out_data.astype(rasterio.uint16), indexes =raster.indexes)

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
    window = ((ur[0], ll[0] + 1), (ll[1], ur[1] + 1))  # rows, cols of ((xmax, xmin+1),(ymin, ymax+1))
    window_shape = ((window[0][1] - window[0][0]),
                    (window[1][1] - window[1][0]))  # or data = raster.read(1,window=window); out_shape = data.shape;


    # to generate 512*512 size mask
    if window_shape[0] <= 512 and window_shape[1] <= 512:
        mask = rasterio.features.rasterize([(geometry, 0)], out_shape=window_shape, transform=shifted_affine, fill=1,
                                           all_touched=True, dtype=np.uint8)
        out_shape = np.zeros((512, 512))
        # expand the window of rasterized polygon into 512*512
        mask_expanded = np.pad(mask, ((0, 512 - window_shape[0]), (0, 512 - window_shape[1])), mode='constant',
                               constant_values=1)

        # generate label array
        label_array = np.empty_like(mask_expanded,dtype=np.float)
        label_array[mask_expanded == 0] = tag
        with rasterio.open("/tmp/label_%s.tif" % name, 'w', driver='GTiff', width=512, height=512,
                           crs=raster.crs, transform=shifted_affine, dtype=np.uint16, nodata=256, count=1) as dst:
            dst.write(label_array.astype(rasterio.uint16), 1)

        # read the original rs image in 512*512
        window_expanded = ((window[0][0], window[0][0] + 512), (window[1][0], window[1][0] + 512))
        out_data = raster.read(window= window_expanded)
        with rasterio.open("/tmp/mask_%s2.tif" % name, 'w', driver='GTiff', width=512,height=512, crs=raster.crs, transform=shifted_affine, dtype=np.uint16,nodata=256,count=raster.count, indexes=raster.indexes) as dst:
            # Write the src array into indexed bands of the dataset. If `indexes` is a list, the src must be a 3D array of matching shape. If an int, the src must be a 2D array.
            dst.write(out_data.astype(rasterio.uint16), indexes =raster.indexes)



    window_expanded = ((window[0][0], window[0][0] + 512), (window[1][0], window[1][0] + 512))
    bands_num = raster.count
    multi_bands_data = []
    for i in range(bands_num):
        # band_name = raster.indexes[i]
        data = raster.read(i + 1, window=window_expanded)
        # rasterize the geometry
        # mask = rasterio.features.rasterize([(geometry, 0)], out_shape=data.shape, transform=shifted_affine, fill=1,
        #                                    all_touched=True, dtype=np.uint8)

        # create a masked numpy array
        masked_data = np.ma.array(data=data, mask=mask_expanded.astype(bool), dtype=np.float32)
        masked_data.data[masked_data.mask] = np.nan  # raster.profile nodata is 256
        out_image = masked_data.data
        multi_bands_data.append(out_image)
    out_data = np.array(multi_bands_data)
    with rasterio.open("/tmp/mask_%s.tif" % name, 'w', driver='GTiff', width=out_data.shape[2],
                       height=out_data.shape[1], crs=raster.crs, transform=shifted_affine, dtype=np.uint16, nodata=256,
                       count=bands_num, indexes=raster.indexes) as dst:
        dst.write(out_image.astype(rasterio.uint16), 1)


if __name__ == '__main__':
    tag = {Tag('landuse', 'residential'), Tag('building', 'residential')}

    # for inner beijing
    bbox = Bbox(39.946, 116.348, 40.006, 116.425)
    # urban_image = os.path.join(data_path, 'urban.tif')
    image = '/mnt/win/image/GF1/add_crs/GF1_PMS2_E116.5_N39.9_20151012_L1A0001093940-MSS2.tiff'

    overpassapi = OverpassApi()
    urban_query = overpassapi._get_query(bbox, tag, False, True, True)
    urban_results = overpassapi._try_overpass_download(urban_query)

    dst_path = os.getcwd() + '/cache'

    geoproj = 'EPSG:4326'
    raster = rasterio.open(image, 'r')
    for geojson in urban_results['features']:
        id = geojson['id']

        geometry = geojson['geometry']

        if geometry['type'] == 'LineString':
            geometry = {'type': 'Polygon', 'coordinates': [geometry['coordinates']]}

        result = mask_image(geometry, geoproj, raster, tag=255, name=id, size = (128,128))
        # result = mask_image_by_geometry(geometry, geoproj, raster, tag=255, name='building')
        if result is None:
            # print("the polygon is not within the raster boundary")
            continue
