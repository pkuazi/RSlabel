from __future__ import print_function

import numpy as np
import scipy.stats.mstats
import fiona, shapely
import rasterio
import rasterio.features
from affine import Affine
from shapely.geometry import shape
from shapely.ops import transform
import osr, ogr
from geomtrans import GeomTrans
import json


class MaskRS:
    def __init__(self, geomjson=None, geomproj=None, rsimage=None):
        self.geomjson = geomjson
        self.geomproj = geomproj
        self.raster = rsimage

    def process_geomjson(self):
        transform = GeomTrans(str(self.geomproj), str(self.raster.crs.wkt))
        geojson_crs_transformed = transform.transform_points(self.geomjson['coordinates'][0])
        geometry = shape({'coordinates': [geojson_crs_transformed], 'type': 'Polygon'})
        self.geometry = geometry

    def get_image_mask(self):
        bbox = self.raster.bounds
        extent = [[bbox.left, bbox.top], [bbox.left, bbox.bottom], [bbox.right, bbox.bottom], [bbox.right, bbox.top]]
        raster_boundary = shape({'coordinates': [extent], 'type': 'Polygon'})

        if not self.geometry.intersects(raster_boundary):
            print('shape do not intersect with rs image')
            return

        # get pixel coordinates of the geometry's bounding box,
        ll = self.raster.index(*self.geometry.bounds[0:2])  # lowerleft bounds[0:2] xmin, ymin
        ur = self.raster.index(*self.geometry.bounds[2:4])  # upperright bounds[2:4] xmax, ymax

        # create an affine transform for the subset data
        t = self.raster.transform
        shifted_affine = Affine(t.a, t.b, t.c + ll[1] * t.a, t.d, t.e, t.f + ur[0] * t.e)
        self.shifted_affine = shifted_affine

        # read the subset of the data into a numpy array
        window = ((ur[0], ll[0] + 1), (ll[1], ur[1] + 1))
        self.window = window
        data = self.raster.read(1, window=window)
        mask = rasterio.features.rasterize([(self.geometry, 0)], out_shape=data.shape, transform=shifted_affine, fill=1,
                                           all_touched=True, dtype=np.uint8)
        # return mask
        self.mask = mask

    def image_read_by_mask(self):
        bands_num = self.raster.count
        self.bands_num = bands_num
        multi_bands_data = []
        for i in range(bands_num):
            # band_name = raster.indexes[i]
            data = raster.read(i + 1, window=self.window)

            # create a masked numpy array
            masked_data = np.ma.array(data=data, mask=self.mask.astype(bool), dtype=np.float32)
            masked_data.data[masked_data.mask] = np.nan  # raster.profile nodata is 256
            out_image = masked_data.data
            multi_bands_data.append(out_image)
        out_data = np.array(multi_bands_data)
        return out_data

    def mask_image_to_tif(self, ndarray, dstfilename):
        with rasterio.open("/tmp/mask_%s.tif" % dstfilename, 'w', driver='GTiff', width=ndarray.shape[2],
                           height=ndarray.shape[1], crs=self.raster.crs, transform=self.shifted_affine, dtype=np.uint16,
                           nodata=256, count=self.bands_num, indexes=raster.indexes) as dst:
            dst.write(ndarray.astype(rasterio.uint16), 1)


def main(geomjson, geomproj, raster, tag, name):
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
    shapefile = "/mnt/win/RSlabel/xiaoshan2013.shp"
    rasterfile = "/mnt/win/RSlabel/xiaoshan_2013.tif"

    # shapefile = '/mnt/win/shp/bj_boundary.shp'
    # rasterfile = '/mnt/win/image/LC81230322014279LGN00_B5.TIF'
    raster = rasterio.open(rasterfile, 'r')

    # method1
    # dr = ogr.GetDriverByName("ESRI Shapefile")
    # shp_ds = dr.Open(shapefile)
    # layer = shp_ds.GetLayer(0)
    # shp_proj = layer.GetSpatialRef()
    # shp_proj4 = shp_proj.ExportToProj4()
    # extent = layer.GetExtent()  # minx, maxx, miny,  maxy
    # feat = layer.GetFeature(0)
    # geom = feat.GetGeometryRef()
    # geojson = json.loads(geom.ExportToJson())
    # main(geojson, shp_proj4, raster, 210)

    # method2
    vector = fiona.open(shapefile, 'r')

    for feature in vector:
        # create a shapely geometry
        # this is done for the convenience for the .bounds property only
        # feature['geoemtry'] is in Json format
        geojson = feature['geometry']
        geoproj = vector.crs_wkt
        main(geojson, geoproj, raster, 220, feature['id'])

        # # shapefile = '/mnt/win/shp/bj_boundary.shp'
        # # rasterfile = '/mnt/win/image/LC81230322014279LGN00_B5.TIF'


