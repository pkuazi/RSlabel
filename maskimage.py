from __future__ import print_function

import numpy as np
import fiona, shapely
import rasterio
import rasterio.features
from affine import Affine
from shapely.geometry import shape
# from shapely.ops import transform
from geomtrans import GeomTrans
from shp2json import Shp2Json


class MaskImage:
    def __init__(self, geomjson=None, geomproj=None, rsimage=None):
        self.geomjson = geomjson
        self.geomproj = geomproj
        self.raster = rsimage
        transform = GeomTrans(str(self.geomproj), str(self.raster.crs.wkt))
        if geomjson['type']=='Polygon':
            geojson_crs_transformed = transform.transform_points(self.geomjson['coordinates'][0])
        elif geomjson['type']=='LineString':
            geojson_crs_transformed = transform.transform_points(self.geomjson['coordinates'])
        print(geojson_crs_transformed)
        geometry = shape({'coordinates': [geojson_crs_transformed], 'type': 'Polygon'})
        self.geometry = geometry

    def get_data(self):
        shifted_affine = self._get_transform()
        window, mask = self._get_mask(shifted_affine)

        bands_num = self.raster.count
        multi_bands_data = []
        for i in range(bands_num):
            # band_name = raster.indexes[i]
            data = raster.read(i + 1, window=window)

            # create a masked numpy array
            masked_data = np.ma.array(data=data, mask=mask.astype(bool), dtype=np.float32)
            masked_data.data[masked_data.mask] = np.nan  # raster.profile nodata is 256
            out_image = masked_data.data
            multi_bands_data.append(out_image)
        out_data = np.array(multi_bands_data)
        return out_data

    def _get_transform(self):
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
        return shifted_affine

    def _get_mask(self, shifted_affine):
        bbox = self.raster.bounds
        extent = [[bbox.left, bbox.top], [bbox.left, bbox.bottom], [bbox.right, bbox.bottom], [bbox.right, bbox.top]]
        raster_boundary = shape({'coordinates': [extent], 'type': 'Polygon'})

        if not self.geometry.intersects(raster_boundary):
            print('shape do not intersect with rs image')
            return

        # get pixel coordinates of the geometry's bounding box,
        ll = self.raster.index(*self.geometry.bounds[0:2])  # lowerleft bounds[0:2] xmin, ymin
        ur = self.raster.index(*self.geometry.bounds[2:4])  # upperright bounds[2:4] xmax, ymax

        # read the subset of the data into a numpy array
        window = ((ur[0], ll[0] + 1), (ll[1], ur[1] + 1))
        data = self.raster.read(1, window=window)
        mask = rasterio.features.rasterize([(self.geometry, 0)], out_shape=data.shape, transform=shifted_affine, fill=1,
                                           all_touched=True, dtype=np.uint8)
        return window, mask

def mask_image_by_geojson_polygon(geojson_polygon, geoproj, raster):
    '''

    :param geojson_polygon: the geojson format of a polygon
    :param geoproj: the projection coordinate system of the input polygon
    :param raster:  the raster data after executing the raster = rasterio.open(raster_image_file, 'r')
    :return: the data cut out from the raster by the polygon, and its geotransformation
    '''
    transform = GeomTrans(str(geoproj), str(raster.crs.wkt))
    geojson_crs_transformed = transform.transform_points(geojson_polygon['coordinates'][0])
    geometry = shape({'coordinates': [geojson_crs_transformed], 'type': 'Polygon'})

    bbox = raster.bounds
    extent = [[bbox.left, bbox.top], [bbox.left, bbox.bottom], [bbox.right, bbox.bottom], [bbox.right, bbox.top]]
    raster_boundary = shape({'coordinates': [extent], 'type': 'Polygon'})

    # if not geometry.intersects(raster_boundary):
    #     return
    if not geometry.within(raster_boundary):
        print('the geometry is not within the raster image')
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
    return out_data, shifted_affine

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

    out_data = raster.read(window=window)
    with rasterio.open("/tmp/mask_%s.tif" % name, 'w', driver='GTiff', width=out_data.shape[2], height=out_data.shape[1], crs=raster.crs,
                       transform=shifted_affine, dtype=np.uint16, nodata=256, count=raster.count,
                       indexes=raster.indexes) as dst:
        # Write the src array into indexed bands of the dataset. If `indexes` is a list, the src must be a 3D array of matching shape. If an int, the src must be a 2D array.
        dst.write(out_data.astype(rasterio.uint16), indexes=raster.indexes)

    # bands_num = raster.count
    # multi_bands_data = []
    # for i in range(bands_num):
    #     # band_name = raster.indexes[i]
    #     data = raster.read(i + 1, window=window)
    #     # rasterize the geometry
    #     mask = rasterio.features.rasterize([(geometry, 0)], out_shape=data.shape, transform=shifted_affine, fill=1,
    #                                        all_touched=True, dtype=np.uint8)
    #
    #     # create a masked numpy array
    #     masked_data = np.ma.array(data=data, mask=mask.astype(bool), dtype=np.float32)
    #     masked_data.data[masked_data.mask] = np.nan  # raster.profile nodata is 256
    #     out_image = masked_data.data
    #     multi_bands_data.append(out_image)
    # out_data = np.array(multi_bands_data)
    # with rasterio.open("/tmp/mask_%s.tif" % name, 'w', driver='GTiff', width=out_data.shape[2],
    #                    height=out_data.shape[1], crs=raster.crs, transform=shifted_affine, dtype=np.uint16, nodata=256,
    #                    count=bands_num, indexes=raster.indexes) as dst:
    #     dst.write(out_image.astype(rasterio.uint16), 1)

    # create a masked label numpy array
    label_array = np.zeros_like(data, dtype=np.float32)
    label_array[mask == 0] = tag
    with rasterio.open("/tmp/label_%s.tif" % name, 'w', driver='GTiff', width=out_image.shape[1],
                       height=out_image.shape[0], crs=raster.crs, transform=shifted_affine, dtype=np.uint16, nodata=256,
                       count=1) as dst:
        dst.write(label_array.astype(rasterio.uint16), 1)


if __name__ == '__main__':
    rasterfile = "/mnt/win/RSlabel/xiaoshan_2013.tif"
    raster = rasterio.open(rasterfile, 'r')

    shapefile = "/mnt/win/RSlabel/xiaoshan2013.shp"
    shp2json = Shp2Json(shapefile)
    geojson_list = shp2json.shp2json_fiona()

    vector = fiona.open(shapefile, 'r')
    geoproj = vector.crs_wkt

    for geojson in geojson_list:
        mask_image_by_geometry(geojson, 'EPSG:4326', raster, 1, 1)

        maskrs = MaskImage(geojson, geoproj, raster)
        shifted_affine = maskrs._get_transform()
        data = maskrs.get_data()
        print(data.shape)

        with rasterio.open("/tmp/mask_%s.tif" % 'band1', 'w', driver='GTiff', width=data.shape[2], height=data.shape[1],
                           crs=raster.crs, transform=shifted_affine, dtype=np.uint16, nodata=256, count=data.shape[0],
                           indexes=raster.indexes) as dst:
            dst.write(data.astype(rasterio.uint16))
            # dst.write(data[0].astype(rasterio.uint16), data.shape[0])
