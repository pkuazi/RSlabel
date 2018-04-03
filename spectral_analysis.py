#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-04-03 09:33
#
import os,re
import rasterio
import ogr
import numpy as np

data_dir ="/mnt/win/water_paper/spectral_sample/data"

# step 1 converts from DN to radiance using the methodology given by
#     <http://landsat.usgs.gov/how_is_radiance_calculated.php>

# run dn_2_rad.py
# step 2 read each band radiance for each pixel
def read_pixels(shp, tif):
    dr = ogr.GetDriverByName("ESRI Shapefile")
    shp_ds = dr.Open(shp)
    layer = shp_ds.GetLayer(0)

    raster = rasterio.open(tif, 'r')
    feat_num = layer.GetFeatureCount()
    print("there are %s polygons for %s"%(feat_num, tif))
    num = 0
    sum = 0
    for i in range(feat_num):
        feat = layer.GetFeature(i)
        geom = feat.GetGeometryRef()
        minx, maxx, miny, maxy = geom.GetEnvelope()
        ll = raster.index(minx, miny)
        ur = raster.index(maxx, maxy)

        # read the subset of the data into a numpy array
        window = ((ur[0], ll[0] + 1), (ll[1], ur[1] + 1))
        data = raster.read(1, window=window)

        num += np.count_nonzero(data[data!=-999])
        sum += np.sum(data[data!=-999])

    return num, sum


if __name__ == '__main__':
    # the shapefile and image have same coordinate system
    shp_dir = "/mnt/win/water_paper/spectral_sample/data/feature_shp"
    img_dir = "/mnt/win/water_paper/spectral_sample/data/image"

    features = ['cloud', 'cloud_shadow', 'mount_shadow', 'snow', 'water']
    stats = {}
    for feat in features:
        print("processing feature of %s"% feat)
        stats[feat]={}
        for file in os.listdir(shp_dir):
            if file.startswith(feat) and file.endswith(".shp"):
                feat_shp = os.path.join(shp_dir, file)
                rst_start = re.findall(r'-(.*?).shp',file)[0]

                for rst in os.listdir(img_dir):
                    if rst.startswith(rst_start) and rst.endswith("_c.tif"):
                        image_rst = os.path.join(img_dir,rst)
                        num, sum = read_pixels(feat_shp, image_rst)
                        tif_band = rst[:-6]
                        feat_avg = sum/num
                        print(num, sum, feat_avg)
                        stats[feat][tif_band]=feat_avg
    print(stats)

