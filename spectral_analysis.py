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

def process_TM_metadata(fname):
    fp = open(fname, 'r') # Open metadata file
    data = fp.read()
    files = re.findall(r'FILE_NAME.*?"(.*?)"',data)
    bandfile_list = []
    for file in files:
        if file.endswith(".TIF") or file.endswith('.tif'):
            bandfile_list.append(file)

    band_num = len(bandfile_list)
    lmax=re.findall(r'LMAX_.*?= +(.*)',data)[:band_num]
    lmin=re.findall(r'LMIN_.*?= +(.*)',data)[:band_num]
    qc_lmin=re.findall(r'QCALMIN_.*?= +(.*)',data)[:band_num]
    qc_lmax = re.findall(r'QCALMAX_.*?= +(.*)',data)[:band_num]

    return (bandfile_list, lmax, lmin, qc_lmax, qc_lmin)


# run dn_2_rad.py
def dn_2_rad(metadata_file):
    bandfile_list, lmax_list, lmin_list, qc_lmax_list, qc_lmin_list = process_TM_metadata(metadata_file)
    num = 0
    for band in bandfile_list:
        band_path = os.path.join("/mnt/win/water_paper/spectral_sample/data/image", band)
        dst_file = "/mnt/win/water_paper/spectral_sample/data/image/%s_c.tif" % band[:-4]
        lmax = float(lmax_list[num])
        lmin = float(lmin_list[num])
        qc_lmax = float(qc_lmax_list[num])
        qc_lmin = float(qc_lmin_list[num])

        raster = rasterio.open(band_path, 'r')
        array = raster.read()

        radiance = np.zeros_like(array, dtype=np.float32)
        # DN to radiance conversion if we have a sensible DN
        passer = np.logical_and(qc_lmin < array, array < qc_lmax)
        out_array = np.where(passer, ((lmax - lmin) / (qc_lmax - qc_lmin)) * ((array) - qc_lmin) + lmin, -999)

        max = np.max(out_array)
        print(max)

        # in_array = array!=0
        #
        # in_array = array()
        # out_array = (lmax-lmin)/(qc_lmax-qc_lmin)*(array-qc_lmin)+lmin

        with rasterio.open(dst_file, 'w', driver='GTiff', width=out_array.shape[2], count=1,
                           height=out_array.shape[1], crs=raster.crs,
                           transform=raster.transform, dtype=np.float32, nodata=-999) as dst:
            dst.write(out_array.astype(np.float32))



def spectral_stats(shp_dir, img_dir, features):
    stats = {}
    for feat_name in features:
        print("processing feature of %s" % feat_name)
        stats[feat_name] = {}
        for file in os.listdir(shp_dir):
            if file.startswith(feat_name) and file.endswith(".shp"):
                feat_shp = os.path.join(shp_dir, file)
                print(feat_shp)
                dr = ogr.GetDriverByName("ESRI Shapefile")
                shp_ds = dr.Open(feat_shp)
                layer = shp_ds.GetLayer(0)
                feat_num = layer.GetFeatureCount()

                num = 0
                sum = 0

                rst_start = re.findall(r'-(.*?).shp', file)[0]

                for rst in os.listdir(img_dir):
                    # to find the corresponding landsat image
                    if rst.startswith(rst_start) and rst.endswith("_c.tif"):
                        print("there are %s polygons for %s" % (feat_num, rst))
                        image_rst = os.path.join(img_dir, rst)
                        raster = rasterio.open(image_rst, 'r')

                        for i in range(feat_num):
                            feat = layer.GetFeature(i)
                            geom = feat.GetGeometryRef()
                            minx, maxx, miny, maxy = geom.GetEnvelope()
                            ll = raster.index(minx, miny)
                            ur = raster.index(maxx, maxy)

                            # read the subset of the data into a numpy array
                            window = ((ur[0], ll[0] + 1), (ll[1], ur[1] + 1))
                            data = raster.read(1, window=window)

                            num += np.count_nonzero(data[data != -999])
                            sum += np.sum(data[data != -999])


                        tif_band = rst[:-6]
                        feat_avg = sum / num
                        # print(num, sum, feat_avg)
                        stats[feat_name][tif_band] = feat_avg

    return stats

def test_dn_2_rad():
    metadata_file = "/mnt/win/water_paper/spectral_sample/data/image/L5134040_04020051111_MTL.txt"
    dn_2_rad(metadata_file)

def test_spectral_stats():
    # the shapefile and image must have same coordinate system
    shp_dir = "/mnt/win/water_paper/spectral_sample/data/feature_shp"
    img_dir = "/mnt/win/water_paper/spectral_sample/data/image"

    features = ['cloud-', 'cloud_shadow-', 'mount_shadow-', 'snow-', 'water-']

    stats = spectral_stats(shp_dir, img_dir, features)
    print(stats)


if __name__ == '__main__':
    # step 1 converts from DN to radiance using the methodology given by <http://landsat.usgs.gov/how_is_radiance_calculated.php>
    # test_dn_2_rad()

    # step 2 read each band radiance for each pixel
    # shp'name starts with features-, endswith L5124036, to find corresponding landsat images
    test_spectral_stats()


