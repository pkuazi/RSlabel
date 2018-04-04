import rasterio
import os, re
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
import rasterio.features
import ogr, fiona
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


# class Remote_Classification:
#     def __init__(self, file_root, if_stretch=True):
#         self.file_root = file_root
#         self.img = cv2.imread(self.file_root + '/img.png')
#         self.label = cv2.imread(self.file_root + '/label.png', cv2.IMREAD_UNCHANGED)
#         self.X = self.img.reshape(-1, 3)
#         self.Y = self.label.reshape(1, -1)[0]
#
#         if if_stretch:
#             img_lashen = self.stretch(self.img)
#             self.img = img_lashen
#             self.X = img_lashen.reshape(-1, 3)
#
#         # 建立results文件夹
#         if not os.path.exists(self.file_root + '/results'):
#             os.makedirs(self.file_root + '/results')
#
#     def get_training_data(self):
#         # training data, the features are the original RGB three values.
#         farm = np.argwhere(self.Y == 1).reshape(1, -1)[0]
#         river = np.argwhere(self.Y == 2).reshape(1, -1)[0]
#         road = np.argwhere(self.Y == 3).reshape(1, -1)[0]
#         roof = np.argwhere(self.Y == 4).reshape(1, -1)[0]
#
#         farm_X = self.X[farm]
#         river_X = self.X[river]
#         road_X = self.X[road]
#         roof_X = self.X[roof]
#
#         farm_y = self.Y[farm]
#         river_y = self.Y[river]
#         road_y = self.Y[road]
#         roof_y = self.Y[roof]
#
#         training_X = np.concatenate([farm_X, river_X, road_X, roof_X])
#         training_y = np.concatenate([farm_y, river_y, road_y, roof_y])
#
#         return training_X, training_y
#
#     def SVM(self):
#         if os.path.exists(self.file_root + '/results/Z_svm.npy'):
#             Z_SVM = np.load(self.file_root + '/results/Z_svm.npy')
#         else:
#             training_X, training_y = self.get_training_data()
#             # SVM
#             clf = svm.SVC()
#             clf.fit(training_X, training_y)
#             Z_SVM = clf.predict(self.X).reshape(512, 512)
#             np.save(self.file_root + '/results/Z_svm.npy', Z_SVM)
#         return Z_SVM
#
#     def RandomForest(self):
#         if os.path.exists(self.file_root + '/results/Z_random_forest.npy'):
#             Z_Random_Forest = np.load(self.file_root + '/results/Z_random_forest.npy')
#         else:
#             training_X, training_y = self.get_training_data()
#             # Random Forest
#             clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
#             clf.fit(training_X, training_y)
#             Z_Random_Forest = clf.predict(self.X).reshape(512, 512)
#             np.save(self.file_root + '/results/Z_random_forest.npy', Z_Random_Forest)
#         return Z_Random_Forest
#
#     def KNN(self, n_neighbors):
#         if os.path.exists(self.file_root + '/results/Z_knn%s.npy' % n_neighbors):
#             Z_KNN = np.load(self.file_root + '/results/Z_knn%s.npy' % n_neighbors)
#         else:
#             training_X, training_y = self.get_training_data()
#             # Random Forest
#             clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#             clf.fit(training_X, training_y)
#             Z_KNN = clf.predict(self.X).reshape(512, 512)
#             np.save(self.file_root + '/results/Z_KNN_%s.npy' % n_neighbors, Z_KNN)
#         return Z_KNN
#
#     def Kmeans(self):
#         if os.path.exists(self.file_root + '/results/Z_kmeans.npy'):
#             Z_KMeans = np.load(self.file_root + '/results/Z_kmeans.npy')
#         else:
#             training_X, training_y = self.get_training_data()
#             # K-means
#             kmeans = KMeans(n_clusters=4, random_state=0).fit(training_X)
#             Z_KMeans = kmeans.predict(self.X).reshape(512, 512)
#             np.save(self.file_root + '/results/Z_kmeans.npy', Z_KMeans)
#         return Z_KMeans
#
#     def plot_results(self):
#         plt.figure(figsize=(10, 7))
#
#         ax = plt.subplot(231)
#         ax.set_title("Original")
#         plt.imshow(self.img)
#
#         ax = plt.subplot(232)
#         ax.set_title("Reference Map")
#         plt.imshow(self.label)
#
#         ax = plt.subplot(233)
#         ax.set_title("SVM Result")
#         Z_svm = self.SVM()
#         plt.imshow(Z_svm)
#
#         ax = plt.subplot(234)
#         ax.set_title("Random Forest Result")
#         Z_random_forest = self.RandomForest()
#         plt.imshow(Z_random_forest)
#
#         ax = plt.subplot(235)
#         ax.set_title("KNN Result (k=10)")
#         Z_knn = self.KNN(10)
#         plt.imshow(Z_knn)
#
#         ax = plt.subplot(236)
#         ax.set_title("KNN Result (k=50)")
#         Z_knn = self.KNN(50)
#         plt.imshow(Z_knn)
#         plt.show()
#
#     def stretch(self, bands, lower_percent=2, higher_percent=98):
#         out = np.zeros_like(bands, dtype=np.float32)
#         n = bands.shape[2]
#         print(n)
#         for i in range(n):
#             a = 0
#             b = 1
#             c = np.percentile(bands[:, :, i], lower_percent)
#             d = np.percentile(bands[:, :, i], higher_percent)
#             t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
#             t[t < a] = a
#             t[t > b] = b
#             out[:, :, i] = t
#         return out.astype(np.float32)
#
#     def stretch_plot(self):
#         ax = plt.subplot(221)
#         img = self.img
#         plt.imshow(img)
#         ax = plt.subplot(222)
#         plt.hist(img.reshape(512 * 512, 3), color=['red', 'green', 'blue'])
#         ax = plt.subplot(223)
#         img_ = self.stretch(img)
#         plt.imshow(img_)
#         ax = plt.subplot(224)
#         plt.hist(img_.reshape(512 * 512, 3), color=['red', 'green', 'blue'])
#         plt.show()



def generate_training_samples(label_shp, tag, band_files,exception_band):
    '''
    :param label_shp: label polygons in shapefile format
    :param tag: the class of the polygon
    :param band_files: band files for this landsat image
    :param exception_band: some bands with different resolution
    :return: X,Y in DataFrame
    '''
    vector = fiona.open(label_shp, 'r')
    geom_list = []
    for feature in vector:
        # create a shapely geometry
        # this is done for the convenience for the .bounds property only
        # feature['geoemtry'] is in Json format
        geojson = feature['geometry']
        geom_list.append(geojson)

    X = np.array([])
    attributes =[]
    # read each band pixels for the same geometries
    for band_file in band_files:
        print(band_file)
        band = re.findall(r'.*?_(B.*?).TIF',band_file)[0]

        # for band6, its resolution is different with other bands
        band_ok = True
        for e_band in exception_band:
            if e_band in band:
                band_ok = False
        if not band_ok:
            continue

        attributes.append(band)
        raster = rasterio.open(band_file, 'r')
        mask = rasterio.features.rasterize(geom_list, out_shape=raster.shape, transform=raster.transform, fill=0,
                                           all_touched=False, default_value=tag, dtype=np.uint8)

        data = raster.read(1)
        print(data.shape)
        assert mask.shape == data.shape
        # each band is a column in X
        X = np.append(X, data[mask==tag])
        # with rasterio.open("/tmp/mask.tif" , 'w', driver='GTiff', width=raster.width,height=raster.height, crs=raster.crs, transform=raster.transform, dtype=np.uint16,nodata=256,count=raster.count, indexes=raster.indexes) as dst:
        #     # Write the src array into indexed bands of the dataset. If `indexes` is a list, the src must be a 3D array of matching shape. If an int, the src must be a 2D array.
        #     dst.write(mask.astype(rasterio.uint16), indexes =1)

    # organize the Training samples in X and Y
    band_num = len(attributes)
    # # X has the same number of columns as the number of bands
    X = X.reshape(band_num, -1).T
    # # Y has the same rows as X, which is X.shape[1], column is X.shape[0]
    Y = np.repeat(tag, X.shape[0])

    X = pd.DataFrame(data=X, columns=attributes)
    Y = pd.DataFrame(data=Y, columns=['tag'])
    print(X)
    print(Y)

    return X,Y


def get_training_data(file_root,sensor):
    # for some bands, they have different resolution, so the samples number is not the same as others
    exception_band = {'TM':['6'], 'ETM':['6','8'], 'OLI':['8','10','11']}

    label_dir = os.path.join(file_root, 'feature_shp')
    img_dir = os.path.join(file_root, 'image')
    tags = {'water': 1, 'mount_shadow': 2, 'cloud': 3, 'cloud_shadow': 4, 'snow': 5, 'other': 6}

    # 建立results文件夹
    result_path = os.path.join(file_root, 'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for rst in os.listdir(img_dir):
        print('get training data samples for %s image'%rst)
        rst_dir = os.path.join(img_dir, rst)
        band_files = [os.path.join(rst_dir, name) for name in os.listdir(rst_dir) if
                      name.endswith('.tif') or name.endswith('.TIF')]

        # to find the corresponding label shapefiles according to the names, eg cloud-L5134036.shp
        for file in os.listdir(label_dir):
            if file.endswith(rst+".shp"):
                print('the first label shapefile for this image is %s'%file)
                label_shp = os.path.join(label_dir,file)

                feat_name = re.findall(r'(.*?)-',file)[0]
                tag = tags[feat_name]

                print("generating the training data from the label shapefile")
                feat_x, feat_y = generate_training_samples(label_shp, tag, band_files, exception_band[sensor])
                result = pd.concat([feat_x, feat_y], axis=1)

                print('the attributes of the label from the images are %s'%feat_x.columns)

                result_name = os.path.join(result_path, '%s_%s.csv'%(rst,feat_name))
                result.to_csv(result_name)



# def get_training_data(file_root, if_stretch=True):
#     label_dir = os.path.join(file_root, 'feature_shp')
#     img_dir = os.path.join(file_root, 'image')
#
#     # tags = {'water': 1, 'mount_shadow': 2, 'cloud': 3, 'cloud_shadow': 4, 'snow': 5, 'other': 6}
#     tags = { 'cloud': 3}
#     for feat_name in tags.keys():
#         tag = tags[feat_name]
#         print("processing feature of %s" % feat_name)
#         for file in os.listdir(label_dir):
#             if file.startswith(feat_name) and file.endswith(".shp"):
#                 label_shp = os.path.join(label_dir, file)
#
#                 rst_start = re.findall(r'-(.*?).shp', file)[0]
#
#                 for rst in os.listdir(img_dir):
#                     # to find the corresponding landsat image
#                     if rst == rst_start:
#                         rst_dir = os.path.join(img_dir, rst)
#
#                         band_files = [os.path.join(rst_dir, name) for name in os.listdir(rst_dir) if
#                                       name.endswith('.tif') or name.endswith('.TIF')]
#
#                         X,attributes, Y = generate_training_samples(label_shp, tag, band_files)
#                         print(attributes)



def test_get_training_data():
    file_root = '/mnt/win/water_paper/training_data/TM'
    get_training_data(file_root,'TM')


if __name__ == '__main__':
    test_get_training_data()

    # remote_class = Remote_Classification('/mnt/win/water_paper/training_data/data')
    # remote_class.plot_results()
