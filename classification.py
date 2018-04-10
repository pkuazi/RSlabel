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



def read_labeled_pixels(label_shp, tag, band_files, exception_band):
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
    attributes = []
    # read each band pixels for the same geometries
    for band_file in band_files:
        print(band_file)
        band = re.findall(r'.*?_(B.*?).TIF', band_file)[0]

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
        X = np.append(X, data[mask == tag])
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

    return X, Y


def generate_training_data(file_root, sensor):
    '''
    读取矢量标记的polygon数据，读取对应的栅格像素，注意矢量标记的命名规则：feat_name +‘-‘+影像L5+path+row
    :param file_root: folder of the label shapefiles and corresponding images
    :param sensor: landsat sensor: TM, ETM, OLI
    :return:save training pixels and labels into dataframe for each feature_image pair
    '''
    # for some bands, they have different resolution, so the samples number is not the same as others
    exception_band = {'TM': ['6'], 'ETM': ['6', '8'], 'OLI': ['8', '10', '11']}

    label_dir = os.path.join(file_root, 'feature_shp')
    img_dir = os.path.join(file_root, 'image')
    tags = {'water': 1, 'mount_shadow': 2, 'cloud': 3, 'cloud_shadow': 4, 'snow': 5, 'other': 6}

    # 建立存放训练数据csv文件的文件夹
    result_path = os.path.join(file_root, 'traing_csv')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for rst in os.listdir(img_dir):
        print('get training data samples for %s image' % rst)
        rst_dir = os.path.join(img_dir, rst)
        band_files = [os.path.join(rst_dir, name) for name in os.listdir(rst_dir) if
                      name.endswith('.tif') or name.endswith('.TIF')]

        # to find the corresponding label shapefiles according to the names, eg cloud-L5134036.shp
        for file in os.listdir(label_dir):
            if file.endswith(rst + ".shp"):
                print('the first label shapefile for this image is %s' % file)
                label_shp = os.path.join(label_dir, file)

                feat_name = re.findall(r'(.*?)-', file)[0]
                tag = tags[feat_name]

                print("generating the training data from the label shapefile")
                feat_x, feat_y = read_labeled_pixels(label_shp, tag, band_files, exception_band[sensor])
                # print(feat_name, feat_x.shape)
                result = pd.concat([feat_x, feat_y], axis=1)

                print('the attributes of the label from the images are %s' % feat_x.columns)

                result_name = os.path.join(result_path, '%s_%s.csv' % (rst, feat_name))
                result.to_csv(result_name)


class Remote_Classification:
    def __init__(self, file_root, image_bands_dir, sensor):
        self.file_root = file_root
        self.training_dir = os.path.join(file_root, 'traing_csv')
        self.image_bands_dir = image_bands_dir
        self.sensor = sensor

    def gen_features(self, original_X):
        '''
        using the six bands to generate other features, such as all kinds of water indeices
        :return: new traing X and Y
        '''
        band1 = original_X['B10']  # blue
        band2 = original_X['B20']  # green
        band3 = original_X['B30']  # red
        band4 = original_X['B40']  # nir
        band5 = original_X['B50']  # swir5
        band7 = original_X['B70']  # swir7
        NDWI = (band4 - band5) / (band4 + band5)
        MNDWI = (band2 - band5) / (band2 + band5)
        EWI = (band2 - band4 - band5) / (band2 + band4 + band5)
        NEW = (band1 - band7) / (band1 + band7)
        NDWI_B = (band1 - band4) / (band1 + band4)
        AWElnsh = 4 * (band2 - band5) - (0.25 * band4 + 2.75 * band7)

        features = pd.concat([NDWI, MNDWI, EWI, NEW, NDWI_B, AWElnsh], axis=1)
        features.columns = ['NDWI', 'MNDWI', 'EWI', 'NEW', 'NDWI_B', 'AWElnsh']
        # DataFrame({}data=[NDWI, MNDWI, EWI, NEW, NDWI_B, AWElnsh], columns=['NDWI', 'MNDWI', 'EWI', 'NEW', 'NDWI_B', 'AWElnsh'])
        # training_X = pd.merge(original_X, features, left_index=True, right_index=True)
        training_X = pd.concat([original_X, features], axis=1)
        return training_X

    def get_training_data(self):
        '''
        read the labeled pixels, and corresponding bands values
        :return: original training X, that is six bands values
        '''
        # training data, the features are the original six band values.
        for file in os.listdir(self.training_dir):
            if file.endswith('water.csv'):
                X_water = pd.read_csv(os.path.join(self.training_dir, file),
                                      usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
            elif file.endswith('cloud.csv'):
                X_cloud = pd.read_csv(os.path.join(self.training_dir, file),
                                      usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
            elif file.endswith('cloud_shadow.csv'):
                X_cloud_shadow = pd.read_csv(os.path.join(self.training_dir, file),
                                             usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
            elif file.endswith('mount_shadow.csv'):
                X_mount_shadow = pd.read_csv(os.path.join(self.training_dir, file),
                                             usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
            elif file.endswith('snow.csv'):
                X_snow = pd.read_csv(os.path.join(self.training_dir, file),
                                     usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
            elif file.endswith('other.csv'):
                X_other = pd.read_csv(os.path.join(self.training_dir, file),
                                      usecols=['B10', 'B20', 'B30', 'B40', 'B50', 'B70'])
        X_nonwater = pd.concat([X_cloud, X_cloud_shadow, X_mount_shadow, X_snow, X_other])

        Y_water = pd.Series(np.repeat(1, X_water.shape[0]))
        Y_nonwater = pd.Series(np.repeat(0, X_nonwater.shape[0]))

        original_X = pd.concat([X_water, X_nonwater])
        training_Y = pd.concat([Y_water, Y_nonwater])

        training_X = self.gen_features(original_X)
        return training_X, training_Y

    def read_bands_to_be_classified(self):
        # for some bands, they have different resolution, so the samples number is not the same as others
        exception_band = {'TM': ['6'], 'ETM': ['6', '8'], 'OLI': ['8', '10', '11']}

        attributes = []
        values = []
        band_files = [os.path.join(self.image_bands_dir, name) for name in os.listdir(self.image_bands_dir) if
                      name.endswith('.tif') or name.endswith('.TIF')]
        for band_file in band_files:
            print(band_file)
            band = re.findall(r'.*?_(B.*?).TIF', band_file)[0]

            # for band6, its resolution is different with other bands
            band_ok = True
            for e_band in exception_band[self.sensor]:
                if e_band in band:
                    band_ok = False
            if not band_ok:
                continue

            attributes.append(band)

            raster = rasterio.open(band_file, 'r')
            # reading test data using window ((row_start, row_stop), (col_start, col_stop))
            # array = raster.read(1,window=((1000,1010),(3000,3006)))
            array = raster.read(1)
            row = array.shape[0]
            col = array.shape[1]
            values.append(array.reshape(-1, 1))
        bands_num = len(attributes)
        value_array = np.array(values)
        resize_array = value_array.reshape(bands_num,-1)

        X = pd.DataFrame(resize_array.T, columns=attributes)
        print(X.shape)

        return X, row, col

    def SVM(self):
        model_npy = os.path.join(self.file_root, '/models/Z_svm.npy')
        if os.path.exists(model_npy):
            Z_SVM = np.load(model_npy)
        else:
            training_X, training_y = self.get_training_data()
            # SVM
            print('begin training SVM model......')
            clf = svm.SVC()
            clf.fit(training_X, training_y)

            print('reading image data %s to be processed....'% self.image_bands_dir.split('/')[-1])
            X, row, col = self.read_bands_to_be_classified()
            return X

            print('using SVM model to predict the image.....')
            Z_SVM = clf.predict(X).reshape(row, col)
            np.save(model_npy, Z_SVM)
        return Z_SVM

    def RandomForest(self):
        model_npy = os.path.join(self.file_root + '/models/Z_random_forest.npy')
        if os.path.exists(model_npy):
            Z_Random_Forest = np.load(model_npy)
        else:
            training_X, training_y = self.get_training_data()
            # Random Forest
            clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
            clf.fit(training_X, training_y)
            Z_Random_Forest = clf.predict(self.X).reshape(512, 512)
            np.save(model_npy, Z_Random_Forest)
        return Z_Random_Forest

    def KNN(self, n_neighbors):
        model_npy = os.path.join(self.file_root, '/models/Z_knn%s.npy' % n_neighbors)
        if os.path.exists(model_npy):
            Z_KNN = np.load(model_npy)
        else:
            training_X, training_y = self.get_training_data()
            # Random Forest
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(training_X, training_y)
            Z_KNN = clf.predict(self.X).reshape(512, 512)
            np.save(model_npy, Z_KNN)
        return Z_KNN

    def Kmeans(self):
        model_npy = os.path.join(self.file_root, '/models/Z_kmeans.npy')
        if os.path.exists(model_npy):
            Z_KMeans = np.load(model_npy)
        else:
            training_X, training_y = self.get_training_data()
            # K-means
            kmeans = KMeans(n_clusters=4, random_state=0).fit(training_X)
            Z_KMeans = kmeans.predict(self.X).reshape(512, 512)
            np.save(model_npy, Z_KMeans)
        return Z_KMeans

    def plot_results(self):
        plt.figure(figsize=(10, 7))

        ax = plt.subplot(231)
        ax.set_title("Original")
        plt.imshow(self.img)

        ax = plt.subplot(232)
        ax.set_title("Reference Map")
        plt.imshow(self.label)

        ax = plt.subplot(233)
        ax.set_title("SVM Result")
        Z_svm = self.SVM()
        plt.imshow(Z_svm)

        ax = plt.subplot(234)
        ax.set_title("Random Forest Result")
        Z_random_forest = self.RandomForest()
        plt.imshow(Z_random_forest)

        ax = plt.subplot(235)
        ax.set_title("KNN Result (k=10)")
        Z_knn = self.KNN(10)
        plt.imshow(Z_knn)

        ax = plt.subplot(236)
        ax.set_title("KNN Result (k=50)")
        Z_knn = self.KNN(50)
        plt.imshow(Z_knn)
        plt.show()

    def stretch_plot(self):
        ax = plt.subplot(221)
        img = self.img
        plt.imshow(img)
        ax = plt.subplot(222)
        plt.hist(img.reshape(512 * 512, 3), color=['red', 'green', 'blue'])
        ax = plt.subplot(223)
        img_ = self.stretch(img)
        plt.imshow(img_)
        ax = plt.subplot(224)
        plt.hist(img_.reshape(512 * 512, 3), color=['red', 'green', 'blue'])
        plt.show()


def test_generate_training_data():
    file_root = '/mnt/win/water_paper/training_data/TM'
    generate_training_data(file_root, 'TM')


def test_classification():
    file_root = '/mnt/win/water_paper/training_data/TM'
    images_to_be_classified = os.path.join(file_root, 'image/L5134036')
    remote_class = Remote_Classification(file_root, images_to_be_classified, 'TM')
    remote_class.SVM()

    # X,Y = remote_class.get_training_data()
    # print(X.shape)
    # print(Y.shape)


if __name__ == '__main__':
    # test_generate_training_data()
    test_classification()
