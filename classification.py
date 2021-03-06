import rasterio
import os, re

import numpy as np
from affine import Affine
import rasterio.features
import fiona
import pandas as pd

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from imagepixel2trainingdata import Imagepixel_Trainingdata


class Remote_Classification:
    def __init__(self, file_root, image_bands_dir, sensor):
        self.file_root = file_root
        self.training_dir = os.path.join(file_root, 'traing_csv')
        self.image_bands_dir = image_bands_dir
        self.sensor = sensor

        imagepixel_trainingdata = Imagepixel_Trainingdata(self.file_root, self.image_bands_dir, self.sensor)
        training_X, training_y = imagepixel_trainingdata.get_training_data()
        self.train_x, self.train_y, self.test_x, self.test_y = imagepixel_trainingdata.split_dataset(training_X,
                                                                                                     training_y, 0.1)
        print('reading image data %s to be processed....' % imagepixel_trainingdata.image_bands_dir.split('/')[-1])
        X, self.src_profile = imagepixel_trainingdata.read_bands_to_be_classified()
        self.X = imagepixel_trainingdata.gen_features(X)

    def SVM(self):
        model_pkl = os.path.join(self.file_root, '/models/svm.pkl')
        if os.path.exists(model_pkl):
            clf = np.load(model_pkl)
        else:
            # SVM
            print('begin training SVM model......')
            clf = svm.SVC()
            clf.fit(self.train_x, self.train_y)
            # save the model by using pickle

        print('using SVM model to predict the image.....')
        row = self.src_profile['height']
        col = self.src_profile['width']

        # 由于影像边缘存在NoData，因此需对边缘NoData位置或者index作标记，预测的结果也直接写入NoData,所以需要提前建好预测结果，并将NoData提案其中，剩下的位置则由预测模型的结果填入。
        # 不能直接使用self.X
        img_SVM = clf.predict(self.X).reshape(row, col)

        dst_path = os.path.join(self.file_root, '/results/%s.tif' % self.image_bands_dir.split('/')[-1])
        # save the results into a raster array
        with rasterio.Env():
            # Write an array as a raster band to a new 8-bit file. For
            # the new file's profile, we start with the profile of the source
            profile = self.src_profile
            # And then change the band count to 1, set the dtype to uint8, and specify LZW compression.
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw')

            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(img_SVM.astype(rasterio.uint8), 1)

                # save the model by using pickle
                # joblib.dump(clf, model_pkl)

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


def test_classification():
    file_root = '/mnt/win/water_paper/training_data/TM'
    images_to_be_classified = os.path.join(file_root, 'image/L5134036')
    remote_class = Remote_Classification(file_root, images_to_be_classified, 'TM')
    remote_class.SVM()


if __name__ == '__main__':
    # test_generate_training_data()
    test_classification()
