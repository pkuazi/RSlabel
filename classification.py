import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


class Remote_Classification:
    def __init__(self, file_root, if_stretch=True):
        self.file_root = file_root
        self.img = cv2.imread(self.file_root + '/img.png')
        self.label = cv2.imread(self.file_root + '/label.png', cv2.IMREAD_UNCHANGED)
        self.X = self.img.reshape(-1, 3)
        self.Y = self.label.reshape(1, -1)[0]

        if if_stretch:
            img_lashen = self.stretch(self.img)
            self.img = img_lashen
            self.X = img_lashen.reshape(-1, 3)

        # 建立results文件夹
        if not os.path.exists(self.file_root + '/results'):
            os.makedirs(self.file_root + '/results')

    def get_training_data(self):
        # training data, the features are the original RGB three values.
        farm = np.argwhere(self.Y == 1).reshape(1, -1)[0]
        river = np.argwhere(self.Y == 2).reshape(1, -1)[0]
        road = np.argwhere(self.Y == 3).reshape(1, -1)[0]
        roof = np.argwhere(self.Y == 4).reshape(1, -1)[0]

        farm_X = self.X[farm]
        river_X = self.X[river]
        road_X = self.X[road]
        roof_X = self.X[roof]

        farm_y = self.Y[farm]
        river_y = self.Y[river]
        road_y = self.Y[road]
        roof_y = self.Y[roof]

        training_X = np.concatenate([farm_X, river_X, road_X, roof_X])
        training_y = np.concatenate([farm_y, river_y, road_y, roof_y])

        return training_X, training_y

    def SVM(self):
        if os.path.exists(self.file_root + '/results/Z_svm.npy'):
            Z_SVM = np.load(self.file_root + '/results/Z_svm.npy')
        else:
            training_X, training_y = self.get_training_data()
            # SVM
            clf = svm.SVC()
            clf.fit(training_X, training_y)
            Z_SVM = clf.predict(self.X).reshape(512, 512)
            np.save(self.file_root + '/results/Z_svm.npy', Z_SVM)
        return Z_SVM

    def RandomForest(self):
        if os.path.exists(self.file_root + '/results/Z_random_forest.npy'):
            Z_Random_Forest = np.load(self.file_root + '/results/Z_random_forest.npy')
        else:
            training_X, training_y = self.get_training_data()
            # Random Forest
            clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
            clf.fit(training_X, training_y)
            Z_Random_Forest = clf.predict(self.X).reshape(512, 512)
            np.save(self.file_root + '/results/Z_random_forest.npy', Z_Random_Forest)
        return Z_Random_Forest

    def KNN(self, n_neighbors):
        if os.path.exists(self.file_root + '/results/Z_knn%s.npy' % n_neighbors):
            Z_KNN = np.load(self.file_root + '/results/Z_knn%s.npy' % n_neighbors)
        else:
            training_X, training_y = self.get_training_data()
            # Random Forest
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(training_X, training_y)
            Z_KNN = clf.predict(self.X).reshape(512, 512)
            np.save(self.file_root + '/results/Z_KNN_%s.npy' % n_neighbors, Z_KNN)
        return Z_KNN

    def Kmeans(self):
        if os.path.exists(self.file_root + '/results/Z_kmeans.npy'):
            Z_KMeans = np.load(self.file_root + '/results/Z_kmeans.npy')
        else:
            training_X, training_y = self.get_training_data()
            # K-means
            kmeans = KMeans(n_clusters=4, random_state=0).fit(training_X)
            Z_KMeans = kmeans.predict(self.X).reshape(512, 512)
            np.save(self.file_root + '/results/Z_kmeans.npy', Z_KMeans)
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

    def stretch(self, bands, lower_percent=2, higher_percent=98):
        out = np.zeros_like(bands, dtype=np.float32)
        n = bands.shape[2]
        print(n)
        for i in range(n):
            a = 0
            b = 1
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :, i] = t
        return out.astype(np.float32)

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


if __name__ == '__main__':
    remote_class = Remote_Classification('./cache/gen_imgs_xilidu/random_gen_076_json')
    remote_class.plot_results()
