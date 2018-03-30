# -*- coding:utf-8 -*-
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


# 随机裁剪512*512的图像
def random_split_img(img, size):
    max_x, max_y, _ = img.shape
    random_x = np.random.randint(max_x)
    random_y = np.random.randint(max_y)  # 随机生成一个点
    while random_x + size[0] >= max_x or random_y + size[1] >= max_y:
        random_x = np.random.randint(max_x)
        random_y = np.random.randint(max_y)  # 超界则重新随机生成
    return img[random_x:random_x + size[0], random_y:random_y + size[1], :]


# 随机生成size大小的RGB图像
def random_gen_imgs(num=100, size=(512, 512)):
    file_idx = [4, 5, 6]  # 组合三个波段的图像为一张图像
    tif_files = []
    for idx in file_idx:
        tif_files.append('./data/culidu/LC81240322017118LGN00_B%d.TIF' % idx)
    img = []
    for i, tif_file in enumerate(tif_files, 0):
        print(tif_file)
        img.append(tiff.imread(tif_file))
    img = np.array(img)
    print(img.shape)
    img = img.transpose([1, 2, 0])
    img = img * 1.0 / 65535 * 256
    img.astype(np.int)
    # plt.imshow(img[:, :, :])
    # plt.show()
    print(img.shape)
    for i in range(num):
        sub_img = random_split_img(img, size)
        if not os.path.exists("./cache/gen_imgs_culidu/"): os.makedirs("./cache/gen_imgs_culidu/")
        cv2.imwrite("./cache/gen_imgs_culidu/random_gen_%.3d.png" % (i), sub_img)


# 随机生成size大小的RGB图像
def random_gen_imgs1(num=10, size=(512, 512)):
    tif_file = './data/xilidu/2006_Level17.tif'
    img = tiff.imread(tif_file)
    print(img)
    img = np.array(img)
    print(img.shape)
    # img = img.transpose([1, 2, 0])
    # img = img * 1.0 / 65535 * 256
    img.astype(np.int)
    print(img.shape)
    for i in range(num):
        sub_img = random_split_img(img, size)
        if not os.path.exists("./cache/gen_imgs_xilidu/"): os.makedirs("./cache/gen_imgs_xilidu/")
        cv2.imwrite("./cache/gen_imgs_xilidu/random_gen_%.3d.png" % (i), sub_img)


if __name__ == '__main__':
    random_gen_imgs1()
