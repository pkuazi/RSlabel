import keras
from keras import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Conv2D, Dropout
from keras.optimizers import SGD
from keras.datasets import mnist

# 输入是7*7的小方块，同一块判为一个类。
from classification import Remote_Classification


def CNN(x_train, y_train, x_test, y_test):
    batch_size = 1
    num_classes = 4
    epochs = 12
    img_rows, img_cols = 512, 512
    input_shape = (img_rows, img_cols, 3)

    # Model begins
    model = Sequential()
    # 32个filter, 尺寸3*3
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # 64个filter, 尺寸3*3
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # 池化，尺寸2*2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # 模型我们使用交叉熵损失函数，最优化方法选用Adadelta
    model.compile(loss=keras.metrics.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))


def split77(img):
    pass

if __name__ == '__main__':
    remote_class = Remote_Classification('./cache/gen_imgs_xilidu/random_gen_028_json')
    img = remote_class.img
    label = remote_class.label
    print(img), print(img.shape)
    print(label), print(label.shape)
