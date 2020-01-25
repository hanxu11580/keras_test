import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Dense, Input
from PIL import Image


# 非监督学习， 没有标签的也就是train_y
(train_x, _), (test_x, test_y) = mnist.load_data()


x_train = train_x.astype('float32') / 255. - 0.5
x_test = test_x.astype('float32') / 255. - 0.5
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# print(x_test.shape) # (10000, 784)
# print(x_test[0].shape)
# exit()

encoding_dim = 2 # 最后将数据压缩为2个

input_img = Input(shape=(784, ))

# encoder层
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder层
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='relu')(decoded)

autoencoder = Model(
    input=input_img,
    output=decoded
)

# 单独encoder层
encoder = Model(input=input_img, output=encoder_output)

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

autoencoder.fit(x_train,
                x_train,
                epochs=30,
                batch_size=256,
                shuffle=True)


autoencoder.save('autoencoder.h5')
# autocoded_imgs = autoencoder.predict(x_test)




# plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=test_y)
# plt.show()





