from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np

np.random.seed(333)
(train_x, train_y),(test_x, test_y) = mnist.load_data()
# 数据处理
train_x = train_x.reshape(train_x.shape[0], -1) / 255
test_x = test_x.reshape(test_x.shape[0], -1) / 255
# 原有是(60000, 28, 28)三维数组，变为(60000,28*28)这样的. reshape(60000,-1)代表变为60000数据，-1电脑会自动计算然后填充
train_y = np_utils.to_categorical(train_y, num_classes=10)
test_y = np_utils.to_categorical(test_y, num_classes=10)
# one-hot 原来的0~9的数字变成了(1*10)的数组，
# print(train_y[0:1, 0:10]) #具体可以这样看，输出1个 10列数据
'''
[5]
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]] ont-hot 数字为5 那么就在索引为几的位置
'''

# 定义层
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax') # 分类
])

# 搭建
rmsprop = RMSprop(lr=0.001, epsilon=1e-08, decay=0.0)

#
model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train
model.fit(train_x, train_y, epochs=2, batch_size=32)

# Test
loss, accuracy = model.evaluate(test_x, test_y)
print('loss: ', loss)
print('accuracy: ', accuracy)



