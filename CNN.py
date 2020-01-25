import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam


(train_x, train_y),(test_x, test_y) = mnist.load_data()


train_x = train_x.reshape(-1, 1, 28, 28)/255
test_x = test_x.reshape(-1, 1, 28, 28)/255
train_y = np_utils.to_categorical(train_y, num_classes=10)
test_y = np_utils.to_categorical(test_y, num_classes=10)

# 定义层
model = Sequential()
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28), # 意思是多少个数据(None), 1是通道数,对应下面的channels_first
    filters=32, #32个卷积核(滤波器), 深度为32
    kernel_size=5,
    strides=1,
    padding='same', # 标识卷积输入输出尺寸不变，'valid'表示输出的尺寸变小
    data_format='channels_first' # keras和tensorflow默认为channels_last也就是通道在最后，这里指定为在前面
)) # ->>>outpu(None, 32, 28, 28)
model.add(Activation('relu'))

model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
)) #->>>output(None, 32, 14, 14)

model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
# ->>>output(None, 64, 14, 14)
model.add(Activation('relu'))

model.add(MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
    data_format='channels_first'
)) # ->>>output(None, 64, 7, 7)

model.add(Flatten()) # ->>>转成(None, 3136)
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print('------训练-------')
model.fit(train_x, train_y, epochs=1, batch_size=64)

print('------测试-------')
loss, accuracy = model.evaluate(test_x, test_y)

print('Loss: ', loss)
print('accracy: ', accuracy)

