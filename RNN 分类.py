import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Activation
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import np_utils
np.random.seed(1337)

TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001


(train_x, train_y),(test_x, test_y) = mnist.load_data()


train_x = train_x.reshape(-1, 28, 28)/255
test_x = test_x.reshape(-1, 28, 28)/255
train_y = np_utils.to_categorical(train_y, num_classes=10)
test_y = np_utils.to_categorical(test_y, num_classes=10)


# 定义层
model = Sequential()

model.add(SimpleRNN(
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),# (50, 28, 28)
    # 这里不要设置样本的数量， 不然下面的evaluate会报错
    output_dim = CELL_SIZE,
))

model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

adam = Adam(lr=LR)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

for step in range(2001):
    x_batch = train_x[BATCH_INDEX: BATCH_SIZE+BATCH_INDEX, :, :]# 每批截50个数据
    y_batch = train_y[BATCH_INDEX: BATCH_SIZE+BATCH_INDEX, :]
    cost = model.train_on_batch(x_batch, y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= train_x.shape[0] else BATCH_INDEX

    if step % 50 == 0:
        loss, accracy = model.evaluate(test_x, test_y, batch_size=test_x.shape[0], verbose=False)
        print('loss: ', loss)
        print('accracy: ', accracy)
