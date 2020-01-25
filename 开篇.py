import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

np.random.seed(333)
# keras兼容tensorflow和theano

x = np.linspace(-1, 1, 200)
np.random.shuffle(x) #打乱
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (200,))

train_x, train_y = x[:160], y[:160]
test_x, test_y = x[160:], y[160:]


# 首先定义层
model = Sequential() # 序贯模型(顺序模型)
model.add(Dense(output_dim=1, input_dim=1)) # Dense(output_dim, input_dim)

# 搭建网络
model.compile(loss='mse', optimizer='sgd')

# train

plt.ion()
plt.show()
print(">>>>>>>>Train<<<<<<<<")

for step in range(301):
    cost = model.train_on_batch(train_x, train_y)
    if step % 50 == 0:
        print("误差: ", cost)
        pred = model.predict(test_x)
        plt.cla()
        plt.scatter(test_x, test_y)
        plt.plot(test_x, pred)
        plt.pause(0.5)


plt.ioff()
plt.show()

# test
# print(">>>>>>Test<<<<<<")
# #此次全部传过去
# cost = model.evaluate(test_x, test_y, batch_size=40) # 评价
# print("test cost", cost)
# w, b = model.layers[0].get_weights()
# print("W:",w)
# print("b", b)


# predict_y = model.predict(test_x)
# plt.scatter(test_x, test_y)
# plt.plot(test_x, predict_y)
# plt.show()
