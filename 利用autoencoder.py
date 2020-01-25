from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt


model = load_model('autoencoder.h5')

(train_x, _), (test_x, test_y) = mnist.load_data()


x_train = train_x.astype('float32') / 255. - 0.5
x_test = test_x.astype('float32') / 255. - 0.5
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))



autocoded_imgs = model.predict(x_test)


plt.subplot(1, 2, 1)
plt.imshow(x_test[0].reshape(28, 28))
# plt.gray()
plt.subplot(1, 2, 2)
plt.imshow(autocoded_imgs[0].reshape(28, 28))
# plt.gray()
plt.show()


