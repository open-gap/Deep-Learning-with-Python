#%% 加载MNIST数据集
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = \
    mnist.load_data()
print('train_images.shape:', train_images.shape)
print('train_labels.shape:', len(train_labels))
print('test_images.shape:', test_images.shape)
print('test_labels.shape:', len(test_labels))

#%% 网络架构
from keras import models, layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

#%% 编译步骤
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
    metrics=['accuracy'])

#%% 准备图像数据
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

#%% 准备标签
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#%% 拟合模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#%% 检测模型性能
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

#%% 显示数据集中的图片
import matplotlib.pyplot as plt 
digit = train_images[4].reshape((28, 28))
plt.imshow(digit, cmap='gray')
plt.show()

#%% 使用函数式API定义相同模型
from keras import optimizers
input_tensor = layers.Input(shape=(28*28, ))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', 
    metrics=['accuracy'])
# model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)
model.fit(train_images, train_labels, batch_size=128, epochs=10)