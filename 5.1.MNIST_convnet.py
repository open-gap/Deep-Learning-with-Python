#%% 导入数据并初步处理
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = \
    mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#%% 建立简单的卷积神经网络模型
from keras import layers, models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', 
    input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) #将矩阵数据展平为一维向量
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary() #输出模型的概况

#%% 编译、训练并评估模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
    metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss , test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)