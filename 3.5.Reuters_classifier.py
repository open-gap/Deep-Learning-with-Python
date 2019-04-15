#%% 加载路透社数据集
from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = \
    reuters.load_data(num_words=10000)
print('train_data len:', len(train_data))
print('test_data len:', len(test_data))
print('train_data sample:', train_data[10])

#%% 将索引解码为新闻文本
word_index = reuters.get_word_index()
reverse_word_index = dict([
    (value, key) for (key, value) in word_index.items()
])
decoded_newswire = ' '.join([
    reverse_word_index.get(i - 3, '?') for i in train_data[0]
]) #同IMDB_calssifier
print('train_data[0]:', decoded_newswire)
print('train_labels[0]:', train_labels[0])

#%% 编码数据
import numpy as np 
from keras.utils.np_utils import to_categorical
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 使用one-hot(独热)编码
# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.0
#     return results
# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)
# 使用Keras内置方法
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

#%% 模型定义与编译
from keras import models, layers
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
# 注：对于独热标签使用categorical_crossentropy，对于整数标签应使用
# sparse_categorical_crossentropy，与前者仅仅接口不同
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
    metrics=['accuracy'])

#%% 留下验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#%% 开始训练网络
history = model.fit(partial_x_train, partial_y_train, epochs=20, 
    batch_size=512, validation_data=(x_val, y_val))

#%% 绘制训练损失和验证损失
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()

#%% 绘制训练精度和验证精度
plt.clf() #清空图像
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

#%% 从头开始训练一个新的模型
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
    metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, 
    validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print('results:', results)

#%% 随机分类器结果
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print(float(np.sum(hits_array)) / len(test_labels))

#%% 在新数据集上生成预测结果
predictions = model.predict(x_test)
print('predictions.shape:', predictions.shape)
print('probability sum:', np.sum(predictions[0]))
print('max probability category:', np.argmax(predictions[0]))
