#%% 加载IMDB数据集
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)
print('train_data sample:', train_data[0])
print('train_labels sample:', train_labels[0])
print('max index:', max([max(sequence) for sequence in train_data]))

#%% 可将评论的词序列解码为英文单词
word_index = imdb.get_word_index() #获取单词到整数的映射字典
reverse_word_index = dict([
    (value, key) for (key, value) in word_index.items() #键值颠倒
])
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
) #索引减3是因为0、1、2是“padding”、“start of sequence”、“unknown”
print(decoded_review)

#%% 将整数序列编码为二进制矩阵
import numpy as np 

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension)) #创建(25000, 10000)的零矩阵
    for i, sequence in enumerate(sequences): #sequence是(1, N)的数组
        results[i, sequence] = 1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print('x_train sample:', x_train[0])
print('x_train shape:', x_train.shape)
print('y_train sample:', y_train[0])

#%% 模型定义
from keras import models, layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#%% 模型的编译、配置优化器
# 
# 注：对于输出概率值模型，交叉熵(crossentropy)往往是最好的选择
# 二分类问题输出一个概率值使用二元交叉熵(binary_crossentropy)，其余的还有
# 均方误差(mean_squared_error, mse)
# 
# 注：激活函数除relu外还有tanh、prelu、elu等
# 
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
#     metrics=['accuracy'])
#
### 自定义配置优化器
# from keras import optimizers
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), 
#     loss='binary_crossentropy', metrics=['accuracy'])
# 
### 使用自定义的损失和指标
from keras import optimizers, losses, metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001), 
    loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

#%% 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#%% 训练模型
history = model.fit(partial_x_train, partial_y_train, epochs=20, 
    batch_size=512, validation_data=(x_val, y_val))
history_dict = history.history
print(history_dict.keys())

#%% 绘制训练损失和验证损失
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()

#%% 绘制训练精度和验证精度
plt.clf() #清空图像
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

#%% 从头训练一个新的网络
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
    metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print('results:', results) #输出测试集结果
print('model prediction:', model.predict(x_test)) #输出每个测试集样本预测概率