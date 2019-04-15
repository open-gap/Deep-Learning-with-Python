#%% 获取并处理数据集
import numpy as np 
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)
print('train_data sample:', train_data[0])
print('train_labels sample:', train_labels[0])
print('max index:', max([max(sequence) for sequence in train_data]))

# 可将评论的词序列解码为英文单词
word_index = imdb.get_word_index() #获取单词到整数的映射字典
reverse_word_index = dict([
    (value, key) for (key, value) in word_index.items() #键值颠倒
])
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
) #索引减3是因为0、1、2是“padding”、“start of sequence”、“unknown”
print(decoded_review)

# 将整数序列编码为二进制矩阵
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

#%% 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#%% 向模型添加L2正则化
from keras import layers, models, regularizers, optimizers, losses, metrics
original_model = models.Sequential()
original_model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
original_model.add(layers.Dense(16, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))
original_model.compile(optimizer=optimizers.RMSprop(), 
    loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), 
    activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), 
    activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(), 
    loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
# 其他的权重正则化项
# regularizers.l1(0.001)
# regularizers.l1_l2(l1=0.001, l2=0.001)
original_history = original_model.fit(partial_x_train, partial_y_train, 
    epochs=20, batch_size=512, validation_data=(x_val, y_val))
history = model.fit(partial_x_train, partial_y_train, epochs=20, 
    batch_size=512, validation_data=(x_val, y_val))

#%% 绘图
import matplotlib.pyplot as plt
original_history_dict = original_history.history
original_val_loss_values = original_history_dict['val_loss']
history_dict = history.history
val_loss_values = history_dict['val_loss']
epochs = range(1, len(val_loss_values) + 1)
plt.plot(epochs, original_val_loss_values, 'bo', label='Original model')
plt.plot(epochs, val_loss_values, 'b', label='L2-regularized model')
plt.title('Original and L2-regularized model Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend(loc='upper left')
plt.show()

#%% 使用dropout
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(), 
    loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
history = model.fit(partial_x_train, partial_y_train, epochs=20, 
    batch_size=512, validation_data=(x_val, y_val))
history_dict = history.history
val_loss_values = history_dict['val_loss']
epochs = range(1, len(val_loss_values) + 1)
plt.plot(epochs, original_val_loss_values, 'bo', label='Original model')
plt.plot(epochs, val_loss_values, 'b', label='Dropout-regularized model')
plt.title('Original and Dropout-regularized model Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend(loc='upper left')
plt.show()