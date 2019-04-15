#%% 启动Keras和TensorFlow
import keras
keras.__version__

#%% 简单RNN的Numpy实现
import numpy as np 
timesteps = 100 #输入序列的时间步数
input_features = 32 #输入特征空间的维度
output_features = 64 #输出特征空间的维数
inputs = np.random.random((timesteps, input_features)) #输入数据
state_t = np.zeros((output_features, )) #初始状态：全零向量
# 创建随机初始的权重矩阵
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features, ))
successice_outputs = [] #输出结果保存的列表
for intput_t in inputs:
    # 计算模块输出
    output_t = np.tanh(np.dot(W, intput_t) + np.dot(U, state_t) + b)
    successice_outputs.append(output_t) #保存输出结果
    state_t = output_t #更新网络状态用于下一时间步
# 将最终输出展开为二维张量
final_output_sequence = np.stack(successice_outputs, axis=0)

#%% 上述Numpy实现对应于实际的Keras层，即SimpleRNN层
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
# 下面例子表示模型只返回最后一个输出
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

#%% 下面例子表示模型返回完整的状态序列
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()

#%% 下面例子表示模型将多个RNN层逐个叠加，中间层返回完整输出序列
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  #默认情况SimpleRNN只返回最后一个输出
model.summary()

#%% 准备IMDB数据集
from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 10000  #考虑的常用单词数量
maxlen = 500  #只考虑每条评论前maxlen个单词，后面的裁切掉
print('Loading data...')
(input_train, y_train), (input_test, y_test) = \
    imdb.load_data(num_words=max_features)
print('Find', len(input_train), 'train sequences')
print('Find', len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

#%% 使用Embedding层和SimpleRNN层来训练模型并绘制结果
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
# 编译并训练模型
model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
    metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128,
    validation_split=0.2)
# 绘制模型结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%% 使用Keras中的LSTM层训练模型并绘制结果
from keras.layers import LSTM
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
# 编译训练模型
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
    metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128,
    validation_split=0.2)
# 绘制结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()