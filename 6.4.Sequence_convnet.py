#%% 启动Keras和TensorFlow
import keras
keras.__version__

#%% 准备IMDB数据
from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 10000  #考虑最常用的10000个单词
max_len = 500  #每条评论的截断长度
print('Loading data...')
(x_train, y_train), (x_test, y_test) = \
    imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

#%% 在IMDB数据上训练并评估一个简单的一维卷积神经网络
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary() #输出模型概况
# 编译并训练网络
model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy',
    metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10,  batch_size=128,
    validation_split=0.2)

#%% 绘制模型结果
import matplotlib.pyplot as plt
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

#%% 读取并处理耶拿数据
import os
import numpy as np
data_dir = 'D:/project/machine_learning_databases'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
with open(fname) as f:
    data = f.read()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step,
            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, _ in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
        
lookback = 1440 
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay,
    min_index=0, max_index=200000, shuffle=True, step=step, 
    batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay,
    min_index=200001, max_index=300000, step=step,
    batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay,
    min_index=300001, max_index=None, step=step,
    batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size

test_steps = (len(float_data) - 300001 - lookback) // batch_size

#%% 在耶拿天气数据上训练并评估一个简单的一维卷积神经网络
from keras import layers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
    input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
# 编译并训练模型
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500,
    epochs=20, validation_data=val_gen, validation_steps=val_steps)
# 绘制结果
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%% 为耶拿数据集准备更高分辨率的数据生成器
step = 3 #之前为6，现在表示每30分钟取一个数据点
lookback = 720  #同上
delay = 144 #同上
train_gen = generator(float_data, lookback=lookback, delay=delay,
    min_index=0, max_index=200000, shuffle=True, step=step)
val_gen = generator(float_data, lookback=lookback, delay=delay,
    min_index=200001, max_index=300000, step=step)
test_gen = generator(float_data, lookback=lookback, delay=delay,
    min_index=300001, max_index=None, step=step)
val_steps = (300000 - 200001 - lookback) // 128
test_steps = (len(float_data) - 300001 - lookback) // 128

#%% 结合一维卷积基和GRU层的模型
model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
    input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.summary()
# 编译并训练模型
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500,
    epochs=20, validation_data=val_gen, validation_steps=val_steps)

#%% 绘制训练结果
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()