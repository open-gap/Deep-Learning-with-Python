#%% 启动Keras和TensorFlow
import keras
keras.__version__

#%% 读取耶拿天气数据集的数据
import os
data_dir = 'D:/project/machine_learning_databases'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
with open(fname, encoding='utf-8') as f:
    data = f.read()
lines = data.split('\n')
header = lines[0].split(',') #读取的CSV文件的首行说明行
lines = lines[1: ] #读取的CSV文件的剩余行
print('Datasets header:', header)
print('Datasets lines:', len(lines))

#%% 解析读取的天气数据并绘制温度数据
import numpy as np
from matplotlib import pyplot as plt
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = np.asarray(values)
temp = float_data[:, 1]  #第2列数据-温度数据(单位：℃)
plt.figure()
plt.plot(range(len(temp)), temp)
plt.show()
plt.figure()
# 由于每10分钟记录一次数据，一天有144个数据点
plt.plot(range(1440), temp[:1440]) #绘制前十天的温度变化情况
plt.show()

#%% 准备模型需要的数据
# 数据标准化，注：只考虑前200000行数据
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std
# 生成时间序列样本及其目标的生成器，函数返回Python生成器
# data指输入的原始浮点数数组
# lookback指输入数据包含过去多少个时间步
# delay指目标在未来多少个时间步之后
# min_index和max_index界定需要抽取哪些时间步
# shuffle指是否打乱样本
# batch_size指每个批量的样本数
# step指数据采样的周期(单位：时间步)，取6指每小时取一个点
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
        targets = np.zeros((len(rows), ))
        for j, row in enumerate(rows):
            indices = range(row - lookback, row, step)
            samples[j] = data[indices]
            targets[j] = data[row + delay][1]
        yield samples, targets #生成器返回包含samples和targets的元组
# 准备训练生成器、验证生成器和测试生成器
lookback = 1440 #给定过去10天的观测数据
step = 6 #观测数据采样频率是每小时一次
delay = 144 #目标为24小时后的数据
batch_size = 128 #批量大小
train_gen = generator( #训练数据生成器
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000, #数据集中前200000个数据用于训练
    shuffle=True,
    step=step, 
    batch_size=batch_size
)
val_gen = generator( #验证数据生成器
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=200001, #数据集中200000到3000000个数据用于验证
    max_index=300000,
    step=step,
    batch_size=batch_size
)
test_gen = generator( #测试数据生成器
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=300001, #数据集中剩余数据用于测试
    max_index=None,
    step=step,
    batch_size=batch_size
)
# 为了查看整个验证数据集需要从val_gen中抽取的次数
val_steps = (300000 - 200001 - lookback) // batch_size
# 为了查看整个测试数据集需要从test_gen中抽取的次数
test_steps = (len(float_data) - 300001 - lookback) // batch_size

#%% 一种基于常识的、非机器学习的基准方法
# 即使用始终预测24小时后温度等于现在温度
# 计算符合常识的基准方法的MAE(平均绝对误差)
def evaluate_naive_method():
    batch_maes = []
    for _ in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print('MAE:', np.mean(batch_maes))
    return np.mean(batch_maes)
celsius_mae = evaluate_naive_method() * std[1]
print('Temperature Error:', celsius_mae)

#%% 训练并评估一个密集连接模型
from keras import layers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import RMSprop
model = Sequential()
# 先展平输入数据序列
model.add(layers.Flatten(input_shape=(
    lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu')) #然后使用全连接层
model.add(layers.Dense(1)) #回归问题输出层不使用激活函数
# 编译并运行模型
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

#%% 训练并评估一个基于GRU的模型
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
# 编译并运行模型
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

#%% 训练并评估一个使用dropout正则化的基于GRU的模型
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
model = Sequential()
# GRU模块中dropout表示该层输入单元的dropout比率，
# recurrent_dropout指定循环单元的dropout比率
model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2,
    input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
# 编译并运行模型
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500,
    epochs=40, validation_data=val_gen, validation_steps=val_steps)
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

#%% 训练并评估一个使用dropout正则化的堆叠GRU模型
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5,
    return_sequences=True, input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu', dropout=0.1, 
    recurrent_dropout=0.5))
model.add(layers.Dense(1))
# 编译并运行模型
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500,
    epochs=40, validation_data=val_gen, validation_steps=val_steps)
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

#%% 使用逆序序列训练并评估一个LSTM
from keras import layers
from keras.datasets import imdb
from keras.models import Sequential
from keras.preprocessing import sequence
max_features = 10000 #最为特征的单词个数
maxlen = 500 #仅考虑每条评论的前500个单词，之后的单词截断
(x_train, y_train), (x_test, y_test) = \
    imdb.load_data(num_words=max_features)
# 反序读取的IMDB评论数据
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]
# 填充数据为二维张量
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# 构建模型
model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))
# 编译并运行模型
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
    metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10,
    batch_size=128, validation_split=0.2)

#%% 训练并评估一个双向LSTM
from keras import backend as K
K.clear_session()
# 新建模型
model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation='sigmoid'))
# 编译并评估模型
model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
    metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, 
    batch_size=128, validation_split=0.2) 

#%% 训练一个双向GRU模型
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Bidirectional(
    layers.GRU(32), input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
# 编译并评估模型
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500,
    epochs=40, validation_data=val_gen, validation_steps=val_steps)
