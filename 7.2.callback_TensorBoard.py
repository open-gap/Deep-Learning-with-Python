#%% ModelCheckpoint与EarlyStopping回调函数
import keras
# 通过fit的callbacks参数将回调函数传入模型中，这个参数接收一个
# 回调函数的列表。你可以传入任意个数的回调函数
callbacks_list = [
    # 中断训练函数，训练结果不再改善则中断训练
    keras.callbacks.EarlyStopping(
        monitor='acc', #监控模型的精度变化
        patience=1, #如果精度在多于1个轮次的时间内不再改善中断训练
    ), 
    # 模型保存函数，每轮过后保存当前权重
    keras.callbacks.ModelCheckpoint(
        filepath='my_model.h5', #目标模型文件保存路径
        monitor='val_loss', #监控指标为验证集损失
        save_best_only=True, #根据监控指标，只保存最好结果
    )
]
model = keras.Model()
model.compile(optimizer='adam', loss='binary_crossentropy', 
    metrics=['acc'])
# 由于回调函数要监控验证损失，所以需要验证数据
x, y, x_val, y_val = [], [], [], []
model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks_list, 
    validation_data=(x_val, y_val))

#%% ReduceLROnPlateau回调函数，监控指标不再改善时降低学习率
callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', #监控模型的验证损失
        factor=0.1, #触发时将学习率降低为原来的0.1倍
        patience=10, #如果监控指标在10轮内没有改善就触发该回调函数
    )
]
model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks_list, 
    validation_data=(x_val, y_val)) #同上个模型

#%% 自定义回调函数
import keras
import numpy as np 
class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model #在训练之前由父模型调用，告诉回调函数是哪个模型
        # 获取模型的每层输出
        layer_outputs = [layer.outputs for layer in model.layers]
        # 使用Model构建包含原模型所有层输出的新模型
        self.activations_model = keras.models.Model(model.input, 
            layer_outputs)

    # 在每轮结束时被调用
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Require validation_data.')
        # 获取验证数据的第一个第一个输入样本
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        with open('activations_at_epoch_' + str(epoch) + '.npz', 'w') as f:
            # 将数组保存到硬盘
            np.savez(f, activations)

    # 在每轮开始时被调用
    def on_epoch_begin(self):
        pass

    # 在处理每个批次之前被调用
    def on_batch_begin(self):
        pass

    # 在处理每个批次之后被调用
    def on_batch_end(self):
        pass

    # 在训练开始时被调用
    def on_train_begin(self):
        pass

    # 在训练结束时被调用
    def on_train_end(self):
        pass

#%% 只有在Keras使用了TensorFlow后端时才能使用TensorBoard处理Keras模型
# 使用TensorBoard的文本分类模型
import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 2000  #考虑最常用的2000个单词
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
# 建立一维卷积模型
model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, 
    name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.summary()
# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', 
    metrics=['acc'])
# 设置回调函数列表
callbacks_list = [
    # TensorBoard回调函数，记录TensorBoard易于读取的日志文件
    keras.callbacks.TensorBoard(
        log_dir='my_log_dir', #日志文件保存地址
        histogram_freq=1, #每1轮后记录激活直方图
        embeddings_freq=1, #每1轮后记录嵌入数据
    )
]
# 模型开始训练后，控制台开启TensorBoard服务器，
# 可以在 https://localhost:6006 看实时图表
# 在浏览器中可以查看SCALARS指标监控、HISTOGRAMS直方图、EMBEDDINGS嵌入图
# 还可以在GRAPHS图标签页查看TensorFlow运算图可视化
history = model.fit(x_train, y_train, epochs=20, batch_size=128, 
    validation_split=0.2, callbacks=callbacks_list)

#%% 利用Keras将模型绘制为层组成的图并保存到文件中
from keras.utils import plot_model
# 保存为图片需要pydot库、pydot-ng库和graphviz库！
plot_model(model, to_file='model.png')
# 利用show_shapes参数将模型拓扑结构可视化
plot_model(model, show_shapes=True, to_file='model_shape.png')