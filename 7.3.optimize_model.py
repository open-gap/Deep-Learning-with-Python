#   标准化(normalization)
#   1.批标准化(batch normalization)，已经用于
# Keras内置的ResNet50、Inception V3和Xception，
# 一般用于卷积层和Dense层之后，axis参数一般不改
#   2.批再标准化(batch renomalization)
#   3.自标准化神经网络(self-normalizing neural network)
# 使用特殊激活函数(selu)和初始化器(lecun_normal)

#   深度可分离卷积(depthwise separable convalution)
# Keras中为SeparableConv2D，即先对每个通道进行独立的空间
# 卷积，再连接起来使用1x1的逐点卷积

#%% 在小型数据集上构建轻量的深度可分离卷积神经网络
from keras import layers
from keras.models import Sequential, Model
height = 64
width = 64
channels = 3
num_classes = 10
model = Sequential()
model.add(layers.SeparableConv2D(32, 3, activation='relu', 
    input_shape=(height, width, channels, )))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.MaxPool2D(2))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.MaxPool2D(2))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.GlobalAvgPool2D())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', 
    metrics=['acc'])

#%% 超参数优化问题
# 可以尝试使用Python的Hyperopt库或Hyperas库，后者将Hyperopt与Keras模型集成

#%% 模型集成(model ensembling)
# 将一组分类器结果汇集在一起，即分类器集成(ensemble the calssfiers)，可以得到
# 好于所以模型的结果，最简单的方式是取平均值，更聪明的方式是加权平均
# (采用Nelder-Mead方法优化)
# 另一种集成方式是宽且深(wide and deep)的模型类型，它集合了深度学习和浅层学习