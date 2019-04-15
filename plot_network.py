#%% 绘制指定模型的结构图，需要pydot、pydot_ng和graphviz库
from keras import utils, layers, models, Sequential, applications
model = applications.VGG16(weights='imagenet', #指定模型初始化的权重检查点
    include_top=False, #指定模型是否包含最后的全连接层
    input_shape=(224, 224, 3) #指定输入到模型中的图像张量形状
)
model.summary() #输出模型概况

# utils.plot_model(model, 'AlexNet.png', show_layer_names=False, 
#     show_shapes=True)


#%% AlexNet
# model = Sequential()
# #第一段
# model.add(layers.Conv2D(filters=96, kernel_size=(11,11),
#                  strides=(4,4), padding='valid',
#                  input_shape=(227,227,3),
#                  activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D(pool_size=(3,3), 
#                        strides=(2,2), 
#                        padding='valid'))
# #第二段
# model.add(layers.Conv2D(filters=256, kernel_size=(5,5), 
#                  strides=(1,1), padding='same', 
#                  activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D(pool_size=(3,3), 
#                        strides=(2,2), 
#                        padding='valid'))
# #第三段
# model.add(layers.Conv2D(filters=384, kernel_size=(3,3), 
#                  strides=(1,1), padding='same', 
#                  activation='relu'))
# model.add(layers.Conv2D(filters=384, kernel_size=(3,3), 
#                  strides=(1,1), padding='same', 
#                  activation='relu'))
# model.add(layers.Conv2D(filters=256, kernel_size=(3,3), 
#                  strides=(1,1), padding='same', 
#                  activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(3,3), 
#                        strides=(2,2), padding='valid'))
# #第四段
# model.add(layers.Flatten())
# model.add(layers.Dense(4096, activation='relu'))
# model.add(layers.Dropout(0.5))
 
# model.add(layers.Dense(4096, activation='relu'))
# model.add(layers.Dropout(0.5))

# # Output Layer
# model.add(layers.Dense(1000, activation='relu'))


#%% LeNet-5
# input_ = models.Input(shape=(28, 28, 1))
# C1 = layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), 
#     padding='same', activation='tanh')(input_)
# S2 = layers.AveragePooling2D(pool_size=(2, 2))(C1)
# C3 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), 
#     activation='tanh')(S2)
# S4 = layers.AveragePooling2D(pool_size=(2, 2))(C3)
# x = layers.Flatten()(S4)
# C5 = layers.Dense(units=120, activation='tanh')(x)
# F6 = layers.Dense(units=84, activation='tanh')(C5)
# output = layers.Dense(units=10, activation='softmax')(F6)
# model = models.Model(input_, output)