#%% 启动Keras和TensorFlow
import keras
keras.__version__

#%% 将VGG16卷积基实例化
from keras.applications import VGG16
conv_base = VGG16(
    weights='imagenet', #指定模型初始化的权重检查点
    include_top=False, #指定模型是否包含最后的全连接层
    input_shape=(150, 150, 3) #指定输入到模型中的图像张量形状
)
conv_base.summary() #输出模型概况

#%% 使用预训练的卷积基提取特征
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
base_dir = 'D:/project/machine_learning_databases/dogs-vs-cats'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
datagen = ImageDataGenerator(rescale=1. / 255) #将图片RGB值转化为浮点数
batch_size = 20 #定义每个批次的大小
# 定义批次读取图片并提取特征函数
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory, #生成器目录地址
        target_size=(150, 150), #输出图片尺寸
        batch_size=batch_size, #批次大小
        class_mode='binary' #使用二进制标签
    )
    i = 0 #循环计数
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break #在读取完所有图像后终止循环
    return features, labels
# 使用上面定义的批次特征提取函数提取训练、验证和测试数据集的特征
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
# 将提取的特征展平，类似于Keras中的Flatten函数，展平的大小取决于VGG16网络特征的输出
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

#%% 定义并训练密集连接分类器（全连接层）
from keras import models, layers, optimizers
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5)) #添加Dropout层减少过拟合
model.add(layers.Dense(1, activation='sigmoid')) #最后输出结果
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_features, train_labels, epochs=30, batch_size=20,
    validation_data=(validation_features, validation_labels))

#%% 绘制结果
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

#%% 使用数据增强的特征提取减少数据集过小带来的过拟合问题
from keras import models, layers
model = models.Sequential() #新建模型
model.add(conv_base) #在新模型中直接添加已有模型
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu')) #添加的Dense层是随机初始化的！
model.add(layers.Dense(1, activation='sigmoid'))
model.summary() #输出模型概况
# 测试冻结网络前后的区别
print('This is the number of trainable weights '
      'before freezing the conv_base:', len(model.trainable_weights))
conv_base.trainable = False #冻结VGG16特征提取部分网络
print('This is the number of trainable weights '
      'after freezing the conv_base:', len(model.trainable_weights))

#%% 利用冻结的卷积基进行端到端地训练模型并保存
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255) #验证数据集不允许增强！
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150), #统一图片大小为 150*150
        batch_size=20,
        class_mode='binary' #因为使用了二进制交叉熵所以需要二进制标签
)
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
)
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), 
    loss='binary_crossentropy', metrics=['acc'])
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      # verbose共有三个值，0为不输出，1为带进度条输出，2为不带进度条输出
      # verbose默认值为 1
      verbose=2
)
model.save('cats_vs_dogs_small_3.h5') #模型的保存

#%% 绘制结果，代码同上完全相同
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

#%% 模型微调(fine-tuning)，即仅解冻已有模型的顶部几层加入训练
conv_base.trainable = True #解冻VGG16模型
set_trainable = False #设置逐层冻结标志
for layer in conv_base.layers: #循环VGG16网络各层直到指定某层网络为止
    if layer.name == 'block5_conv1': #根据VGG16网络各层名称决定
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
# 调节模型后记得重新编译模型使得改变生效，否则模型的改动无效
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50, #原始代码训练次数为100次，为节约时间减少为50次
      validation_data=validation_generator,
      validation_steps=50
)
model.save('cats_vs_dogs_small_4.h5') #模型的保存

#%% 绘制结果，代码同上和上上完全相同
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

#%% 使用指数移动平均法获得平滑曲线
def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points
plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%% 在测试数据集上最终评估模型
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
)
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('Test accuracy:', test_acc)