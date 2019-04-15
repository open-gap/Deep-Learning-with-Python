#%% 从Kaggle竞赛猫狗分类数据集中构建训练、验证和测试的小型数据集
import os, shutil
original_dataset_dir = 'E:/dogs vs cats/train' #原始训练数据集地址
# 模型保存地址
base_dir = 'D:/project/machine_learning_databases/dogs-vs-cats'
os.mkdir(base_dir) #生成文件夹
# 分别建立训练、验证和测试目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
# 分别建立猫和狗的训练、验证和测试目录
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# 从Kaggle竞赛猫狗分类问题数据集中的训练集中抽取部分数据构建小型数据集
# 默认抽取前1000张作为训练集，接着500张验证集，接着500张测试集
# 数据集共2000张训练图、1000张验证图、1000张测试图
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 检查构建的新数据集的图片数量
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

#%% 构建小型卷积神经网络
from keras import models, layers, optimizers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', 
    input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary() #输出模型概况
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), 
    loss='binary_crossentropy', metrics=['acc'])

#%% 图像数据预处理
# 利用Keras自动工具构建自动处理图像的Python生成器
from keras.preprocessing.image import ImageDataGenerator
# 将所有图像的RGB值缩放为1/255倍，即将整数值变为0~1之间的浮点数
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir, #目标目录
    target_size=(150, 150), #将所有图像大小统一
    batch_size=20, #规定每批次读取图片数量
    class_mode='binary' #使用二进制标签
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
# 测试生成器
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break #必须设置循环中断条件

#%% 利用批量生成器拟合模型
history = model.fit_generator(
    train_generator, #训练数据生成器
    #每个批次20个样本情况下，需要100批次才能读取完所有2000张训练图片
    steps_per_epoch=100, 
    epochs=30, #训练次数
    validation_data=validation_generator, #验证数据生成器
    validation_steps=50 #说明需要从验证数据生成器中抽取多少个批次
)
# 模型的保存 ###################################################
save_dir = 'D:/project/tensorflow_keras_project'
model.save(os.path.join(save_dir, 'cats_vs_dogs_small_1.h5'))

#%% 绘制训练过程损失和精度曲线
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

#%% 使用数据增强
from keras.preprocessing import image #图像预处理工具模块
datagen = ImageDataGenerator(
    rotation_range=40, #图像随机旋转角度范围，最小为0
    width_shift_range=0.2, #宽度平移比例
    height_shift_range=0.2, #高度平移比例
    shear_range=0.2, #随机错切变换比例
    zoom_range=0.2, #随机缩放比例
    horizontal_flip=True, #随机一半图片水平翻转
    fill_mode='nearest' #缺失像素填充方式
)
# 显示数据增强后的图片效果
fnames = [os.path.join(train_cats_dir, fname) 
    for fname in os.listdir(train_cats_dir)]
img_path = fnames[3] #选定某张图片进行测试
img = image.load_img(img_path, target_size=(150, 150)) #读取图片并统一大小
x = image.img_to_array(img) #转化为numpy数组
x = x.reshape((1,) + x.shape) #改变数组维度
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break #必须添加循环终止条件
plt.show()

#%% 在原来模型基础的Flatten层后添加Dropout层降低过拟合并训练
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', 
    input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary() #输出模型概况
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), 
    loss='binary_crossentropy', metrics=['acc'])
# 利用数据增强生成器训练神经网络
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255) #验证数据不能增强！！
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
)
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50
)
# 保存模型
model.save(os.path.join(save_dir, 'cats_vs_dogs_small_2.h5'))

#%% 再次绘制训练过程损失和精度曲线
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