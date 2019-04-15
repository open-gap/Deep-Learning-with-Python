#%% 启动Keras和TensorFlow
import keras
keras.__version__

#%% GAN生成器网络，它将一个潜在空间向量转化为一张候选图像
import numpy as np
from keras import layers
latent_dim = 32 #潜在空间向量的维度
height = 32 #生成器产生图像的高度
width = 32 #生成器产生图像的宽度
channels = 3 #生成器产生图像的通道数
# 建立模型
generator_input = keras.Input(shape=(latent_dim, ))
# 首先，将输入转化为大小16x16的128个通道的特征图
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)
# 然后，添加卷积层
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
# 上采样为32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)
# 更多的卷积层
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
# 产生一个大小为32x32的指定通道特征图(即CIFAR10图像的形状)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x) #将生成器模型实例化
generator.summary()

#%% GAN判别器网络，它接收一张候选图片输入并分类为真实图像和生成图像之一
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x) #一个dropout层：这是很重要的技巧
x = layers.Dense(1, activation='sigmoid')(x) #分类层
discriminator = keras.models.Model(discriminator_input, x) #模型实例化
discriminator.summary()
# 设置模型优化器，使用clipvalue参数进行梯度裁剪(限制梯度的范围)，
# 使用decay参数进行学习率衰减
discriminator_optimizer = keras.optimizers.Adam(lr=0.0003, 
    clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, 
    loss='binary_crossentropy')

#%% 对抗网络
# 结合生成器模型和判别器模型，建立总体的GAN模型
gan_input = keras.Input(shape=(latent_dim, ))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan.summary()
# 设置优化器，参数含义同判别器网络
gan_optimizer = keras.optimizers.Adam(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
discriminator.trainable = False #将判别器网络权重设置为不可训练(仅用于GAN模型)

#%% 实现GAN的训练
import os
from keras.preprocessing import image
iterations = 10000 #训练迭代次数
batch_size = 16 #批量大小
save_dir = './temp/' #指定保存生成图像的目录
# 加载CIFAR10数据集的数据
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
# 选择青蛙图像(类别编号为6)
x_train = x_train[y_train.flatten() == 6]
x_train = x_train[: (len(x_train) // batch_size) * batch_size]
# 数据标准化
x_train = x_train.reshape(
    (x_train.shape[0], )+(height, width, channels)).astype('float32') /255.0
# 开始训练循环
start = 0
for step in range(iterations):
    # 在潜在空间中采样随机点
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # 将采样的随机点解码为虚假图像
    generated_images = generator.predict(random_latent_vectors)
    # 将虚假图像与真实图像混合在一起
    real_images = x_train[start: (start + batch_size)]
    combined_images = np.concatenate([generated_images, real_images])
    # 合并标签，区分真实图像和虚假图像
    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    # 向标签中添加随机噪声，这是一个很重要的技巧
    labels += 0.05 * np.random.random(labels.shape)
    # 训练判别器
    d_loss = discriminator.train_on_batch(combined_images, labels)
    # 在潜在空间中采样随机点
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # 合并标签，全部都是“真实图像”(这是在撒谎)
    misleading_targets = np.zeros((batch_size, 1))
    # 通过GAN模型来训练生成器，此时冻结判别器权重
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    # 设置新的起始图片标签位置
    start += batch_size
    if start > len(x_train) - batch_size:
      start = 0
    # 每100步保存并绘图
    if step % 100 == 0:
        # 保存模型权重
        gan.save_weights('gan.h5')
        # 将指标打印出来
        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))
        # 保存一张生成图像
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
        # 保存一张真实图像用于对比
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))

#%% 显示生成的图像
import matplotlib.pyplot as plt
# 在潜在空间中随机采样点
random_latent_vectors = np.random.normal(size=(10, latent_dim))
# 将它们解码为虚假图像
generated_images = generator.predict(random_latent_vectors)
# 循环遍历生成的虚假图像批次
for i in range(generated_images.shape[0]):
    img = image.array_to_img(generated_images[i] * 255., scale=False)
    plt.figure()
    plt.imshow(img)
# 显示所有图像
plt.show()