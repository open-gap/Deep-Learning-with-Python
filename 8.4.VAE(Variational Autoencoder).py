#%% 启动Keras和TensorFlow
import keras
from keras import backend as K
K.clear_session()
keras.__version__

#%% 自编码器实现的示例代码
# # 将输入编码为平均值和方差两个参数
# z_mean, z_log_variance = encoder(input_img)
# # 使用小随机数epsilon来抽取一个潜在点
# z = z_mean + exp(z_log_variance) * epsilon
# # 将z解码为一张图像
# reconstructed_img = decoder(z)
# # 将自编码器模型实例化，它将一张输入图像映射为它的重构
# model = Model(input_img, reconstructed_img)

#%% 构建VAE自编码器网络模型
import keras
import numpy as np
from keras import Input, layers
from keras import backend as K
from keras.models import Model
img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2  #潜在空间的维度：一个二维平面
# 构建编码器模型，将输入图像编码为潜在空间中的点
input_img = Input(shape=img_shape)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='relu', 
    strides=(2, 2))(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x) #保存模型输出未展平前的尺寸
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
# 输入图像最终被编码为平均值和方差两个参数
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# 构建潜在空间采样的函数
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
        mean=0.0, stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon
# 在Keras任何对象都应该是一个层，否则应该将其包装到一个Lambda层(或自定义层)中
z = layers.Lambda(sampling)([z_mean, z_log_var])

# 构建VAE解码器网络，将潜在空间点映射为图像
decoder_input = layers.Input(K.int_shape(z)[1:]) #需要将z输入到这里
# 对输入进行上采样
x = layers.Dense(np.prod(shape_before_flattening[1:]), 
    activation='relu')(decoder_input)
# 将z转换为特征图，使其形状与编码器模型最后一个展平前输出特征图形状相同
x = layers.Reshape(shape_before_flattening[1:])(x)
# 使用Conv2DTranspose层和Conv2D层将z解码为与原始输入图像具有相同尺寸的特征图
x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu',
    strides=(2, 2))(x) #又名反卷积层
x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
# 将解码器模型实例化，它将decoder_input转化为解码后的图像
decoder = Model(decoder_input, x)
z_decoded = decoder(z) #将这个实例应用于z，以得到解码后的z

# 用于计算VAE损失的自定义层
class CustomVariationalLayer(keras.layers.Layer):
    # 计算变分自编码器的双重损失，并组合双重损失
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    # 通过编写一个call方法来实现自定义层
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x #我们不使用这个输出，但层必须要返回值
# 对输入和解码后的输出调用自定义层，以得到最终的模型输出
y = CustomVariationalLayer()([input_img, z_decoded])

#%% 将模型实例化并训练VAE模型
from keras.datasets import mnist
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()
# 加载MNIST数据集用于训练模型，由于训练过程不需要传入目标数据，因此省略y_train
(x_train, _), (x_test, y_test) = mnist.load_data()
# MNIST数据集数据的预处理
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))
# 训练模型，同时由于使用自定义损失函数，因此无需指定loss外部损失函数参数
vae.fit(x=x_train, y=None, shuffle=True, epochs=10, batch_size=batch_size,
    validation_data=(x_test, None))

#%% 模型训练好后，从二维潜在空间中采样一组点的网格，并将其解码为图像
from scipy.stats import norm
import matplotlib.pyplot as plt
n = 15  #我们将显示15x15的数字网格(共255个图片)
digit_size = 28 #每个图片的大小
figure = np.zeros((digit_size * n, digit_size * n))
# 使用SciPy的ppf函数对线性分隔的坐标进行变换，以生成潜在变量z的值，
# 因为潜在空间的先验分布是高斯分布
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
# 对潜在空间的网格坐标遍历生成新的图像
for i, xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        # 将z重复多次以便构建一个完整批次
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        # 将批量第一个数字的形状从28x28x1转变为28x28
        digit = x_decoded[0].reshape(digit_size, digit_size)
        # 将结果保存到输出图像缓存
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit
# 绘制结果图像
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()