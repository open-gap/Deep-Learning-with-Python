#%% 启动Keras和TensorFlow
import keras
keras.__version__

#%% 加载预训练的Inception V3模型
from keras import backend as K
from keras.applications import inception_v3
K.set_learning_phase(0) #使用该命令禁用所有与训练有关的操作
# 构建不包括全连接层的Inception V3网络，使用ImageNet权重
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
model.summary()

#%% 设置DeepDream配置
# 这个字典将层的名称映射为一个系数，这个系数定量表示该层激活对你要最大化的损失
# 的贡献大小。注意：层的名称硬编码在内置的Inception V3应用中。
layer_contributions = { 
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
}

#%% 定义需要最大化的损失
# 获取不同层的名称，建立字典实现名称与层的实例的映射
layer_dict = dict([(layer.name, layer) for layer in model.layers])
loss = K.variable(0.0) #在定义损失时将层的贡献添加到这个标量变量中
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name] #获取当前层的权重
    activation = layer_dict[layer_name].output #获取当前层的输出
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    # 将该层特征的L2范数添加到loss中，为了避免出现边界伪影，损失中仅包含非边界的像素
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

#%% 设置梯度上升过程
dream = model.input #这个张量用于保存生成的图像，即梦境图像
grads = K.gradients(loss, dream)[0] #计算损失相对于梦境图像的梯度
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7) #将梯度标准化(重要技巧)
outputs = [loss, grads] #设置输出结果保存列表
fetch_loss_and_grads = K.function([dream], outputs) #设置输入输出的Keras函数
# 使用之前定义的Keras函数的Python函数
def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values
# 这个函数运行iteration次梯度上升
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values #梯度上升
    return x

#%% 设置需要使用的辅助函数，注意需要SciPy库和Pillow库
import scipy
from PIL import Image
from keras.preprocessing import image
# 改变图像尺寸
def resize_img(img, size):
    img = np.copy(img)
    factors = (1, float(size[0]) / img.shape[1], 
        float(size[1]) / img.shape[2], 1)
    return scipy.ndimage.zoom(img, factors, order=1)
# 保存图像
def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    pil_img = Image.fromarray(pil_img)
    pil_img.save(fname)
# 打开图像、改变图像大小并预处理图像得到Inception V3模型能够处理的张量
def preprocess_image(image_path):
    with open(image_path): #测试文件是否存在
        pass
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img
# 将处理后的图像张量转化为可以显示的图片格式数据
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.0
    x += 0.5
    x *= 255.0
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#%% 在多个连续尺度上运行梯度上升，注意：更改超参数可以生成不同的结果图
import numpy as np
step = 0.01  #梯度上升步长
num_octave = 3  #运行梯度上升的尺度次数
octave_scale = 1.4  #连续尺度的放大倍数
iterations = 20  #在每个尺度上运行梯度上升的次数
max_loss = 10.0 #当损失函数超过预设值中断训练避免得到丑陋的伪影
base_image_path = 'D:/picture/lofter_harexhare_Ruby.jpg' #原始图片地址！！！
img = preprocess_image(base_image_path) #读取并预处理图像
original_shape = img.shape[1:3] #输入图像尺寸
successive_shapes = [original_shape] #图像尺寸列表
for i in range(1, num_octave + 1):
    # 得到一个运行梯度上升的不同尺寸组成的元组
    shape = tuple([int(dim / (octave_scale ** i)) \
        for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1] #将列表翻转，变为升序
# 将元素图片调整为我们计算得到的最小尺寸
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])
# 根据图像尺寸列表逐尺度的连续放大
for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape) #将梦境图像放大
    img = gradient_ascent(img, iterations=iterations, step=step,
        max_loss=max_loss) #运行梯度上升，改变梦境图像
    # 将原始图像的较小版本放大，它会变得像素化
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    # 由原始大图重新生成该尺寸图像，它具有较高图像质量
    same_size_original = resize_img(original_img, shape)
    # 二者图像的差别就是在放大过程中丢失的细节
    lost_detail = same_size_original - upscaled_shrunk_original_img
    img += lost_detail #将丢失的细节重新注入到梦境图像中
    # 得到下次循环时原始的较小图像
    shrunk_original_img = resize_img(original_img, shape)
    # 保存不同尺寸的梦境图像
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
# 保留最终结果的梦境图像
save_img(img, fname='final_dream.png')

#%% 显示最终的结果图像
from matplotlib import pyplot as plt
plt.imshow(deprocess_image(np.copy(img)))
plt.show()