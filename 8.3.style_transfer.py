#%% 启动Keras和TensorFlow
import keras
keras.__version__

#%% 定义初始变量
from keras.preprocessing.image import load_img, img_to_array
# 想要变换的图像路径
target_image_path = 'E:/pixiv_8913281.jpg'
# 风格图像的路径
style_reference_image_path = 'E:/glaxy.jpg'
width, height = load_img(target_image_path).size #读取读取图像尺寸
img_height = 400 #统一图像的高度
img_width = int(width * img_height / height) #变换后的图像宽度

#%% 辅助函数，用于图像的加载、预处理和后处理
import numpy as np
from keras.applications import vgg19
# 图像的读取与预处理
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img
# 图像的后处理，将生成图像张量转化为可显示的RGB图像格式
def deprocess_image(x):
    # 由于vgg19的preprocess_input的作用是减去ImageNet的平均像素值，
    # 这里相当于是上述操作的逆操作
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1] #将BGR格式转化为RGB格式，也是preprocess_input的逆操作
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#%% 加载预训练的VGG19网络，并将其应用于三张图像
from keras import backend as K
# 分别将目标图像和风格参考图像设置为不变的常量
target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(
    style_reference_image_path))
# 生成用于保存生成图像的占位符，它是可变的
combination_image = K.placeholder((1, img_height, img_width, 3))
# 将三张图像合并为一个批量
input_tensor = K.concatenate([target_image, style_reference_image,
    combination_image], axis=0)
# 加载ImageNet预训练权重的VGG19网络，设置网络输入为图像组合批量
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')

#%% 定义损失函数
# 定义内容损失函数，保证目标图像和生成图像在网络顶层具有相似结果
def content_loss(base, combination):
    return K.sum(K.square(combination - base))
# 定义风格损失函数，保证风格参考图像和生成图像在不同的层激活内保持
# 相似的内部相互关系
def gram_matrix(x): #计算输入矩阵的格拉姆矩阵(Gram matrix)
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# 计算风格损失
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels**2) * (size**2))

# 总变差损失(total variation loss)即正则化(损失)函数，避免结果过度像素化
def total_variation_loss(x):
    a = K.square( x[:, :img_height - 1, :img_width - 1, :] - \
        x[:, 1:, :img_width - 1, :])
    b = K.square( x[:, :img_height - 1, :img_width - 1, :] - \
        x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# 定义所需要最小化的最终损失函数
# outputs_dict将层的名称映射为激活张量的字典
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2' #计算内容损失需要的靠顶部的层的名称
# 风格损失需要计算的一系列层的名称
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
    'block4_conv1', 'block5_conv1']
# 损失分量的加权平均所使用的权重
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025
# 添加内容损失
loss = K.variable(0.0) #在定义内容损失时将所有分量添加到这个标量变量中
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, 
    combination_features)
# 添加每个目标层的风格损失分量
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
# 添加总变差损失
loss += total_variation_weight * total_variation_loss(combination_image)

#%% 设置梯度下降过程
grads = K.gradients(loss, combination_image)[0] #获取损失对于生成图像的梯度
# 用于获取当前损失值和当前梯度值的函数
fetch_loss_and_grads = K.function([combination_image], [loss, grads])
# 定义类将上者包装起来，让你可以利用两个单独的方法调用来获取损失和梯度，
# 这是我们要使用SciPy优化器所要求的
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
# 实例化Evaluator类
evaluator = Evaluator()

#%% 风格迁移循环
import time
from PIL import Image
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
result_prefix = 'style_transfer_result' #生成图像名称前缀
iterations = 20 #循环次数
x = preprocess_image(target_image_path) #这是初始状态的目标图像
x = x.flatten() #将图像展平，因为scipy.optimize.fmin_l_bfgs_b只能处理展平向量
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    # 对生成图像的像素使用L-BFGS最优化，以将神经风格损失最小化。
    # 注意：必须将计算损失的函数和计算梯度的函数作为两个单独参数输入
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, 
        fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # 保存当前生成的图像
    # img = x.copy().reshape((img_height, img_width, 3))
    # img = deprocess_image(img)
    # fname = result_prefix + '_at_iteration_%d.png' % i
    # img = Image.fromarray(img).save(fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
# 保存最终结果图像
img = x.copy().reshape((img_height, img_width, 3))
img = deprocess_image(img)
fname = result_prefix + '.png'
Image.fromarray(img).save(fname)
print('\nImage saved as', fname)

#%% 显示风格迁移处理结果图像
import matplotlib.pyplot as plt 
plt.imshow(img)