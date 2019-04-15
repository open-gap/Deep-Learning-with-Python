#%% 启动TensorFlow和Keras
import keras
keras.__version__

#%% 导入之前保存的模型文件
from keras.models import load_model
model = load_model('cats_vs_dogs_small_2.h5')
model.summary()  #输出模型概况

#%% 方法1：可视化不同网络层的输出结果
# 预处理单张图像并显示
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image #Keras的自动图像处理模块
img_path = \
    'D:/project/machine_learning_databases/dogs-vs-cats/test/cats/cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img) #将导入图像转化为numpy数组
# 在图片数组开头增加一个维度，构造4维数组
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.0 #记住训练模型输入数据需要预处理为浮点数！
print('Image shape:', img_tensor.shape)
plt.imshow(img_tensor[0])
plt.show()

#%% 建立Model多输出模型获取原始模型中间层的输出结果
from keras import models
# 提取模型的前8层输出结果
layer_outputs = [layer.output for layer in model.layers[: 8]]
# 创建一个Model模型，在给定模型输入情况下返回指定中间层输出(多输出模型)
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor) #获取新模型输出的所有特征图
first_layer_activation = activations[0] #得到输出结果中的第一个特征图
print('Activation-map shape:', first_layer_activation.shape)

#%% 将模型输出的特征图可视化
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()
plt.figure()
plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()

#%% 就每个中间激活的所有通道可视化
layer_names = [] #利用循环得到前8层的名称
for layer in model.layers[: 8]:
    layer_names.append(layer.name)
images_per_row = 16 #规定输出结果中每行特征图的个数
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1] #特征图中特征的个数
    size = layer_activation.shape[1] #特征图形状(1, size, size, n_features)
    n_cols = n_features // images_per_row #决定输出结果列数
    # 建立空的输出结果图用于显示所以特征图
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): #对输出图逐列循环绘制
        for row in range(images_per_row): #逐行遍历逐个子图
            channel_image = layer_activation[0, :, :,
                col * images_per_row + row]
            # 对特征图进行数据处理优化显示效果
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            # 图像数组取阈值避免出现不合理值
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # 在结果网格中显示图片
            display_grid[col * size : (col + 1) * size,
                row * size : (row + 1) * size] = channel_image
    scale = 1. / size #确定特征图缩放比例
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()

#%% 方法2：利用空白输入图像和梯度上升将指定过滤器最大响应图像可视化
import numpy as np
from keras import backend as K #Keras模型参数计算模块
from keras.applications import VGG16
model = VGG16(weights='imagenet', include_top=False) #导入VGG16特征提取模型
# 定义预处理函数，将梯度上升结果图转化为合理的梯度图像
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1) #取阈值0~1
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8') #转化为0~255的整数RGB数组
    return x
# 定义过滤器可视化函数，输入网络层名称和过滤器索引，输出使得激活最大化的输入图像
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output #获取指定网络层输出
    loss = K.mean(layer_output[:, :, :, filter_index]) #取得不同图片平均loss值
    # gradients返回张量列表，本例中列表长度为一，故仅保留第一个梯度元素
    grads = K.gradients(loss, model.input)[0] #获取损失对于输入的梯度
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) #梯度标准化，1e-5防止除0
    # 定义目标处理函数，输入为模型的输入，输出为loss值和梯度！
    iterate = K.function([model.input], [loss, grads])
    # 定义初始带噪声的灰度图
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.0
    step = 1.0 #梯度上升步长
    for i in range(40): #进行40次梯度上升操作，更新输入图像
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0] #取3维RGB模型数据作为图片
    return deprocess_image(img)
plt.imshow(generate_pattern('block3_conv1', 0)) #测试上述函数
plt.title('block3_conv1')
plt.show()

#%% 生成某一层中所有过滤器相应模式组成的网络
layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
for layer_name in layer_names:
    size = 64 #结果网格图尺寸(仅显示前64个过滤器相应模式图)
    margin = 5 #网格黑边宽度
    # 生成保存结果的黑色空网络图
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3), 
        dtype='uint8')
    for i in range(8):  #遍历results行
        for j in range(8):  #遍历results列
            # 生成指定网络层第i+(j*8)个过滤器模式
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
            # 确定图片位置坐标
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            # 保存模式结果到网格图中
            results[horizontal_start: horizontal_end, 
                vertical_start: vertical_end, :] = filter_img
    plt.figure(figsize=(20, 20)) #循环生成结果图
    plt.imshow(results)
    plt.title(layer_name)
    plt.show()

#%% 方法3：类激活图(CAM, class activation map)可视化
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
K.clear_session() #清除之前训练结果
model = VGG16(weights='imagenet') #重新加载完整VGG16网络
img_path = 'E:/temp.jpg'
img = image.load_img(img_path, target_size=(224, 224)) #载入图像并调整大小
x = image.img_to_array(img) #转化PIL(Python图像库)图像为浮点numpy数组
plt.imshow(x / 255.0)
plt.title('Original picture')
plt.show()
x = np.expand_dims(x, axis=0) #增加一个维度构造4维向量
x = preprocess_input(x) #对批量进行批处理(按通道进行颜色标准化)

#%% 生成类激活热力图
import numpy as np
preds = model.predict(x) #利用VGG16预测图片类别
print('Predicted:', decode_predictions(preds, top=3)[0]) #输出预测结果
print('Index number:', np.argmax(preds[0])) #输出最大类别索引
# 应用Grad-GAM算法生成图像类激活热力图
# 获取预测向量中的“非洲象”元素
african_elephant_output = model.output[:, np.argmax(preds[0])]
# 获取VGG16最后一个卷积层输出特征图
last_conv_layer = model.get_layer('block5_conv3')
# 得到“非洲象”类别对于block5_conv3输出特征图的梯度
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2)) #取特定特征图通道的梯度平均大小
# 建立目标函数，输入为模型输入，输出为平均梯度和最后一层卷积输出
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x]) #使用目标函数处理图像
# 将特征图数组的每个通道乘以这个通道对于“非洲象”类别的重要程度
for i in range(pooled_grads.shape[0]):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
# 得到的特征图的逐通道平均值即为类激活热力图
heatmap = np.mean(conv_layer_output_value, axis=-1)
# 热力图后处理
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

#%% 使用OpenCV将热力图与原始图叠加
import cv2
img = cv2.imread(img_path) #读取图像
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) #调整大小
heatmap = np.uint8(255 * heatmap) #转化热力图为RGB格式
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) #将热力图用于原始图
superimposed_img = heatmap * 0.4 + img #这里0.4是热力图强度因子
# 保存图片
cv2.imwrite('E:/temp2.jpeg', superimposed_img) #保存成功返回True