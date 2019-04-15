#%% 启动Keras和TensorFlow
import keras
keras.__version__

#%% 一个简单的Sequenctial模型及对应的函数式API实现
from keras import Input, layers
from keras.models import Sequential, Model
# 前面使用的Sequential模型
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64, )))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))
# 使用函数式API实现相同的模型
input_tensor = Input(shape=(64, ))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = Model(input_tensor, output_tensor)
model.summary()

#%% 编译模型并生成测试模型用的随机数据
import numpy as np 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))
model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)
print('Model score:', score)

#%% 用函数式API实现双输入的问答模型
from keras import Input, layers
from keras.models import Sequential, Model
text_vocabulary_size = 10000
question_vovabulary_size = 10000
answer_vocabulary_size = 500
# 定义模型输入为可变长度整数序列，通过name指定输入名称
text_input = Input(shape=(None, ), dtype='int32', name='text')
# 将输入嵌入长度为64的向量
embedding_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedding_text) #利用LSTM将向量编码为单个向量
# 对输入问题进行同样的处理(使用不同的层实例)
question_input = Input(shape=(None, ), dtype='int32', name='question')
embedding_question = layers.Embedding(question_vovabulary_size, 
    32)(question_input)
encoded_question = layers.LSTM(16)(embedding_question)
concatenaded = layers.concatenate([encoded_text, encoded_question], 
    axis=-1) #将编码后的问题和文本连接起来
# 在上面添加一个softmax分类器
answer = layers.Dense(answer_vocabulary_size, 
    activation='softmax')(concatenaded)
model = Model([text_input, question_input], answer) #模型实例化指定两个输入
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
    metrics=['acc'])

#%% 将数据输入到多输入模型中
import numpy as np 
num_samples = 1000
max_length = 100
# 生成虚构的Numpy数据
text = np.random.randint(1, text_vocabulary_size, 
    size=(num_samples, max_length))
question = np.random.randint(1, question_vovabulary_size, 
    size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, 
    size=(num_samples, max_length))
# 回答是one-hot编码，需要使用Keras内置函数进行转化
answer = keras.utils.to_categorical(answer, answer_vocabulary_size)
# 使用输入组成的列表来拟合
model.fit([text, question], answer, epochs=10, batch_size=128)
# 使用输入组成的字典来拟合，需要对输入进行命名后才能使用
model.fit({'text': text, 'question': question}, answer, 
    epochs=10, batch_size=128)

#%% 用函数式API实现一个三输出模型
from keras.models import Model
from keras import Input, layers
vocabulary_size = 50000
num_income_groups = 10
posts_input = Input(shape=(None, ), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPool1D(5)(x)
x = layers.Dense(128, activation='relu')(x)
# 注意：输出层均定义了名称
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, 
    activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', 
    name='gender')(x)
model = Model(posts_input, [age_prediction, income_prediction, 
    gender_prediction])
model.summary()
# 使用多种方式编译模型
model.compile(optimizer='adam', loss=['mse', 'categorical_crossentropy', 
    'binary_crossentropy'])
# 与上述写法等效，但仅输出层有名称时才能采用的写法
model.compile(optimizer='adam', loss={
    'age': 'mse', 
    'income': 'categorical_crossentropy', 
    'gender': 'binary_crossentropy'
})
# 多输出模型的编译选项：损失加权
model.compile(optimizer='adam', loss=['mse', 'categorical_crossentropy', 
    'binary_crossentropy'], loss_weights=[0.25, 1.0, 10.0])
# 与上述写法等效的只能在输出层具有名称时才能采用的写法
model.compile(optimizer='adam', loss={
    'age': 'mse', 
    'income': 'categorical_crossentropy', 
    'gender': 'binary_crossentropy'
}, loss_weights={
    'age': 0.25, 
    'income': 1.0, 
    'gender': 10.0
})
# 将数据输入到多输出模型中，假设输入为posts数组，输出为age_targets、
# income_targets、gender_targets数组
posts, age_targets, income_targets, gender_targets = [], [], [], []
model.fit(posts, [age_targets, income_targets, gender_targets], 
    epochs=10, batch_size=64)
# 与上述写法等效，只有输出层具有名称时才能采用
model.fit(posts, {
    'age': age_targets, 
    'income': income_targets, 
    'gender': gender_targets
}, epochs=10, batch_size=64)

#%% 使用Keras构建Inception模块，假设模块的输入为x
from keras import layers
# 分支a，strides表示步幅
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
# 分支b，具有两层卷积
branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
# 分支c，3x3的平均池化层用到了步幅
branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
# 分支d
branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
# 将分支输出连接在一起，得到模块的输出
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], 
    axis=-1)
# 注：完整的Inception V3位置在keras.applications.inception_v3.InceptionV3
# 注：Xception(extreme inception)也在keras.applications中

#%% 若残差尺寸相同，使用恒等残差连接(identity residual connnection)
from keras import layers
x = []
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.add([y, x]) # 将原始x与输出特征相加

#%% 若特征图尺寸不同，使用线性残差连接(linear residual connection)
from keras import layers
x = []
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPool2D(2, strides=2)(y)
# 使用1x1卷积。将原始x张量线性下采样为与y具有相同的形状
residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
# 将残差张量与输出特征相加
y = layers.add([y, residual])

#%% 连体LSTM(Simese LSTM)或共享LSTM(shared LSTM)，即层共享
from keras.models import Model
from keras import Input, layers
lstm = layers.LSTM(32) #将一个LSTM层实例化
# 构建模型的左分支：输入是长度128的向量组成的变长序列
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)
# 构建模型的右分支，如果调用已有的层示例，那么就会重复使用权重
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)
merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)
# 将模型实例化训练， 训练时基于两个输入对LSTM层的权重进行更新
model = Model([left_input, right_input], predictions)
left_data, right_data, targets = [], [], []
model.fit([left_data, right_data], targets)

#%% 在Keras中实现连体视觉模型(共享卷积基)
from keras import Input, layers, applications
# 图像处理基础模型采用Xception网络(只包括卷积基)
xception_base = applications.Xception(weights=None, 
    include_top=False)
# 输入为250x250的RGB图像
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))
# 对相同的视觉模型调用两次
left_features = xception_base(left_input)
right_features = xception_base(right_input)
# 合并后的特征包含来自左右两个视觉输入中的信息
merged_features = layers.concatenate([left_features, 
    right_features], axis=-1)
