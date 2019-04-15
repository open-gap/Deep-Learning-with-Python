#%% 启动Keras和TensorFlow
import keras
keras.__version__

#%% 下载并解析原始文本数据
import keras
import numpy as np
path = keras.utils.get_file(
    'nietzsche.txt', origin=\
    'https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with open(path) as f:
    text = f.read().lower()
print('Corpus length:', len(text))

#%% 将字符序列向量化
maxlen = 60 #提取60个字符组成的序列
step = 3 #每3个字符采样一个新序列
sentences = [] #保存所提取的序列
next_chars = [] #保存目标(即下一个字符)
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))
chars = sorted(list(set(text))) #利用set(集合)得到语料中唯一字符组成的列表
print('Unique characters:', len(chars))
# 一个字典，将唯一的字符映射为它在列表中索引值
char_indices = dict((char, chars.index(char)) for char in chars)
print('Vectorization...')
# 将字符one-hot编码为二进制数组
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1.0
    y[i, char_indices[next_chars[i]]] = 1.0

#%% 构建用于预测下一个字符的单层LSTM模型并编译
from keras import layers
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
# 编译模型
optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

#%% 给定模型预测，采样下一个字符的函数，即采样函数(sampling function)
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # 通过输入的preds列表概率重新生成概率分布列表
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas) #取得重分布概率矩阵中最大概率元素的序号

#%% 文本生成循环
import sys
import random
for epoch in range(60): #将模型训练60轮
    print('epoch', epoch + 1)
    model.fit(x, y, batch_size=128, epochs=1) #将数据在模型上拟合一次
    # 随机选择文本种子
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')
    # 尝试不同的采样温度值
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)
        # 从种子文本开始生成400个字符
        for i in range(400):
            # 对目前生成的字符进行one-hot编码
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.0
            # 对下一个字符进行采样
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            # 将采样的下一个字符添加到原文本中，构建新的文本
            generated_text += next_char
            generated_text = generated_text[1:]
            # 将下一个字符写入到输出缓冲中，循环结束后利用print语句输出
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()