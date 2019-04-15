#%% 启动Keras和TensorFlow
import keras
keras.__version__

#%% 单词级的one-hot编码示例
import numpy as np
# 示例列表，列表元素可为单词、句子和文章
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
token_index = {} #构建数据中所有标记的索引字典
for sample in samples: #遍历列表
    for word in sample.split(): #以空格分离句子得到单词列表
        # 为每个唯一单词指定唯一索引，没有0索引
        if word not in token_index:
            token_index[word] = len(token_index) + 1
max_length = 10 #设置只考虑样本的前max_length个单词
results = np.zeros((len(samples), max_length, 
    max(token_index.values()) + 1)) #构造空的结果向量
# 通过循环变量样本前max_langth个单词构造编码索引字典
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.0

#%% 字符级的one-hot编码示例
import string
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable  #获取所有可打印的ASCII字符列表
# 根据ASCII字符列表构造字符索引字典
token_index = dict(zip(characters, range(1, len(characters) + 1)))
max_length = 50 #设置只考虑前max_length个字符
results = np.zeros((len(samples), max_length, 
    max(token_index.values()) + 1)) #构造空的结果向量
# 通过循环变量样本前max_langth个字符构造编码索引字典
for i, sample in enumerate(samples):
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1.0

#%% 使用Keras实现单词级的one-hot编码示例
from keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words=1000) #创建只考虑1000个单词的分词器
tokenizer.fit_on_texts(samples) #使用示例列表构建单词索引
# 利用分词器将字符串(列表)转化为整数索引组成的列表
sequences = tokenizer.texts_to_sequences(samples)
# 也可直接得到one-hot的二进制表示，此外还有其他向量化模式
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index #找回单词索引
print('Found %s unique tokens.' % len(word_index))

#%% 使用散列技巧的单词级的one-hot编码示例
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# 将单词保存为长度为1000的向量，若单词数量接近1000(或更多)则会导致散列冲突
dimensionality = 1000
max_length = 10 #只考虑前10个单词
results = np.zeros((len(samples), max_length, dimensionality)) #结果向量
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        # 将单词散列为0~1000的一个随机整数索引
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.0

#%% 加载IMDB数据集，准备用于Embedding层
from keras.datasets import imdb
from keras import preprocessing
max_features = 10000 #最大词索引
maxlen = 20 #只考虑前20个词，后面的词截断
(x_train, y_train), (x_test, y_test) = \
        imdb.load_data(num_words=max_features) #读取IMDB数据，加载为整数列表
# 将整数列表转化成形状为(samples, maxlen)的二维整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

#%% 在IMDB数据集上使用Embedding层和分类器构建模型并训练
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
model = Sequential()
# 实例化Embedding层，其中1000指最大单词索引加1，8指嵌入的维度
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten()) #将三维嵌入张量展平为(samples, maxlen * 8)的二维张量
model.add(Dense(1, activation='sigmoid')) #输出层
model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
        metrics=['acc'])
model.summary() #输出模型概况
history = model.fit(x_train, y_train, epochs=10, batch_size=32,
        validation_split=0.2) #最后一个参数指定验证集的比例

#%% 读取IMDB原始文本数据保存到内存中(原创！！)
texts = [] #评论文本列表
labels = [] #标签列表
with open('D:/project/machine_learning_databases/IMDb_movie_data.csv', 
        '+r', encoding='utf-8') as f:
        firstline = f.readline() #读取首行并排除
        for line in f:
                text = f.readline().split(',')
                texts.append(text[0])
                labels.append(text[-1])

#%% 对IMDB原始文本进行分词
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
maxlen = 100  #在100个单词后截断评论
training_samples = 200  #训练样本个数
validation_samples = 10000  #验证样本个数
max_words = 10000  #只考虑数据集前10000个最常见单词
tokenizer = Tokenizer(num_words=max_words) #定义分词器
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
# 将数据划分为训练集和验证集
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

#%% 解析GloVe词嵌入文件并将GloVe词嵌入矩阵
import os
glove_dir = 'D:/project/machine_learning_databases' #文件目录
embeddings_index = {} #Embedding层
with open(os.path.join(glove_dir, 'glove.6B.100d.txt'), 
        encoding='utf-8') as f:
        for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))
embedding_dim = 100 #
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # 嵌入索引(embeddings_index)中找不到的词嵌入向量全为零
            embedding_matrix[i] = embedding_vector

#%% 定义模型并将预训练的词嵌入加载到Embedding层中
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.layers[0].set_weights([embedding_matrix]) #加载初始层参数
model.layers[0].trainable = False #冻结初始层

#%% 训练模型与评估模型
import matplotlib.pyplot as plt
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
        metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32,
        validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5') #模型保存
# 绘制模型结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
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

#%% 在不使用预训练词嵌入的基础上训练相同模型
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
        metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32,
        validation_data=(x_val, y_val))
# 绘制模型结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
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

#%% 选取测试数据，加载并评估第一个模型
x_test = data[-validation_samples: ]
y_test = labels[-validation_samples: ]
model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)