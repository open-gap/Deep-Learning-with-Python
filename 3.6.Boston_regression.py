#%% 加载波士顿房价数据集
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = \
    boston_housing.load_data()
print('train_data.shape:', train_data.shape)
print('test_data.shape:', test_data.shape)
print('train_targets:', train_targets)

#%% 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
# 必须使用训练集的均值与方差，不能使用任何测试集数据
test_data -= mean
test_data /= std

#%% 构建网络
from keras import models, layers
def build_model(): #由于要多次将同一模型实例化，故采用函数构建模型
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', 
        input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) #输出层只有一个单元且无激活函数属于标量回归
    # 注：损失函数为均方误差(mean squared error)，训练过程中监控的为平均绝对误差
    # (mean absolute error)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

#%% 使用k折验证验证网络表现
import numpy as np 
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # 准备验证数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = \
        train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partical_train_data = np.concatenate(
        (train_data[: i * num_val_samples], 
            train_data[(i + 1) * num_val_samples: ]), axis=0
    )
    # 准备训练数据
    partical_train_targets = np.concatenate(
        (train_targets[: i * num_val_samples], 
            train_targets[(i + 1) * num_val_samples: ]), axis=0
    )
    model = build_model() #构建Keras模型
    # 训练模型(静默模式，设置verbose=0)
    model.fit(partical_train_data, partical_train_targets, epochs=num_epochs, 
        batch_size=1, verbose=0)
    # 在验证数据上评估模型
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
print('all_scores:', all_scores)
print('mean scores:', np.mean(all_scores))

#%% 增加训练轮次，并记录每折的验证结果
import matplotlib.pyplot as plt 
num_epochs = 500
all_mae_history = []
for i in range(k):
    print('processing fold #', i)
    # 准备验证数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = \
        train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partical_train_data = np.concatenate(
        [train_data[: i * num_val_samples], 
            train_data[(i + 1) * num_val_samples: ]], axis=0
    )
    # 准备训练数据
    partical_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples], 
            train_targets[(i + 1) * num_val_samples: ]], axis=0
    )
    model = build_model() #构建Keras模型
    # 训练模型(静默模式，设置verbose=0)
    history = model.fit(partical_train_data, partical_train_targets, 
        validation_data=(val_data, val_targets), epochs=num_epochs, 
        batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_history.append(mae_history)
# 计算所有轮次中的k折验证分数的平均值
average_mae_history = [
    np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)
]
# 绘制验证分数
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#%% 重新绘制验证分数(删除前10个数据点，取滑动平均值)
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
smoothed_mae_history = smooth_curve(average_mae_history[10: ])
plt.plot(range(1, len(smoothed_mae_history) + 1), smoothed_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#%% 根据验证分数训练最终模型
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print('Test mae score:', test_mae_score)