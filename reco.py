import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

warnings.filterwarnings('ignore')

#from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
#from keras.models import Sequential

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#check the data
#train_data.isnull().any().describe()
#test_data.isnull().any().describe()

label = train_data.values[:,0]
train_data = train_data.values[:,1:]

test_data = test_data.values[1:,:]

#重新整理数据，归一化，加快cnn运行速度
train_data = train_data/255.0
test_data = test_data/255.0

#reshape，最后一个1表示是灰度图像。rgb图像则是3，该维度用于keras，过滤器的深度必须与输入内容的深度相同
train_data = train_data.reshape(-1,28,28,1)
test_data = test_data.reshape(-1,28,28,1)

#plt.imshow(train_data[5][:,:,0])
#plt.show()

#构建交叉验证数据集
x_train,x_test,y_train,y_test = train_test_split(train_data,label,test_size = 0.3)

#构建卷积神经网路
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters = 28, kernel_size = (5,5), padding = 'same',activation ='relu', input_shape = (28,28,1)))
model.add(tf.keras.layers.Conv2D(filters = 28, kernel_size = (5,5), padding = 'same',activation ='relu', input_shape = (28,28,1)))
#model.add(tf.keras.layers.MaxPool2D(pool_size = ))