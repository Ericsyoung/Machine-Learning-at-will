import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
warnings.filterwarnings('ignore')

#灰度图为28*28,读取为DataFrame格式
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
labels = train_data["label"]
train = train_data.drop(['label'], axis = 1)#drop函数默认删除行，列需要加axis = 1

#按行读取数据，整理为28*28,归一化加快学习速度
train = train.values.reshape(-1,28,28,1)
test = test_data.values.reshape(-1,28,28,1)
train = train/255.0
test = test/255.0

#对标签进行one hot编码，划分交叉验证数据集
labels = keras.utils.to_categorical(labels,10) 

x_train,x_val, y_train, y_val = train_test_split(train,labels,test_size=0.3)

#CNN网络
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
#dropout用于正则化项，随机丢失一些节点，防止网络过拟合
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation = "relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation = "softmax"))

model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

#saver = tf.train.Saver()

history = model.fit(x_train, y_train, batch_size = 86, epochs = 14, 
          validation_data = (x_val, y_val), verbose = 2)

# ax1 = plt.subplot(2,1,1)
# ax1.plot(history.history['acc'], color = b, label = 'traing accuracy')
# ax1.plot(history.history['val_acc'], color = r, label = 'validation accuracy')
# ax2 = plt.subplot(2,1,2)
# ax2.plot(history.history['loss'], color = b, label = 'training loss')
# ax2.plot(history.history['val_loss'], color = r, label = 'validation loss')

# # predict results
results = model.predict(test)
# # select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)