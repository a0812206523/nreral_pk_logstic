#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:44:31 2017

@author: luogan
"""

import pandas as pd

inputfile1='horseColicTraining.xlsx' 
input1= pd.read_excel(inputfile1).values


data_train=input1
'''
data_max = data_train.max()
data_min = data_train.min()


# data_std = data_train.std()
#data_train1 = (data_train-data_min)/(data_max-data_min) #数据标准化

'''
train_input=data_train[:,0:21]
train_label=data_train[:,21:22]

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

model = Sequential() #建立模型
model.add(Dense(input_dim = 21, output_dim = 48)) #添加输入层、隐藏层的连接
model.add(Activation('tanh')) #以Relu函数为激活函数

model.add(Dense(input_dim = 48, output_dim = 48)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dropout(0.2))

model.add(Dense(input_dim = 48, output_dim = 36)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dropout(0.2))
model.add(Dense(input_dim = 36, output_dim = 36)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数

model.add(Dense(input_dim = 36, output_dim = 12)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(input_dim = 36, output_dim = 36)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数


model.add(Dense(input_dim = 36, output_dim = 1)) #添加隐藏层、输出层的连接
model.add(Activation('sigmoid')) #以sigmoid函数为激活函数
#编译模型，损失函数为binary_crossentropy，用adam法求解
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_input, train_label, nb_epoch = 1000, batch_size = 20) #训练模型


inputfile2='horseColicTest.xlsx' 
input2= pd.read_excel(inputfile1).values



data_test=input2
'''
test_max = data_test.max()
test_min = data_test.min()


# data_std = data_train.std()
test_train = (data_test-test_min)/(test_max-test_min) #数据标准化

'''
test_input=data_test[:,0:21]
test_label=data_test[:,21:22]


r = pd.DataFrame(model.predict_classes(test_input))


print('/n')         
from sklearn.metrics import accuracy_score
print(accuracy_score(test_label, r))  



