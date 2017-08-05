#基于鸢尾花数据集训练感知器模型
#鸢尾花数据集来源于UCI机器学习库中
#1.读取数据集
#2.提取前100个类标，分别对应50个山鸢尾花（Setosa，-1）和50个变色鸢尾花（versicolor,1)
#3.将整数类标（1，-1）赋给感知器对象的学习方法的y向量
#4.提取100个训练样本的第一个特征（萼片长度）和第3个特征（花瓣长度）赋给属性矩阵X
#5.利用matplotlib库用二维散点图进行数据可视化

#-*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)  #读数据集
y=df.iloc[0:100,4].values     #取前100个训练样本的第5个列下标的值组成数组
y=np.where(y=='Iris-setosa',-1,1)  #标准类标为1，-1
X=df.iloc[0:100,[0,2]].values    #取前100个训练样本的第1,3个列下标的值组成二维数组
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')    #画山鸢尾花数据集的散点图，前50个数据集。红色圆点，label
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')  #画变色鸢尾花数据集的散点图，50-100个数据集。蓝色叉叉，label
plt.xlabel('花瓣长度',fontproperties=font)
plt.ylabel('萼片长度',fontproperties=font)
plt.legend(loc='upper left')
#plt.show()

