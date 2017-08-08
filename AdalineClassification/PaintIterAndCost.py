# 实践中，我们往往需要通过实验来找到合适的学习速率，这里我们分别使用0.01和0.0001的学习速率来绘制迭代次数与代价函数之间的关系图像，以观察Adaline通过训练数据进行学习的效果
#-*- coding: utf-8 -*-
import AdalineClassification.Adaline as Adaline
import AdalineClassification.AdalineSGD as AdalineSGD
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import numpy as np
import PerceptronClassfication.DecisionRegions as decideRegion

# 读取数据
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)  # 读数据集
y = df.iloc[0:100, 4].values  # 取前100个训练样本的第5个列下标的值组成数组
y = np.where(y == 'Iris-setosa', -1, 1)  # 标准类标为1，-1
X = df.iloc[0:100, [0, 2]].values  # 取前100个训练样本的第1,3个列下标的值组成二维数组

# 一行两列的展示两个子图
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# 构造第一个学习速率下的自适应线性神经元
ada1 = Adaline.AdalineGD(eta=0.01, n_iter=10).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('迭代次数',fontproperties=font)
ax[0].set_ylabel('log(均方根误差)',fontproperties=font)
ax[0].set_title('Adaline-学习速率为0.01',fontproperties=font)
ada2 = Adaline.AdalineGD(eta=0.0001, n_iter=10).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('迭代次数',fontproperties=font)
ax[1].set_ylabel('均方根误差',fontproperties=font)
ax[1].set_title('Adaline-学习速率为0.0001',fontproperties=font)
plt.show()

#从上述的图中可以看出我们面临两种不同类型的问题
#1.学习速率过大0.01时-并没有使代价函数的值尽可能的小，反而因为算法跳过了全局最优解，导致误差随着迭代次数的增加而增加
#2.虽然第2张图中代价函数是逐渐减小的，但是学习速率0.0001太小了，为了达到算法收敛的目标，需要迭代更多的次数

#梯度下降就是通过特征缩放而受益的众多算法之一，在这里使用一种标准化的特征缩放的方法，此方法可以使得数据具备标准正态分布的特性
#下面我们再看看通过对数据的2个特征进行标准化的特征缩放之后，代价函数是否随着迭代次数收敛还有画出此时的判定区域
X_std=np.copy(X)
X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()
#标准化后构造Adaline
ada3=Adaline.AdalineGD(eta=0.01,n_iter=15)
#学习
ada3.fit(X_std,y)
#画出决策区域
decideRegion.plot_decision_regions(X_std,y,classifier=ada3)
plt.title('Adaline-梯度下降',fontproperties=font)
plt.xlabel('经标准化处理的萼片长度',fontproperties=font)
plt.ylabel('经标准化处理的花瓣长度',fontproperties=font)
plt.legend(loc='upper left')
plt.show()

#画出代价函数（均方根误差）随迭代次数的变化趋势
plt.plot(range(1,len(ada3.cost_)+1),ada3.cost_,marker='o')
plt.xlabel('迭代次数',fontproperties=font)
plt.ylabel('均方根误差',fontproperties=font)
plt.show()

#我们使用fit方法训练AdalineSGD分类器，并应用plot_decision_regions绘制训练结果
#同时使用的也是经标准化的训练数据
ada4=AdalineSGD.AdalineSGD(eta=0.01,n_iter=15,random_state=1)
ada4.fit(X,y)
decideRegion.plot_decision_regions(X_std,y,classifier=ada4)
plt.title('AdalineSGD-随机梯度下降',fontproperties=font)
plt.xlabel('经标准化处理的萼片长度',fontproperties=font)
plt.ylabel('经标准化处理的花瓣长度',fontproperties=font)
plt.legend(loc='upper left')
plt.show()
#绘制代价函数随迭代次数变化趋势
plt.plot(range(1,len(ada4.cost_)+1),ada4.cost_,marker='o')
plt.xlabel('迭代次数',fontproperties=font)
plt.ylabel('平均代价',fontproperties=font)
plt.show()
