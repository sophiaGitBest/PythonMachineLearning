#使用Python实现（罗森布拉特感知器）感知器学习算法
#定义感知器的接口，使得我们可以初始化新的感知器对象
#fit方法从数据中学习
#predict对新的样本进行测试

import numpy as np
class Perceptron(object):
    #以下是多行注释
    """ Perceptron Classifier

    Parameters
    -----------
    eta:float
        学习率（0到1之间）
    n_iter:int
        训练数据集上迭代次数

    Attributes
    -----------
    w_:ld-array
        学习后的权重
    errors_:list
        每次迭代后错误分类数
    """
    def _init_(self,eta=0.01,n_iter=10):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self,X,y):
        """
        学习训练数据
        :param X:  {array-like},shape=[n_samples,n_features]训练向量，n_samples代表样本数量，n_features代表每个样本的特征数量
        :param y: array-like，shape=[n_samples] 目标值
        :return: self：object
        """
        self.w_=np.zeros(1+X.shape[1])  #权值初始化为0
        self.errors_=[]   #每次迭代错误分类数的列表

        for _ in range(self.n_iter):
            errors=0
            for xi,target in zip(X,y):
                update=self.eta*(target-self.predict(xi))    #计算得到的更新值
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors+=int(update!=0.0)
            self.errors_.append(errors)
        return self


    def net_input(self,X):
        """
        计算净输入:w0x0+....wmxm
        x0=1
        :param X: 一个数组
        :return: 数值
        """
        return np.dot(X,self.w_[1:])+self.w_[0]

    def predict(self,X):
        """
        对新数据进行类标预测
        :param X: 新样例
        :return: 1或者-1
        """
        return np.where(self.net_input(X)>=0.0,1,-1)