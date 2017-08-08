import numpy as np


# 使用Python实现自适应线性神经元
class AdalineGD(object):
    """
      自适应线性神经元分类器

      参数：
      eta:float
      0到1之间的学习速率
      n_iter:int
      训练集上的迭代次数

      属性：
      w_：id-array
      学习后的权重
      errors:list
      每次迭代中错误分类数组成的列表
    """

    def __init__(self, eta=0.01, n_iter=50):
        """
        对象的构造器，self指代对象本身
        :param eta: 默认参数学习速率
        :param n_iter: 默认参数迭代次数
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        训练集的学习过程
        :param X: 二维数组，行数：样本个数，列数：选取的特征数，训练向量
        :param y: 一维向量，行数：样本个数，代表每个样本学习之后的输出值
        :return: self:object
        """
        # w_初始化为样本个数+1个0，+1原因是x0
        self.w_ = np.zeros(X.shape[1] + 1)
        # 设置一个列表来存储代价函数J(w)的输出值以检查本轮训练后算法是否收敛
        self.cost_ = []

        # 批量更新权值
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """
        计算净输入
        :param X:
        :return:
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """
        计算线性激励函数
        :param X:
        :return:
        """
        return self.net_input(X)

    def predict(self, X):
        """
        训练收敛后返回类标
        :param X:
        :return:
        """
        return np.where(self.activation(X) >= 0.0, 1, -1)


