# 针对大规模的机器学习，使用基于所有样本的累积误差更新权重进行梯度下降的自适应线性学习，计算成本很大。
# 在这里，每次使用一个样本渐进地更新权重，这样更快收敛，也适用于在线学习
# 1.更改fit方法使用单个训练样本来更新权重
# 2.新增partial_fit方法，对于在线学习，此方法不会重置权重
# 3.我们通过_shuffle方法打乱训练数据
from numpy.random import seed
import numpy as np


class AdalineSGD(object):
    """
    自适应线性神经元随机梯度下降学习分类器
    参数：
    eta:float
    基于0到1之间的学习速率
    n_iter:int
    训练数据集的迭代次数
    属性：
    w_:ld-array
    每次学习之后权重数组
    errors:list
    每次迭代的错误分类数目的列表
    shuffle:bool (default:True)
    是否打乱数据，设置为True即打乱数据防止陷入循环
    random_state:int(default:None)
    为打乱数据设置随机状态，初始化权重
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False  # 默认未初始化权重
        self.shuffle = shuffle  # 默认需要打乱数据
        if random_state:
            seed(random_state)  # 指定随机数种子

    def fit(self, X, y):
        """
        训练数据集学习
        :param X: 二维数组，行数：样本数量，列数：2，两个特征，形成所有样本的特征矩阵
        :param y: 类标向量
        :return:self:object
        """
        # 初始化权重
        self._initialize_weights(X.shape[1])
        # 代价函数即均方根误差列表
        self.cost_ = []
        for i in range(self.n_iter):
            # 在每次迭代之前判断是否需要打乱输入数据，需要就调用函数打乱
            if self.shuffle:
                X, y = self._shuffle(X, y)
            # 此次迭代的代价
            cost = []
            # 计算每个样本更新权重
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            # 一次迭代过程中所有样本训练后将平均代价值作为此次迭代的代价值
            avg_cost = sum(cost) / len(y)
            self.cost_.append(cost)
        return self

    def partial_fit(self, X, y):
        """
        对于在线学习，该方法不会重置权值
        :param X:
        :param y:
        :return:
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
            # raval()，返回连续扁平数组，输入和输出都是array_like。numpy.ravel(a, order='C')
            # >>> x = np.array([[1, 2, 3], [4, 5, 6]])
            # >>> print np.ravel(x)
            # [1 2 3 4 5 6]
            # <span style="font-size:14px;">>>> a = np.array([[2,2,2],[3,3,3]])
            # >>> print a
            # [[ 2  2  2 ]
            #   [ 3  3  3 ]]
            # >>> a.shape
            # (2, 3)</span>
            #shape[0]返回行数
        #如果样本数大于1，则对每个样本进行渐进更新权值，否则只对该样本进行一次更新权值（即具体看每次在线学习时新样本的数量）
        if y.ravel().shape[0] > 1:
            for xi,target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self

    def _initialize_weights(self, m):

        """
        初始化m个权重都为0
        :param m:
        :return:
        """
        self.w_ = np.zeros(m + 1)
        self.w_initialized = True

    def _shuffle(self, X, y):
        """
        打乱训练数据
        :param X:
        :param y:
        :return:
        """
        r = np.random.permutation(len(y))  # 将0-len(y)之前的数全排列
        return X[r], y[r]

    def _update_weights(self, xi, target):
        """
        根据某个样本的误差来更新权重
        :param xi:
        :param target:
        :return: 返回单个样本训练后的代价函数值，而且更新权重
        """
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

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
        在学习之后返回样本类标
        :param X:
        :return:
        """
        return np.where(self.activation(X) >= 0.0, 1, -1)
