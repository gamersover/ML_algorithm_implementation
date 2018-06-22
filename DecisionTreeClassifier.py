#coding:utf-8
"""
决策树分类器算法实现
"""
import numpy as np
from collections import Counter


def getEntropy(data):
    """得到一个数据分布的熵

    # Arguments
        data: list, 一维数据分布

    # Returns
        entropy: decimal, 数据分布的熵
    """
    count_dict = Counter(data)
    N = data.shape[0]
    entropy = 0
    for k, v in count_dict.items():
        entropy += v/N * np.log2(v/N+10e-6)
    return -entropy

def getCondEntropy(feat, label):
    """得到类别关于一个特征的条件熵

    # Arguments
        feat: ndarray, [N,] 一个特征
        label: ndarray, [N,] 类别

    # Returns
        entropy: decimal, 条件熵
    """
    N = feat.shape[0]
    feat_count_dict = Counter(feat)
    entropy = 0
    for k, v in feat_count_dict.items():
        temp_arr = label[feat==k]
        entropy += v/N * getEntropy(temp_arr)
    return entropy

def getInfoGain(data, feat_used):
    """得到所有特征的信息增益 id3算法使用的评判标准
    
    # Arguments
        data: ndarray, 二维数据，最后一列为标签，前面的列为特征
        feat_used: boolean list, 特征是否被使用

    # Returns
        gains: list of list, 每个元素是特征的信息增益, 特征索引，特征可能取值
    """
    label = data[:, -1]
    feat_n = data.shape[1] - 1
    Hy = getEntropy(label)
    gains = []
    for i in range(feat_n):
        feat = data[:, i]
        k = np.sort(np.unique(feat)).tolist()
        # 特征没有使用并且特征取值个数大于1
        if not feat_used[i] and len(np.unique(feat)) > 1:
            Hy_a = getCondEntropy(feat, label)
            gains.append([Hy-Hy_a, i, k])
    return gains

def getInfoGainRatio(data, feat_used):
    """得到所有特征的信息增益比 c4.5算法使用的评判标准

    # Arguments
        data: ndarray, 数据集
        feat_used: boolean list

    # Returns
        gain_ratios: list of list, 每个元素是特征的信息增益比，特征索引，特征可能取值
    """
    gains = getInfoGain(data, feat_used)
    gain_ratios = []
    for i in range(len(gains)):
        gain_ratio = gains[i][0] / getEntropy(data[:, gains[i][1]])
        gain_ratios.append([gain_ratio, gains[i][1], gains[i][2]])
    return gain_ratios

def getGini(data):
    """得到某一分布的gini系数

    # Arguments
        data: ndarray, [N, ] 一个分布

    # Returns：
        param: decimal, gini系数
    """
    count_dict = Counter(data)
    N = data.shape[0]
    s = 0
    for k, v in count_dict.items():
        s += (v/N)**2
    return 1 - s

def getFeatGini(feat, label, ith):
    """得到某个特征的所有切分点的gini系数

    # Arguments
        feat: ndarray, 特征分布
        label: ndarray, 标签
        ith: int, 特征所在索引

    # Returns
        ginis: list of list, 每个元素包含gini系数，特征索引，特征取值
    """
    feat_count = Counter(feat)
    N = feat.shape[0]
    ginis = []
    for k, v in feat_count.items():
        n = feat[feat==k].shape[0]
        gini = n/N * getGini(label[feat==k]) + (N-n)/N * getGini(label[feat!=k])
        ginis.append([gini, ith, k])
    return ginis

def getAllGini(data):
    """得到数据集中所有未使用的特征的每个切分点的gini系数

    # Arguments
        data: ndarray, 数据集
        feat_used: boolean list

    # Returns
        all_ginis: list of list, 
    """
    feat_n = data.shape[1] - 1
    label = data[:, -1]
    all_ginis = []
    for i in range(feat_n):
        feat = data[:, i]
        if len(np.unique(feat)) > 1:
            all_ginis += getFeatGini(feat, label, i)
    return all_ginis

class Node(object):
    """决策树中节点定义

    # Attributes
        cut: list, 该节点的切割特征, 特征取值
        cls: int, 该节点的类别
        childs: Node list, 该节点的孩子节点
    """
    def __init__(self):
        self.cut = None
        self.cls = None
        self.childs = None


class DecisionTree(object):
    """决策树

    # Arguments
        criterion: "gini", "id3", "c4.5", 评判标准
    """
    def __init__(self, criterion="gini"):
        self.criterion = criterion

    def fit(self, x, y):
        """训练模型

        # Arguments
            x: ndarray, [N, M] 输入特征
            y: ndarray, [N, ] 输入标签
        """
        feat_used = [False for _ in range(x.shape[1])]
        self.x = x
        self.y = y
        self.data = np.hstack([x, y[:, np.newaxis]])
        self.root = Node()
        self.construct_tree(self.data, feat_used, self.root)

    def predict(self, x):
        """预测

        # Arguments
            x: ndarray, 输入特征

        # Returns
            pred: list, 模型预测结果
        """
        pred = []
        for i in range(x.shape[0]):
            pred.append(self._predict_one(x[i]))
        return pred

    def get_accuracy(self):
        """获取模型准确率
        """
        preds = self.predict(self.x)
        return np.mean(np.equal(preds, self.y).astype(int))

    def displayTree(self):
        """打印树的结构
        """
        self._showTree(self.root, 0)

    def construct_tree(self, arr, feat_used, root):
        """构建决策树

        # Arguments
            arr: ndarry
            feat_used: boolean list
            root: Node
        """
        cut, ret = self._split(arr, feat_used)
        # 可以切分
        if cut is not None:
            # 设置切割特征
            root.cut = cut
            n = len(ret)
            root.childs = [Node() for i in range(n)]
            for i in range(n):
                # 保证对每个子节点feat_used是一样的
                feat_used_copy = feat_used.copy()
                self.construct_tree(ret[i], feat_used_copy, root.childs[i])
        # 无法切分
        else:
            # 设置节点类别
            root.cls = ret

    def _getSplit(self, arr, cut):
        """获取根据某个特征的取值，切割后的数据集

        # Arguments
            arr: ndarray, 特征+标签数据
            cut: tuple, 确定哪个特征用来切割数据集

        # Returns:
            split_data: ndarray list, 切割后的数据集组成的列表
        """
        if self.criterion == "gini":
            split_data = self._getBinSplit(arr, cut[0], cut[1])
        else:
            feat = arr[:, cut[0]]
            split_data = []
            for v in cut[1]:
                temp_arr = arr[feat==v]
                split_data.append(temp_arr)
        return split_data

    def _getBinSplit(self, arr, ith, value):
        """二叉树划分数据集

        # Arguments
            arr: ndarray, 原始数据集
            ith: int, 划分的特征
            value: decimal, 划分的特征取值
        """
        feat = arr[:, ith]
        #  不等于特征取值的数据集划为左子节点，等于的数据集划为右子节点
        split_data = [arr[feat!=value], arr[feat==value]]
        return split_data

    def _get_standard(self, arr, feat_used):
        """获得特征评价标准

        # Arguments
            arr: ndarray, 数据集
            feat_used: boolean list

        # Returns
            standard: list of list, 评价标准
        """
        if self.criterion == "id3":
            standard = getInfoGain(arr, feat_used)

        elif self.criterion == "c4.5":
            standard = getInfoGainRatio(arr, feat_used)

        elif self.criterion == "gini":
            # 求解gini系数时，不需要feat_used，因为下一次划分该特征可以继续使用
            standard = getAllGini(arr)
        else:
            raise ValueError("criterion must in [gini, id3, c4.5]")

        return standard

    def _split(self, arr, feat_used):
        """对节点进行切割

        # Arguments
            arr: ndarray, 特征+标签数据
            feat_used: boolean list

        # Returns
            (param1, param2): 如果节点不能再分，则为None， 类别
                              如果可以再分，则为切点特征，切割后的数据集

        # Raise
            ValueError: criterion value
        """
        label_count = Counter(arr[:, -1])
        standard = self._get_standard(arr, feat_used)
        # 评价标准为空或者类别只有一个，表示不可再分
        if len(standard) == 0 or len(label_count) == 1:
            return (None, label_count.most_common(1)[0][0])

        cut = tuple(max(standard)[1:])
        feat_used[cut[0]] = True
        return (cut, self._getSplit(arr, cut))

    def _showTree(self, root, layer):
        """显示决策树结构

        # Arguments
            root: Node, 决策树根节点
            layer: int, 决策树的层
        """
        print("layer {}, cut {}, cls {}".format(layer, root.cut, root.cls))
        if root.childs is not None:
            for child in root.childs:
                self._showTree(child, layer+1)

    def _predict_one(self, x):
        """预测一个样本

        # Argument
            x: ndarray, 输入特征

        # Returns
            param: int, 输出类别
        """
        node = self.root
        while node.cut is not None:
            if self.criterion == "gini":
                # 找到特征取值等于切分点的子节点
                child_idx = int(x[node.cut[0]]==node.cut[1])
            else:
                # 找到特征取值等于切分点的子节点
                child_idx = node.cut[1].index(x[node.cut[0]])
            node = node.childs[child_idx]
        return node.cls


if __name__ == "__main__":
    np.random.seed(46)
    feat1 = np.random.choice(3,20) + 2
    feat2 = np.random.choice(2,20)
    feat3 = np.random.choice(2,20) + 1
    feat4 = np.random.choice(3,20)
    feat5 = np.random.choice(3,20) + 4
    feat6 = np.random.choice([4,7], 20)
    label = np.random.choice(3,20)
    arr = np.vstack([feat1, feat2, feat3, feat4, feat5, feat6, label]).T

    x = arr[:,:-1]
    y = arr[:, -1]
    dt = DecisionTree(criterion="gini")
    dt.fit(x, y)
    # print(dt.predict(x))
    # dt.displayTree()
    print(dt.get_accuracy())