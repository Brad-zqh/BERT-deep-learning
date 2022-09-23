#-*- coding: UTF-8 -*-
import numpy as np
import pandas
import pandas as pd

class Dataset:
    """定义Dataset类（字母开头大写）,用于准备数据集"""
    def __init__(self, train_ratio: float=0.8, boot_strap_samples: int=None):
        """ __init__()特殊方法，当基于类新建实例，会自动运行此方法

                    :param: self: init必要参数，当基于类创建实例，self代表实例本身（类被调用对象），
                          类中所有方法自动传递self,让实例与类的属性、方法取得链接
                    :param: train_ratio: 训练集所占比率 （创建实例时需要传递的初始化参数）
                    :param: boot_strap_samples: 有放回的重复抽样方法（用于解决数据不足）
                """

        self.df = pd.read_excel("Data/label_0825.xlsx") # 读取csv文件，以pd形式存储于df属性
        """任何以self前缀的变量，在类中的所有方法都可用,可以在任何基于类创建的实例中访问这些变量，也被称为属性"""

        self.label_counts = [0, 0, 0]
        """ list列表与array数组区别：
                    list中的元素的数据类型可以不一样。array里的元素的数据类型必须一样；
                    list不可以进行数学四则运算，array可以进行数学四则运算；
                    相对于array，list会使用更多的存储空间。
        """
        for label in range(3):
            # label in [0,1,2,3]循环
            self.label_counts[label] = np.sum(self.df['label'] == label)
            # label_count[label]赋值为：df['label']对应为label的评论数量总和，即标签统计值
        self.train_idx = int(train_ratio * len(self.df))
        # train_idx（训练集数量）= train_ratio(0.8)* df行数
        self.train_label_counts = [0, 0, 0]
        # 初始化用于训练的标签统计数组
        for label in range(3):
            self.train_label_counts[label] = np.sum(self.df['label'].iloc[:self.train_idx] == label)
            # 统计训练集的标签统计值   （df的0到self.train_idx行）

        self.test_label_counts = [0, 0, 0]
        # 初始化测试集标签统计值
        for label in range(3):
            self.test_label_counts[label] = np.sum(self.df['label'].iloc[self.train_idx:] == label)
        # 统计测试集的标签统计值  （df的self.train_idx到末行）
        self.train_df = self.df.iloc[:self.train_idx]
        # 定义训练集的dataframe
        self.test_df = self.df.iloc[self.train_idx:]
        # 定义测试集的dataframe

        self.boot_strap_samples = boot_strap_samples
        # 定义boot_strap_samples属性为输入的int=None
        if boot_strap_samples is not None:
            # 如果输入不是None
            sub_dfs = []
            # 初始化sub_dfs列表
            for label in range(3):
                sub_dfs.append(self.train_df[self.train_df['label'] == label].sample(boot_strap_samples, replace=True))
                """ self.train_df[self.train_df['label'] == label]即为训练集里面对应标签的数据集[0,1,2,3]
                                    序列.sample(boot_strap_samples)：
                                    从序列a中随机有放回（replace=true）抽取boot_strap_samples个元素，并将这些元素生以list形式返回
                """

            self.train_df = pandas.concat(sub_dfs)
            # 将sub_dfs融合起来（默认纵向合并dataframe对象）

    def get_train_samples(self, batch_size):
        """定义get_train_samples方法（得到训练样本）
                    :param: self:初始化参数（实例本身）
                    :param: batch_size: 单次训练所选取的样本数 （影响模型的优化程度和速度，并直接影响GPU内存的使用情况）
        """

        indices = np.random.choice(len(self.train_df), batch_size)
        # 在train_df中随机抽取batch_size个的索引值
        train_xs = self.train_df['content'].iloc[indices].values.tolist()
        # 定义训练集中的x特征，为['content']字段指定索引行的值，转为list()
        train_ys = self.train_df['label'].iloc[indices].values
        # 定义训练集中的y标签，为['label']字段指定索引行的值
        train_ys[np.isnan(train_ys)] = 2
        # 将train_ys中标签为空的替换为2
        return train_xs, train_ys
        # 返回训练集的x,y


    def get_test_samples(self):
        """定义一个获取测试集样本的方法"""
        return self.test_df['content'].values.tolist(),\
               self.test_df['label'].values
        # 测试集[微博内容]字段，以及label字段

if __name__ == '__main__':
    dataset = Dataset()

"""一个python文件通常有两种使用方法，
    一是作为脚本直接执行，
    二是 import 到其他的 python 脚本中被调用（模块重用）执行。
    if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，
    而 import 到其他脚本中（__name__为模块名称，并非main),是不会被执行的。
    链接：https://blog.csdn.net/heqiang525/article/details/89879056
"""