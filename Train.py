#-*- coding: UTF-8 -*-
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.optim import Adam, SGD
from Data.DataReader import Dataset
from Model import TextClassificationModel
import os

dataset = Dataset(boot_strap_samples=10000)
# 利用Dataset创建实例dataset(随机有放回地取10000个训练样本)
model = TextClassificationModel().cuda()
# 利用TextClassificationModel类创建model实例模型
optimizer = Adam(model.parameters(), 1e-4)
# 利用Adam模块创建优化器optimizer（会为我们更新模型的权重）
# model.parameters()为模型的参数,1e-4代表学习率

def to_tensor(x):
    """ 定义张量转化函数 （限制x类型为numpy数组）"""
    return torch.from_numpy(x).cuda()
    #返回 将numpy数组转为tensor的x变量

def train(n_rounds, batch_size):
    """ 定义训练函数
        :param n_rounds: 整数型，代表迭代次数
        :param batch_size: 整数型，代表batch大小
    """
    loss_func = nn.CrossEntropyLoss()
    # 损失函数为交叉熵损失函数

    def test():
        """ 定义测试函数 """
        test_xs, test_ys = dataset.get_test_samples()
        # 测试集的x特征，y标签由dataset的get_test_samples方法生成
        pred_ys = []
        # 初始化模型预测pred_ys列表
        test_batch_size = batch_size
        # 测试的batch_size为输入的batch_size
        for j in range(int(len(test_ys) / test_batch_size) + 1):
            # 遍历每个batch
            if j * test_batch_size == len(test_ys):
                # 如果到了最后一个batch
                break
                # 则跳出循环
            pred_batch_ys = model(test_xs[j * test_batch_size: (j + 1) * test_batch_size])
            # batch标签的预测集为 本组batch每个x的模型预测输出
            pred_ys.append(np.argmax(pred_batch_ys.detach().cpu().numpy(), axis=1))
            # 标签总预测集 添加，
            # argmax返回数组最大数的索引.参数axis是1,表示第1维的最大值。
        pred_ys = np.concatenate(pred_ys, axis=0)
        # 形成y预测集
        print(confusion_matrix(test_ys, pred_ys))
        # 输出混淆矩阵

    optimizer.zero_grad()
    # 把梯度置零，把loss关于weight的导数变成0
    for i in range(n_rounds):
        # forward（前向传递train-predict) + backward(反向传递求loss) + optimize(优化调整参数）
        # 循环梯度下降
        if i % 100 == 0:
            # 循环次数是100次的整数倍
            test()
            # 执行测试函数
        train_xs, train_ys = dataset.get_train_samples(batch_size)
        # 获取训练集的x,y
        pred_ys = model(train_xs)
        # 前向传递得到预测集y
        loss = loss_func(pred_ys, to_tensor(train_ys).long())
        # 计算损失
        loss.backward()
        # 反向传递求梯度
        if i % 3 == 1:
        # 如果 i 除 3 余1：
            optimizer.step()
            # 更新所有参数
            optimizer.zero_grad()
            # 梯度初始化为零
        print(f"{i}: {loss.item():.4f}")
        # f “{ 表达式}”是用于格式化输出的，即把i 、 loss.item():.4f 的值打印出来 （相当于format）


    torch.save(model.linear.state_dict(), "model.pth")
    # 将训练好的模型model的参数、框架 都保存到 路径"model.pth"

def output(file_name: str):
    """定义输出函数（file_name输出文件名限制为str）"""
    #df = pd.read_csv(file_name, encoding='gb18030')
    df = pd.read_excel(file_name)
    # 读取数据集
    batch_size = 256
    # 设置batch大小为256
    for i in range(int(len(df) / batch_size) + 1):
        # 遍历每个batch
        print(f"Start compute batch {i}")
        # 打印 “开始计算第i个batch”
        xs = df.iloc[i * batch_size: (i + 1) * batch_size]['content'].tolist()
        # x数据集 = df数据集中 第i个batch中每一条的’content‘字段文字（ 即文字信息）
        if len(xs) == 0:
            # 如果x为空（说明读完了）
            break
            # 跳出循环，读下一个
        ys = model(xs).detach().cpu().numpy()
        # ys 为 model 对xs数据集的预测输出（概率）
        # detach(): 返回一个新的Tensor，但返回的结果是没有梯度的
        # cpu(): 把gpu上的数据转到cpu上
        # numpy(): 将tensor格式转为numpy
        ys = np.argmax(ys, axis=-1).tolist()
        # axis=-1和axis=1返回结果一样(是在行中比较，选出最大的列索引，即预测概率最大的对应等级)
        df.loc[i * batch_size: (i + 1) * batch_size - 1,'label'] = ys
        # 在df的对应索引行，’label‘列，输入对应的模型预测标签
    df.to_excel("Result/data_predict_batch_size(1000,256).xlsx")
    # 存储为 csv 文件

if __name__ == '__main__':
    # 如果是在本文件下执行：
    train(1000, 256)
    # 在迭代次数为1000，batch大小为256的情况下，进行训练
    output("Data/data_all_0825(pro).xlsx")
    # 输出预测模型


