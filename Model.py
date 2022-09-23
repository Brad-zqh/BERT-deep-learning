#-*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertTokenizer, BertModel

class TextClassificationModel(nn.Module):
    """定义文字分类器模型类 （继承父类nn.Module的所有属性、方法，并可以定义新的属性、方法） """
    def __init__(self, device='cuda:0'):
        # 初始化父类的属性参数
        super(TextClassificationModel, self).__init__()
        # super特殊函数，帮助连接父类superclass和子类subclass，唤醒父类__init__,并传递给子类所有父类的属性
        # super(TextClassificationModel, self) ==  super()
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # 分词器属性 = 从预训练模型chinese-roberta-wwm-ext（哈工大讯飞联合实验室）的BertTokenizer
        self.roberta_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # 模型属性 = 从预训练模型chinese-roberta-wwm-ext中的BertModel
        self.linear = nn.Linear(768, 4)
        """  PyTorch的nn.Linear（）用于设置网络中的全连接层，
                    在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量(tensor多维数组)，形状通常为[batch_size, size]，不同于卷积层要求输入输出是四维张量
                    nn.Linear(in_features, out_features)
                    in_features是输入的二维张量的大小，即输入[batch_size, size]中的size。
                    out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。
                    从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features=768]的张量变换成了[batch_size, out_features=4]的输出张量。
        """
        self.device = device
        # device设备属性设置为 device变量值（'cuda:0'），都是唯一的那一张GPU（指定GPU编号）

    def forward(self, x):
        """ 定义forward方法（子类的独特方法）
                    :param: self：指代实例本身
                    :param: x：输入的张量数据（特征）
        """
        encoded = self.tokenizer(x, padding=True,truncation=True, return_tensors="pt", max_length=32).to(self.device)
        """" 
                encode(编码）：将原始数据准备成模型需要的输入格式和内容
                (1)首先tokenizer()，利用cuda设备，将文本拆分为tokens(符号）（单词的一部分，标点符号，在中文里可能就是词或字，模型不同拆分算法不同）
                (2)然后tokenizer能够将tokens转换为数字，以便能够构建张量并输入到模型中; 
                (3)大多数预训练语言模型都需要额外tokens才能作为一次正常的输入（例如，BERT中的[CLS]），这些都会由tokenizer自动完成。

                在神经网络中，我们常常是通过一个batch（批）的形式来作为一次输入，这个时候你可能想要：
                    如果必要，将每个文本序列填充到最大的长度；
                    如果必要，将每个文本序列截断到模型可以接受的最大长度；
                    返回张量。

                :param: padding 用于填充（布尔值或字符串）
                        True或”longest“：填充到最长序列（如果你仅提供单个序列，则不会填充）
                        参数 “max_length”：用于指定你想要填充的最大长度，
                                       如果max_length=Flase，那么填充到模型能接受的最大长度
                                      （即使只输入单个序列，也会被填充到指定长度）
                        False或“do_not_pad”：不填充序列（默认行为）
                :param: return_tensors 表示返回数据的类型，
                        可选tf’, ‘pt’ or ‘np’ ，
                        分别表示tf.constant, torch.Tensor或np.ndarray
        """
        nsp_feature = self.roberta_model(**encoded)['pooler_output']
        """BERT两个预训练任务：
                 （1）MLN（Masked Language Model)
                    完形填空：随机mask句子中15%词，利用上下文预测 ：i am strong-> i am [Mask]
                 （2）NSP（Next Sentence Prediction)
                     预测两段文本的蕴含关系（分类任务）
                     句子对A、B，50%的B是A的下一条句子（正样本），另50%B是语料库随机选择（负样本），学习相关性
                     roberta删除了NSP任务，huggingface添加这个pooler output应该是为了方便下游的句子级别的文本分类任务。
                     pooler output是取[CLS]标记处对应的向量后面接个全连接再接tanh激活后的输出
        """

        output = self.linear(nsp_feature)
        # 设置输入为[batch_size, in_features=768]的张量nsp_feature变换成[batch_size, out_features=4]的输出张量

        return output
        # 返回output模型输出
