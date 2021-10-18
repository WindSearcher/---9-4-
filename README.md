## 飞桨常规赛：中文新闻文本标题分类 9月第4名方案

### 一.赛题介绍：

本次比赛赛题为新闻标题文本分类 ，选手需要根据提供的新闻标题文本和类别标签训练一个新闻分类模型，然后对测试集的新闻标题文本进行分类，评价指标上使用Accuracy = 分类正确数量 / 需要分类总数量。同时本次参赛选手需使用飞桨框架和飞桨文本领域核心开发库PaddleNLP，PaddleNLP具备简洁易用的文本领域全流程API、多场景的应用示例、非常丰富的预训练模型，深度适配飞桨框架2.x版本。

比赛传送门：https://aistudio.baidu.com/aistudio/competition/detail/107/0/introduction
### 二.项目简介：

飞桨常规赛：中文新闻文本标题分类9月第4名方案，分数0.89+，基于PaddleNLP通过预训练模型的微调完成新闻14分类模型的训练与优化，并利用训练好的模型对测试数据进行预测并生成提交结果文件。

### 三.AI Studio项目地址：
https://aistudio.baidu.com/aistudio/projectdetail/2399755

主要参考： https://aistudio.baidu.com/aistudio/projectdetail/2345384

### 四.运行说明：
直接运行baseline.iypnb文件即可

### 五.进一步优化方向：

1.可以针对训练数据进行数据增强从而增大训练数据量以提升模型泛化能力。[NLP Chinese Data Augmentation 一键中文数据增强工具](https://github.com/425776024/nlpcda/)，EDA

2.AEDA，21年八月份上的一篇关于数据增强的顶会，对文本随机插入标点符号，插入标点符号个数是1/3句子长度，需注意的是，AEDA只适用于文本分类任务，在其他任务中不适用。与EDA方法相比，AEDA会增加模型精度上和收敛速度。https://mp.weixin.qq.com/s/R6uDbn3CqxFkOye73Rqpqg   https://github.com/akkarimi/aeda_nlp

3.可以尝试使用不同的预训练模型如ERNIE和NEZHA等，并对多模型的结果进行投票融合。[竞赛上分Trick-结果融合](https://aistudio.baidu.com/aistudio/projectdetail/2315563)

4.可以发现模型存在过拟合现象，可以考虑加入正则化