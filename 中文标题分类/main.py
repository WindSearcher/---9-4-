# 使用pandas读取数据集
import pandas as pd
train = pd.read_table('data/train.txt', sep='\t',header=None)  # 训练集
dev = pd.read_table('data/dev.txt', sep='\t',header=None)      # 验证集
test = pd.read_table('data/test.txt', sep='\t',header=None)    # 测试集

# 添加列名便于对数据进行更好处理
train.columns = ["text_a",'label']
dev.columns = ["text_a",'label']
test.columns = ["text_a"]

# 保存处理后的数据集文件
# train.to_csv('data/train.csv', sep='\t', index=False)  # 保存训练集，格式为text_a,label
# dev.to_csv('data/dev.csv', sep='\t', index=False)      # 保存验证集，格式为text_a,label
# test.to_csv('data/test.csv', sep='\t', index=False)    # 保存测试集，格式为text_a

# 导入所需的第三方库
import math
import numpy as np
import os
import collections
from functools import partial
import random
import time
import inspect
import importlib
from tqdm import tqdm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import IterableDataset
from paddle.utils.download import get_path_from_url


# 导入paddlenlp所需的相关包
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import MapDataset
from paddle.dataset.common import md5file
from paddlenlp.datasets import DatasetBuilder

# 定义要进行分类的14个类别
label_list=list(train.label.unique())
print(label_list)

# 定义数据集对应文件及其文件存储格式
class MyDataset(DatasetBuilder):
    SPLITS = {
        'train':'./data/train.csv',
        'dev':'./data/dev.csv'
    }

    def _get_data(self, mode: str, **kwargs):
        filename = self.SPLITS[mode]
        return filename

    def _read(self, filename: str, *args):
        with open(filename,'r',encoding='utf-8') as f:
            head = None
            for line in f:
                data = line.strip().split('\t')
                if not head:
                    head = data
                else:
                    text_a,label = data
                    yield {"text_a": text_a, "label": label}  # 此次设置数据的格式为：text_a,label，可以根据具体情况进行修改

    def get_labels(self):
        return label_list


def load_dataset(name=None,data_files=None,splits=None,lazy=None,**kwargs):

    reader_cls = MyDataset


    if not name:
        reader_instance = reader_cls(lazy=lazy,**kwargs)
    else:
        reader_instance = reader_cls(lazy=lazy,name=name,**kwargs)

    datasets = reader_instance.read_datasets(data_files=data_files,splits=splits)
    return datasets

train_ds, dev_ds = load_dataset(splits=["train", "dev"])


def convert_example(example,tokenizer,max_seq_length=128,is_test=False):
    qtconcat = example['text_a']
    encoder_inputs = tokenizer(text=qtconcat,max_seq_length=max_seq_length)
    input_ids = encoder_inputs['input_ids']
    token_type_ids = encoder_inputs['token_type_ids']

    if not is_test:
        label = np.array(example['label'],dtype='int64')
        return input_ids,token_type_ids,label
    else:
        return input_ids, token_type_ids

def create_dataloader(dataset,mode='train',batch_size=16,batchify_fn=None,trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,batch_size,shuffle=True)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,batch_size,shuffle=False)

    return paddle.io.DataLoader(
        dataset,
        batch_sampler,
        collate_fn=batchify_fn,
        return_list=True
    )

# 参数设置：
# 批处理大小，显存如若不足的话可以适当改小该值
batch_size = 300
# 文本序列最大截断长度，需要根据文本具体长度进行确定，最长不超过512。 通过文本长度分析可以看出文本长度最大为48，故此处设置为48
max_seq_length = 100

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length
)

