import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer,BertForMaskedLM,BertConfig,BertTokenizerFast


class MyDataset(Dataset):
    def __init__(self, filename, bert_path, max_length):
        # 数据集初始化
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.num_class = self.tokenizer.vocab_size#使用该方法可以得到预训练模型的字典大小
        self.texts = []
        self.ori_input_ids = []
        self.ori_token_type_ids = []
        self.ori_attention_mask = []
        self.true_input_ids = []
        self.true_token_type_ids = []
        self.true_attention_mask = []
        self.load_data(filename)

    def load_data(self, filename):
        # 加载数据
        print('loading data from:', filename)
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        q = 1
        for line in tqdm(lines, ncols=100):
            ori_sen,true_sen = self.process_one_line(line)
            self.texts.append(np.array(ori_sen))
            #这里将错误的句子转换成bert的三个输入向量
            token = self.tokenizer(ori_sen, add_special_tokens=True, padding='max_length', truncation=True,
                                   max_length=self.max_length)
            self.ori_input_ids.append(np.array(token['input_ids']))
            self.ori_token_type_ids.append(np.array(token['token_type_ids']))
            self.ori_attention_mask.append(np.array(token['attention_mask']))
            #对正确的句子也做同样的处理
            true_token = self.tokenizer(true_sen, add_special_tokens=True, padding='max_length', truncation=True,
                                   max_length=self.max_length)
            self.true_input_ids.append(np.array(true_token['input_ids']))
            self.true_token_type_ids.append(np.array(true_token['token_type_ids']))
            self.true_attention_mask.append(np.array(true_token['attention_mask']))
            q = q + 1
            if q > 32:
                break

    def __getitem__(self, index):
        return self.ori_input_ids[index], self.ori_token_type_ids[index], self.ori_attention_mask[index],self.true_input_ids[index], self.true_token_type_ids[index], self.true_attention_mask[index]
        # self.texts[index],

    def __len__(self):
        return len(self.ori_input_ids)

    def get_num_class(self):
        return self.num_class

    def get_texts(self):
        return self.texts

    # 读一条数据
    # 构造bert输入格式，word embdding,最终处理过的句子应该时一个str而不是list，否则会报错
    def process_one_line(self, line):
        #text_list 为原始错句， tag_list 为正确句对应的index序列
        content_list = line.strip().split('\t')
        # 为什么加这句话？因为每行数据是三个部分，包含错误句子，正确句子
        assert len(content_list) == 2  # 断言，当条件不满足时报错
        # 文本序列，遍历并加到list中
        # 这里的<SEP>标注出了一个句子应该终止的地方
        text_list = [w for w in content_list[0].strip()]  
        # 这是正确句子的序列
        tag_name_list = [w for w in content_list[1].strip()] 
        # TtT论文中说的三种情况，T>T',T=T',T<T'
        # T<T'插入操作 输出大于输入
        if len(tag_name_list) > len(text_list):
            text_list += ['[MASK]'] * (len(tag_name_list) - len(text_list))  # 在原句中添加mask
            # text_list += ['']  # 标志句子的结束
            # tag_name_list += ['']
        # T>T'删除操作，输出小于输入,这里在[sep]后补pad会导致该数据进入tokenizer后出来的数据最后还有一个sep
        elif len(tag_name_list) < len(text_list):
            tag_name_list += ['[PAD]'] * (len(text_list) - len(tag_name_list))
            # text_list += ['']
        # T=T'
        else:
            tag_name_list += ['']
            text_list += ['']
        assert len(text_list) == len(tag_name_list)  # 断言保证句子长度一样
        # tag_list = list()
        # for token in tag_name_list:
        #     tag_list.append(self.label_dict.token2idx(token))  # label_dict是Bert词表
        #在这里对处理好的句子进行拼接
        text = ""
        for tok in text_list:
            text = text+tok
        tag = ""
        for tok in tag_name_list:
            tag = tag+tok
        return text, tag