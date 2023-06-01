import os
from typing import Any
import pandas as pd
import sys
import re
import csv


sources = []
targets = []
src_tgts = []
def data_process(fliename):
    with open(fliename, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
    rf.close()
    for line in lines:
        parts = re.split(r'\t+', line.strip())
        id,source = parts[:2]
        for part in parts[2:]:
            if part != '没有错误':
                sources.append(source)
                target = part
                targets.append(target)
                src_tgts.append(source+'\t'+target)
            else:
                targets.append(source)
    #将sources按行写入文件src.txt
    with open('src.txt', 'w', encoding='utf-8') as wf:
        for source in sources:
            wf.write(source+'\n')
    #将targets按行写入文件tgt.txt
    with open('tgt.txt', 'w', encoding='utf-8') as wf:
        for target in targets:
            wf.write(target+'\n')
    #将src_tgt按行写入文件src_tgt.txt
    with open('src_tgt.txt', 'w', encoding='utf-8') as wf:
        for src_tgt in src_tgts:
            wf.write(src_tgt+'\n')

def data_remove_filter(source_filename,filter_filename):
    with open(source_filename,encoding='utf-8') as source_sentences:
        sources = source_sentences.readlines()
    with open(filter_filename,encoding='utf-8') as filter_sentences:
        filters = filter_sentences.readlines()
    r_sources=[]
    r_filters=[]
    for line in sources:
        r_line = replace_punctuation(line)
        r_sources.append(r_line)
    for line in filters:
        r_line = replace_punctuation(line)
        r_filters.append(r_line)

    sources_data = [line.strip().split('\t') for line in r_sources]
    filtered_data = [line for line in sources_data if line[1] not in r_filters]

    with open('new_data.txt', 'w', encoding='utf-8') as new_file:
        for row in filtered_data:
            line = '\t'.join(row)+'\n'
            new_file.write(line)


def replace_punctuation(sentence):
    punctuation_dict = {'“':'\"','”':'\"','’':'\'','‘':'\'','：':':'}  # 添加更多中英文标点的映射关系
    replaced_sentence = ''
    for char in sentence:
        if char in punctuation_dict:
            replaced_sentence += punctuation_dict[char]
        else:
            replaced_sentence += char

    return replaced_sentence


if __name__ == '__main__':
    # filename = 'new_data.txt'
    print(os.getcwd())  # 这样就能看到目前Python搜索路径在哪里，如果报错找不到，多半是你这个路径下没有文件
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 这里保险的就是直接先把绝对路径加入到搜索路径
    sys.path.insert(0, os.path.join(BASE_DIR))
    sys.path.insert(0, os.path.join(BASE_DIR, 'data'))  # 把data所在的绝对路径加入到了搜索路径，这样也可以直接访问dataset.csv文件了
    # 这句代码进行切换目录
    os.chdir(BASE_DIR)   # 把目录切换到当前项目，这句话是关键

    source_filename='./MuCGEC/MuCGEC_dev.txt'
    filter_filename='./MuCGEC/filter_sentences.txt'
    data_remove_filter(source_filename,filter_filename)
    # data_process(filename)
    
