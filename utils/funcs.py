import numpy as np

'''
    从tokenizer的输出中生成原始的句子
'''
def get_ids_to_sen(input_ids,attention_mask,id_label_dict,token_type_ids=None):
    text_list = []
    input_ids = input_ids.tolist()
    attention_mask = attention_mask.tolist()
    for index,lab in enumerate(input_ids):
        if attention_mask[index] != 0:
            token = id_label_dict[lab]
            text_list.append(token)
    return text_list


def process_batch_tag(in_batch_tag_list, label_dict):
    
    max_len = 0
    #从列表中取出句子更新最大长度，这个最大长度就是bert输入的最大长度。这里传入的句子是已经word embdding的句子
    for instance in in_batch_tag_list:
        max_len = max(len(instance), max_len)
    max_len += 1 # for [CLS]
    #print (max_len)
    #结果list()？
    result_batch_tag_list = list()
    for instance in in_batch_tag_list:
        #取出一个句子
        one_tag_list = []
        #在句子前面加[cls]的embdding
        one_tag_list.append(label_dict.token2idx('<-CLS->')) # for [CLS]
        one_tag_list.extend(instance)
        #把句子补到最大长度
        len_diff = max_len - len(one_tag_list)
        for _ in range(len_diff):#给句子后边补pad
            one_tag_list.append(label_dict.token2idx('<-PAD->')) # for padding
        #在返回结果中加已经扩展好的句子list
        result_batch_tag_list.append(one_tag_list)

    #矩阵结构第一个维度是句子数量，第二维度是句子长度
    result_batch_tag_matrix = np.array(result_batch_tag_list)
    #print (result_batch_tag_matrix.shape)
    assert result_batch_tag_matrix.shape == (len(in_batch_tag_list), max_len)
    return result_batch_tag_matrix

#制作一个batch中的句子的mask embdding
def make_mask(in_batch_tag_list):
    #找出最长的句子作为bert每句话的长度
    max_len = 0
    for instance in in_batch_tag_list:
        max_len = max(len(instance), max_len)
    max_len += 1 # for [CLS]

    #制作返回结果
    result_mask_matrix = []
    for instance in in_batch_tag_list:
        #创建一个list，记录每句话的mask embdding
        one_mask = list()
        for _ in range(len(instance) + 1): # 1 for [CLS]
            #在句子中非<pad>的地方mask标记为1
            one_mask.append(1.0)
        len_diff = max_len - len(one_mask)
        for _ in range(len_diff):
            #在补长<pad>的地方mask标为0
            one_mask.append(0.0)
        result_mask_matrix.append(one_mask)
    # result shape = [batch_size, seq_len]
    result_mask_matrix = np.array(result_mask_matrix)
    assert result_mask_matrix.shape == (len(in_batch_tag_list), max_len)
    return result_mask_matrix

#比较预测结果,pred_batch_tag_matrix:预测出的句子，true_batch_matrix:真正的句子
#该函数应该是用来修正数据，因为预测的结果中有[cls],所以预测结果与真实结果有1个字符的偏移
def get_valid_predictions(pred_batch_tag_matrix, true_batch_matrix):
    pred_tag_result_matrix = []
    assert len(pred_batch_tag_matrix) == len(true_batch_matrix)#比较长度
    batch_size = len(true_batch_matrix)
    for i in range(batch_size):
        valid_len = len(true_batch_matrix[i])#某句话的长度
        #取对应的预测的句子，从1开始是因为添加了[cls]
        one_pred_result = pred_batch_tag_matrix[i]#[1: valid_len + 1]

        assert len(one_pred_result) == len(true_batch_matrix[i])
        pred_tag_result_matrix.append(one_pred_result)
    return pred_tag_result_matrix

#联合结果输出到一个文件中
def combine_result(gold_lines, pred_path, out_path, id_label_dict):
    #输出结果？
    with open(out_path, 'w', encoding = 'utf8') as o:
        with open(pred_path, 'r', encoding = 'utf8') as p:
            pred_lines = p.readlines()
        assert len(gold_lines) == len(pred_lines)
        #数据的长度
        data_num = len(gold_lines)
        for i in range(data_num):
            #取一条数据，一句话
            pred_l = pred_lines[i]
            text_list = gold_lines[i][0]
            gold_label_list = gold_lines[i][1]
            
            pred_l = pred_lines[i]
            pred_content_list = pred_l.strip('\n').split('\t')
            pred_label_str = pred_content_list[1]
            
            pred_label_list = pred_label_str.split()
            assert len(gold_label_list) == len(pred_label_list)
            
            instance_len = len(text_list)
            for j in range(instance_len):
                out_str = text_list[j] + ' ' + id_label_dict[gold_label_list[j]] + ' ' + pred_label_list[j]
                o.writelines(out_str + '\n')
            o.writelines('\n')

'''
方法返回两个矩阵，矩阵行为一条句子，列为每个句子中的字，两个矩阵分别是标签矩阵，mask矩阵
'''
def get_tag_mask_matrix(batch_text_list):
    tag_matrix = []
    mask_matrix = []
    batch_size = len(batch_text_list)
    max_len = 0
    for instance in batch_text_list:
        max_len = max(len(instance), max_len)
    max_len += 2 # 1 for [CLS] 1 for [SEP]
    for i in range(batch_size):
        one_text_list = batch_text_list[i]
        one_tag = list(np.zeros(max_len).astype(int))
        tag_matrix.append(one_tag)
        one_mask = [1]
        #某条句子的长度
        one_valid_len = len(batch_text_list[i])
        for j in range(one_valid_len):
            one_mask.append(1)
        #把句子中<pad>延长的部分找出mask0
        len_diff = max_len - len(one_mask)
        for _ in range(len_diff):
            one_mask.append(0)
        mask_matrix.append(one_mask)
        assert len(one_mask) == len(one_tag)
    return np.array(tag_matrix), np.array(mask_matrix)

'''
跟序列中间插入空格
'''
def join_str(in_list):
    out_str = ''
    for token in in_list:
        out_str += str(token) + ' '
    return out_str.strip()

def predict_one_text_split(text_split_list, seq_tagging_model, label_dict):
    # text_split_list is a list of tokens ['word1', 'word2', ...]
    text_list = [text_split_list]
    tag_matrix, mask_matrix = get_tag_mask_matrix(text_list)

    decode_result = seq_tagging_model(text_list, mask_matrix, tag_matrix, fine_tune = False)[0]
    valid_text_len = len(text_split_list)

    valid_decode_result = decode_result[0][1: valid_text_len + 1]

    tag_result = []
    for token in valid_decode_result:
        tag_result.append(label_dict[int(token)])
    return tag_result
    #return valid_decode_result

def get_text_split_list(text, max_len):
    result_list = []
    text_list = text.split()
    valid_len = len(text_list)
    split_num = (len(text_list) // max_len) + 1
    if split_num == 1:
        result_list = [text_list]
    else:
        b_idx = 0
        e_idx = 1
        for i in range(max_len):
            b_idx = i * max_len
            e_idx = (i + 1) * max_len
            result_list.append(text_list[b_idx:e_idx])
        if e_idx < valid_len:
            result_list.append(text_list[e_idx:])
        else:
            pass
    return result_list

def predict_one_text(text, max_len, seq_tagging_model, label_dict):
    text_split_list = get_text_split_list(text, max_len)
    all_text_result = []
    all_decode_result = []
    for one_text_list in text_split_list:
        one_decode_result = predict_one_text_split(one_text_list, seq_tagging_model, label_dict)
        all_text_result.extend(one_text_list)
        all_decode_result.extend(one_decode_result)
    result_text = join_str(all_text_result)
    tag_predict_result = join_str(all_decode_result)
    return result_text + '\t' + tag_predict_result

def get_id_label_dict(label_path):
    label_dict = {}
    with open(label_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            content_list = l.strip('\n').split()
            label_id = int(content_list[1])
            label = content_list[0]
            label_dict[label_id] = label
    return label_dict
