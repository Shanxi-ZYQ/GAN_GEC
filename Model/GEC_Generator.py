import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertModel
from utils import DynamicCRF,nll_loss,fc_nll_loss,fc_nll_loss_mean,batchify

'''生成器模型,采用CRF的结构'''
class Generator(nn.Module):
    def __init__(self, num_class,embedding_size, batch_size, bert_path='chinese_wwm_ext_pytorch', device='cpu', dropout=0.1):
        super(Generator, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.num_class = num_class
        # self.vocab = vocab
        self.fc = nn.Linear(self.embedding_size, self.num_class)
        self.CRF_layer = DynamicCRF(num_class)
        # self.bert_vocab = vocab
    
    def forward(self,input_ids,attention_mask,token_type_ids,in_mask_matrix, in_tag_matrix,seq_len, fine_tune=False, gamma=None):
        batch_size = len(input_ids)
        # max_len = 0
        # for instance in text_data:
        #     max_len = max(max_len, len(instance))
        # seq_len = max_len + 1#加上[CLS]

        # mask_matrix = torch.tensor(in_mask_matrix, dtype=torch.uint8).t_().contiguous().to(self.device)
        # tag_matrix = torch.LongTensor(in_tag_matrix).t_().contiguous().to(self.device) # size = [seq_len, batch_size]
        # assert mask_matrix.size() == tag_matrix.size()
        # assert mask_matrix.size() == torch.Size([seq_len, batch_size])

        mask_matrix = torch.tensor(in_mask_matrix, dtype=torch.uint8).t_().contiguous().to(self.device)
        in_tag_matrix = torch.tensor(in_tag_matrix,dtype=torch.long)
        tag_matrix = torch.LongTensor(in_tag_matrix).t_().contiguous().to(self.device)
        assert mask_matrix.size() == tag_matrix.size()
        assert mask_matrix.size() == torch.Size([seq_len, batch_size])

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # input text_data.size() = [batch_size, seq_len]
        # data = batchify(text_data, self.vocab) # data.size() == [seq_len, batch_size]，转化为tensor
        # data = data.cuda(self.device)
        sequence_representation = output.last_hidden_state

        #sequence_representation为原始错句经过bert的encoder表示，[seq_len, batch_size, embedding_size]
        # sequence_representation = self.bert_model.work(data)[0].cuda(self.device)
        # dropout
        sequence_representation = F.dropout(sequence_representation, p=self.dropout, training=self.training)
        sequence_representation = sequence_representation.contiguous().view(batch_size*seq_len,self.embedding_size)
        sequence_representation = sequence_representation.view(batch_size * seq_len, self.embedding_size)

        #2.4公式(3)
        #self.fc为线性层，sequence_emissions维度为[seq_len, batch_size, num_class]，num_class即字典中不同的字符个数
        sequence_emissions = self.fc(sequence_representation)
        #调整维度
        sequence_emissions = sequence_emissions.view(seq_len, batch_size, self.num_class)

        #计算encoder直接生成的输出的loss
        probs = torch.softmax(sequence_emissions, -1)
        probs = torch.tensor(probs,dtype=torch.int64)
        
        loss_ft_fc, g = fc_nll_loss_mean(probs, tag_matrix, mask_matrix, gamma=gamma)

        sequence_emissions = sequence_emissions.transpose(0, 1)
        tag_matrix = tag_matrix.transpose(0, 1) 
        mask_matrix = mask_matrix.transpose(0, 1)

        #CRF层
        loss_crf_fc = -self.CRF_layer(sequence_emissions, tag_matrix, mask_matrix,reduction='token_mean')

        decode_result = self.CRF_layer.decode(sequence_emissions, mask = mask_matrix)
        self.decode_scores, self.decode_result = decode_result
        self.decode_result = self.decode_result.tolist()

        loss = loss_crf_fc + loss_ft_fc
        #decode_result为解出的token_id
        return self.decode_result, loss, loss_crf_fc.item(), loss_ft_fc.item()
                                         
                                                 