import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig

from tqdm import tqdm
from utils import DynamicCRF,nll_loss,fc_nll_loss,fc_nll_loss_mean,batchify

from data_process import MyDataset
from Model import Generator, Discriminator
from utils import *



def train():
    #--------------------设置参数---------------------
    bert_path='./pretrained_model/roberta-base-chinese'
    train_data_path='F:/python/GAN_GEC/data/MuCGEC_data/src_tgt.txt'
    dev_data_path='./data/HybirdSet/dev.txt'
    test_data_path='./data/HybirdSet/test.txt'
    vocab_pach = './pretrained_model/roberta-base-chinese/vocab.txt'
        # parser.add_argument('--label_data',type=str)
    batch_size=4#64
    val_batch_size = 1
    max_length = 128#每句话的最大长度
    lr=1e-5
    dropout=0.1
    freeze=0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # parser.add_argument('--freeze',type=int)
        # parser.add_argument('--number_class', type = int)
    number_epoch=1
    # gpu_id=0
    # fine_tune=True
    print_every=5#100
    save_every=5#2000
    # bert_vocab = 'model/model/bert/vocab.txt'
    loss_type = 'FC_FT_CRF'
    gamma=0.5#在焦点损失函数中的一个超参数
    model_save_path='./result_model'
    model_save_name='model_1007_0'
    prediction_max_len=64
    # dev_eval_path='./models/test1dev_pred1.txt'
    # final_eval_path='./models/test1dev_eval.txt'
    l2_lambda=1e-5
    training_max_len=64
    # plot_path = './models/test1'
    num_class = 2

    # --- create model save path --- #
    directory = model_save_path
    try:
        os.stat(directory)
    except:
        os.mkdir(directory) 

    #准备字典
    id_label_dict = {}
    with open(vocab_pach, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
        #这里从0开始是因为，bert的编码是从0开始算的，所以保持一致
        i = 0
        for line in lines:
            id_label_dict[i] = line.strip('\n')
            i=i+1
    
    
    train_dataset = MyDataset(train_data_path, bert_path, max_length)
    # val_dataset = Mydataset(dev_data_path, bert_path, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    vocab_class = train_dataset.get_num_class()

    bert_config = BertConfig.from_pretrained(bert_path)
    generator = Generator(num_class=vocab_class,embedding_size=bert_config.hidden_size,batch_size=batch_size,bert_path=bert_path,device=device, dropout=dropout).to(device)
    discriminator = Discriminator(bert_path=bert_path, num_class=num_class, embedding_size=bert_config.hidden_size,dropout=dropout).to(device)

    adversarial_loss = nn.BCELoss().to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(number_epoch):
        print('----------------------------------------')
        print('%d epoch have run' % epoch)
        loss_accumulated = 0.
        loss_crf_accumulated = 0.
        loss_ft_accumulated = 0.

        result_ori_sentences = []
        result_pred_sentences = []
        result_pair = []
        train_bar = tqdm(train_dataloader, ncols=100)
        for error_input_ids, error_token_type_ids, error_attention_mask, \
            true_input_ids, true_token_type_ids, true_attention_mask in train_bar:

            valid = torch.ones((batch_size, 1)).to(device)#正例
            fake = torch.zeros((batch_size, 1)).to(device)#负例

            #病句
            error_input_ids = error_input_ids.to(device)
            error_attention_mask = error_attention_mask.to(device)
            error_token_type_ids = error_token_type_ids.to(device)
            #无语病句子
            true_input_ids = true_input_ids.to(device)
            true_attention_mask = true_attention_mask.to(device)
            true_token_type_ids = true_token_type_ids.to(device)

            #训练判别器
            discriminator_optimizer.zero_grad()
            
            true_pred = discriminator(true_input_ids,true_attention_mask,true_token_type_ids)
            true_loss = adversarial_loss(true_pred, valid)

            d_gen_token_ids,_,_,_ = generator(error_input_ids, error_attention_mask, error_token_type_ids,true_attention_mask,true_input_ids,max_length)
            d_gen_token_ids = torch.tensor(d_gen_token_ids)
            fake_pred = discriminator(d_gen_token_ids,error_attention_mask,error_token_type_ids)
            fake_loss = adversarial_loss(fake_pred, fake)

            discriminator_loss = true_loss + fake_loss #除2?
            fake_reg_loss = 0.
            for param in discriminator.parameters():
                fake_reg_loss += torch.norm(param, p=2)
            discriminator_loss += l2_lambda * fake_reg_loss

            discriminator_loss.backward()
            discriminator_optimizer.step()

            #训练生成器
            generator_optimizer.zero_grad()
            g_gen_token_ids,fc_loss,_,_ = generator(error_input_ids, error_attention_mask, error_token_type_ids,true_attention_mask,true_input_ids,max_length)
            g_gen_token_ids = torch.tensor(g_gen_token_ids)
            gentrain_loss = adversarial_loss(discriminator(g_gen_token_ids,error_attention_mask,error_token_type_ids), valid)

            gen_loss = gentrain_loss + fc_loss

            gen_reg_loss = 0.
            for param in generator.parameters():
                gen_reg_loss += torch.norm(param, p=2)
            gen_loss += l2_lambda * gen_reg_loss

            gen_loss.backward()
            generator_optimizer.step()
            if epoch == number_epoch - 1:
                assert true_input_ids.shape==g_gen_token_ids.shape
                #生成原句
                for idx,input_ids in enumerate(error_input_ids):
                    spl_ori_sentences = get_ids_to_sen(input_ids,error_attention_mask[idx],id_label_dict)
                    ori_sentence = ''.join(char for char in spl_ori_sentences)
                    result_ori_sentences.append(ori_sentence)
                #生成预测
                for idx,input_ids in enumerate(g_gen_token_ids):
                    spl_pred_sentences = get_ids_to_sen(input_ids,error_attention_mask[idx],id_label_dict)
                    pred_sentence = ''.join(char for char in spl_pred_sentences)
                    result_pred_sentences.append(pred_sentence)

        with open('result.txt', 'w', encoding='utf-8') as f:
            for idx,ori_sentence in enumerate(result_ori_sentences):
                f.write(str(idx)+'\t'+ori_sentence+'\t'+result_pred_sentences[idx]+ '\n')


if __name__ == '__main__':
    train()



