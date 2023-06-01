import torch
from utils.bert_data import Vocab, CLS, SEP, MASK

#处理字符的工具
def init_bert_model(args, device, bert_vocab):
    bert_ckpt= torch.load(args.bert_path)
    bert_args = bert_ckpt['args']
    bert_vocab = Vocab(bert_vocab, min_occur_cnt=bert_args.min_occur_cnt, specials=[CLS, SEP, MASK])
    bert_model = BERTLM(device, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
        bert_args.dropout, bert_args.layers, bert_args.approx)
    bert_model.load_state_dict(bert_ckpt['model'])
    bert_model = bert_model.cuda(device)
    if args.freeze == 1:
        for p in bert_model.parameters():
            p.requires_grad=False
    return bert_model, bert_vocab, bert_args

def ListsToTensor(xs, vocab):
    batch_size = len(xs)
    lens = [ len(x)+2 for x in xs]
    mx_len = max(lens)
    ys = []
    for i, x in enumerate(xs):
        y =  vocab.token2idx([CLS]+x) + ([vocab.padding_idx]*(mx_len - lens[i]))
        ys.append(y)

    data = torch.LongTensor(ys).t_().contiguous()
    return data

def batchify(data, vocab):
    return ListsToTensor(data, vocab)