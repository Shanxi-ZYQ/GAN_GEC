import torch


def nll_loss(y_pred, y, y_mask, avg=True):
    #每句中每个字的直接预测的损失
    cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))#取出y_pred在正确vocab上的值，再log
    cost = cost.view(y.shape)#(max_seq_length，batch_size)，每个值为正确单词的预测概率
    y_mask = y_mask.view(y.shape)

    #对每句的cost进行累加或平均
    if avg:
        cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
    else:
        cost = torch.sum(cost * y_mask, 0)
    cost = cost.view((y.size(1), -1))

    #对当前batch内的全部句的cost进行平均
    return torch.mean(cost) 

def fc_nll_loss_mean(y_pred, y, y_mask, gamma=None, avg=True):
    '''
    计算焦点损失
    '''
    if gamma is None:
        gamma = 2
    sum_cost=y_pred.sum(2)
    #取出y_pred在正确vocab上的值，再log
    #(max_seq_length，batch_size)，每个值为正确单词的预测概率
    p = torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)).view(y.shape)
    #p_sum=1/sum_cost.item()#=torch.full(p.shape,sum_cost.item())
    g = (1-torch.clamp(p, min=0.01, max=0.99))**gamma #g = (1 - p) ** gamma 
    cost = -g * torch.log(torch.div((p+1e-8),sum_cost))
    cost = cost.view(y.shape)
    y_mask = y_mask.view(y.shape)
    
    #对每句的cost进行累加或平均
    if avg:
        cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
    else:
        cost = torch.sum(cost * y_mask, 0)
    cost = cost.view((y.size(1), -1))

    #对当前batch内的全部句的cost进行平均
    return torch.mean(cost),g

def fc_nll_loss(y_pred, y, y_mask, gamma=None, avg=True):
    #2.5公式(10)，与nll_loss相似
    if gamma is None:
        gamma = 2
    p = torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1))
    g = (1-torch.clamp(p, min=0.01, max=0.99))**gamma
    #g = (1 - p) ** gamma 
    cost = -g * torch.log(p+1e-8)
    cost = cost.view(y.shape)
    y_mask = y_mask.view(y.shape)
    if avg:
        cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
    else:
        cost = torch.sum(cost * y_mask, 0)
    cost = cost.view((y.size(1), -1))
    return torch.mean(cost), g.view(y.shape)