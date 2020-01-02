import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import json
from preprocess.dataloader import BagRELoader
from config import DefaultConfig
from model.PCNN import PCNN
from criterion.AverageMeter import AverageMeter
import os
from tqdm import tqdm
import numpy as np
path = 'data/nyt10/'
ckpt = 'ckpt/nyt10_pcnn_one.pth.tar'
def main():
    opt = DefaultConfig()
    rel2id = json.load(open('data/nyt10/nyt10_rel2id.json'))
    wordi2d = json.load(open( 'pretrain/glove/glove.6B.50d_word2id.json'))
    word2vec = np.load('pretrain/glove/glove.6B.50d_mat.npy')

    if not '[UNK]' in wordi2d:
        wordi2d['[UNK]'] = len(wordi2d)
    if not '[PAD]' in wordi2d:
        wordi2d['[PAD]'] = len(wordi2d)

    id2rel = {}
    for rel,id in rel2id.items():
        id2rel[id] = rel

    train_loader = BagRELoader(path=path+'nyt10_train.txt',opt=opt,rel2id=rel2id,token2id=wordi2d)
    val_loader = BagRELoader(path=path+'nyt10_val.txt',opt=opt,rel2id=rel2id,token2id=wordi2d)
    test_loader = BagRELoader(path=path + 'nyt10_test.txt',opt=opt,rel2id=rel2id,token2id=wordi2d)

    model = PCNN(opt,token2id=wordi2d,vectors=word2vec)

    if opt.loss_weight:
        criterion = nn.CrossEntropyLoss(weight=train_loader.dataset.weight)
    else:
        criterion = nn.CrossEntropyLoss()

    params = model.parameters()
    optimizer = optim.Adam(params,lr=opt.lr,weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        model.cuda()

    best_auc = 0.0
    for epoch in range(opt.max_epoch):
        model.train()
        print("=== Epoch %d train ===" % epoch)
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_pos_acc = AverageMeter()

        t = tqdm(train_loader)
        for iter,data in enumerate(t):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass

            label = data[0]

            bag_name = data[1]

            scope = data[2]

            args = data[3:]


            logits = model(data[3],data[4],data[5],data[6])


            # get each bag's instances
            bag_logits = torch.zeros((len(scope),opt.num_classes))
            bag_logits = bag_logits.cuda()

            for i in range(len(scope)):
                temp_bag = logits[scope[i][0]:scope[i][1]]

                instance_bag = torch.argmax(temp_bag,dim=0)

                bag_logits[i] = temp_bag[instance_bag[label[i] - 1]]

            # print(bag_logits)




            loss = criterion(bag_logits,label)


            score, pred = bag_logits.max(-1)  # (B)  problem

            acc = float((pred == label).long().sum()) / label.size(0)

            pos_total = (label != 0).long().sum()

            pos_correct = ((pred == label).long() * (label != 0).long()).sum()
            if pos_total > 0:
                pos_acc = float(pos_correct) / float(pos_total)
            else:
                pos_acc = 0

            # Log
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)

            # Optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #val
            print("=== Epoch %d val ===" % epoch)
            result = eval_model(model,val_loader,id2rel,num_class=opt.num_classes)
            print("auc: %.4f" % result['auc'])
            print("f1: %.4f" % (result['f1']))

            if result['auc'] > best_auc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': model.state_dict()}, ckpt)
                best_auc = result['auc']
            print("Best auc on val set: %f" % (best_auc))

def eval_model(model,eval_loader,id2rel,num_class):
    model.eval()
    with torch.no_grad():
        t = tqdm(eval_loader)
        pred_result = []
        for iter, data in enumerate(t):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass
            label = data[0]
            bag_name = data[1]
            scope = data[2]
            args = data[3:]
            logits = model(data[3],data[4],data[5],data[6])  # results after softmax
            bag_logits = torch.zeros((len(scope), num_class))
            bag_logits = bag_logits.cuda()

            for i in range(len(scope)):
                temp_bag = logits[scope[i][0]:scope[i][1]]

                instance_bag = torch.argmax(temp_bag, dim=0)

                bag_logits[i] = temp_bag[instance_bag[label[i] - 1]]

            for i in range(bag_logits.size(0)):
                for relid in range(num_class):
                    if id2rel[relid] != 'NA':
                        # print(bag_name[i])
                        pred_result.append({
                            'entpair': bag_name[i][:2],
                            'relation':id2rel[relid],
                            'score': bag_logits[i][relid].item()
                        })
        result = eval_loader.dataset.eval(pred_result)
    return result

if __name__ == '__main__':
    main()
