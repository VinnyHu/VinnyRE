
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import random
from tokenization.word_tokenizer import WordTokenizer
import sklearn
from sklearn import metrics
class BagREDataset(data.Dataset):

    def __init__(self,path,opt,rel2id,token2id):
        super().__init__()
        self.path = path
        self.tokenizer = WordTokenizer(opt=opt,vocab=token2id)
        self.rel2id = rel2id
        self.entpair_as_bag = opt.entpair_bag
        self.bag_size = opt.bag_size
        self.data = []
        with open(self.path) as f:
            for line in f:
                line = line.rstrip()
                if len(line) > 0:
                    self.data.append(eval(line))


        if opt.mode == None:
            self.weight = np.zeros((len(self.rel2id)),dtype=np.float32)
            self.bag_scope = []
            self.name2id = {}
            self.bag_name = []
            self.facts = {}

            for idx,item in enumerate(self.data):
                fact = (item['h']['id'], item['t']['id'], item['relation'])
                # print(type(fact))
                # print(fact)
                if item['relation'] != 'NA':
                    self.facts[fact] = 1
                if self.entpair_as_bag:
                    name = (item['h']['id'], item['t']['id'])
                else:
                    name = fact


                if name not in self.name2id:
                    self.name2id[name] = len(self.name2id)
                    self.bag_scope.append([])
                    self.bag_name.append(name)
                self.bag_scope[self.name2id[name]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0

            self.weight = 1.0 / (self.weight ** 0.05)
            self.weight = torch.from_numpy(self.weight)

            #print(self.bag_scope)
            #print(self.bag_scope.size())
        else:
            pass

    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self,index):

        bag = self.bag_scope[index]
        if self.bag_size is not None:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag

        seqs = None
        rel = self.rel2id[self.data[bag[0]]['relation']]
        for sent_id in bag:
            item = self.data[sent_id]
            seq = list(self.tokenizer.tokenizer(item))
            #print(len(seq))
            #print(seq)
            if seqs is None:
                seqs = []
                for i in range(len(seq)):
                    seqs.append([])

            for i in range(len(seq)):
                seqs[i].append(seq[i])

        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)


        return [rel,self.bag_name[index], len(bag)] + seqs

    def collate_fn(data):
        """
        It operates the data from __getitem__
        get a list,which size is batch_size and item is the result of __getitem__
        get a list
        :return:
        """
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        #print(seqs[0])
        for i in range(len(seqs)):   #The style is not clear,it can be rewrite
            seqs[i] = torch.cat(seqs[i],0)

        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert(start == seqs[0].size(0))

        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope] + seqs

    def eval(self, pred_result):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)
        for i, item in enumerate(sorted_pred_result):
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        auc = metrics.auc(x=rec, y=prec)

        np_prec = np.array(prec)
        np_rec = np.array(rec)
        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        return {'prec': np_prec, 'rec': np_rec, 'mean_prec': mean_prec, 'f1': f1, 'auc': auc}


def BagRELoader(path,opt,rel2id, token2id, collate_fn=BagREDataset.collate_fn):

    dataset = BagREDataset(path,opt, rel2id,token2id)


    data_loader = data.DataLoader(dataset=dataset,
            batch_size=opt.batch_size,
            shuffle=opt.shuffle,
            pin_memory=True,
            num_workers=opt.num_workers,
            collate_fn=collate_fn
            )

    return data_loader

# def test():
#     a = torch.tensor([[1,2,3],[2,4,6],[1,4,6],[4,6,8]])
#     print(a)
#     print("---------------")
#     print(a[1:])
#     b = torch.argmax(a[1:],dim=0)
#     print(b)
#
#
# test()

def test():
    from tqdm import tqdm
