import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import gzip
import pickle
import json

# VGG: using PosNeg blocks
from PosNegVGG import PosNegVgg
from Res_PosNeg_VGG import Res_PosNeg_VGG

def _dump(loss_collect, accu_collect, block_count):
    json.dump([loss_collect, accu_collect], open("./records/block{}.json".format(block_count), "w"))

def train(vgg, config):
    # file IO
    fhand = gzip.open("./mnist.pkl.gz")
    data = pickle.load(fhand, encoding="latin")
    train, test = data[0], data[2]

    # meta-params
    bs = config['bs']
    epochs = config['epochs']
    lossF = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg.parameters(), lr=config['lr'], momentum=config['momentum'])
    zero_pad = nn.ZeroPad2d(2)

    # training
    batchcount = 0
    loss_collect = []
    accu_collect = []
    # block design
    loss_per_block = []
    accu_per_block = []
    block_count = 0
    # training
    for epo in range(epochs):
        for t in range(len(train[1]) // bs):
            # batch
            batchcount += 1
            minibatch = (torch.Tensor(train[0][t*bs: (t+1)*bs]), torch.LongTensor(train[1][t*bs: (t+1)*bs]))

            # forwards
            inputs = zero_pad(minibatch[0].reshape(bs, 1, 28, 28))
            logits = vgg.forward(inputs)
            loss = lossF(logits, minibatch[1])

            # BP
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # summarize
            temp = np.array(np.argmax(logits.detach().numpy(), axis=1) == minibatch[1].detach().numpy())
            hits = np.sum(temp)
            print("batchcount = #{}, loss = {}, hits = {}/{}".format(\
                batchcount, loss.item(), hits, bs)\
            )
            # print(logits[0])
            loss_collect.append(loss.item())
            accu_collect.append(hits/bs)

            if len(loss_collect) == 100:
                loss_per_block.append(np.mean(loss_collect))
                accu_per_block.append(np.mean(accu_collect))
                _dump(loss_collect, accu_collect, block_count)
                loss_collect = []
                accu_collect = []
                block_count += 1

    print("training loss = {}, accu = {}".format( np.mean(loss_per_block), np.mean(accu_per_block) ))
    if len(loss_collect) > 0:
        _dump(loss_collect, accu_collect, block_count)
        loss_collect = []
        accu_collect = []
        block_count += 1

    # block design: final empty block as ending sign
    _dump([], [], block_count)

    # test
    loss_collect = []
    accu_collect = []
    bs = 100
    for t in range(len(test[1]) // bs):
        print("test-batch = {}".format(t))
        # batch
        minibatch = (torch.Tensor(train[0][t*bs: (t+1)*bs]), torch.Tensor(train[1][t*bs: (t+1)*bs]))
        # forward
        logits = vgg.forward(minibatch[0].reshape(bs, 1, 28, 28))
        loss = lossF(logits, minibatch[1])
        # summarize
        loss_collect.append(loss.item())
        accu_collect.append(np.sum(np.argmax(logits.detach().numpy(), axis=1) == minibatch[1]))

    print("test loss = {}, accu = {}".format( np.mean(loss_collect), np.mean(accu_collect) ))

    return vgg

if __name__ == '__main__':
    vgg = PosNegVgg()
    config = {'bs':100,'epochs':5,'lr':5e-2, 'momentum':0.9}
    vgg = train(vgg, config)
    pickle.dump(vgg, open("PosNegVgg.pkl", "wb"))