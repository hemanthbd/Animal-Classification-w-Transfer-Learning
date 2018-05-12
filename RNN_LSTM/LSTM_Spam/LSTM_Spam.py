"""
Run similar model as in lab 4

modified from https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# import sklearn.svm

import numpy as np
import matplotlib.pyplot as plt

from util import plot_confusion_matrix


from torchtext.data import Field
from torchtext.data import BucketIterator
from torchtext.data import TabularDataset
import torch.nn as nn

DATA_ROOT = "./datasets/"

hidden_size = 500
embed = 200
tar_size = 2
batch_size = 3

use_gpu = torch.cuda.is_available()


def load_dataset(db_name, batch_size):
    """
    Load the csv datasets into torchtext files

    Inputs:
    db_name (string)
       The name of the dataset. This name must correspond to the folder name.
    batch_size
       The batch size
    """
    print("Loading " + db_name + "...")

    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)

    tv_datafields = [("sentence", TEXT),
                     ("label", LABEL)]

    trn, vld = TabularDataset.splits(
        path=DATA_ROOT + db_name,  # the root directory where the data lies
        train='train.csv', validation="test.csv",
        format='csv',
        skip_header=False,
        fields=tv_datafields)

    TEXT.build_vocab(trn)

    print("vocab size: %i" % len(TEXT.vocab))

    train_iter, val_iter = BucketIterator.splits(
        (trn, vld),
        batch_sizes=(batch_size, batch_size),
        device=-1,  # specify dont use gpu
        sort_key=lambda x: len(x.sentence),  # sort the sentences by length
        sort_within_batch=False,
        repeat=False)

    return train_iter, val_iter, len(TEXT.vocab)


class BaseNet(nn.Module):
    def __init__(self, embed, hidden_size, tar_size, len_vocab, batch_size):
        super(BaseNet, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(len_vocab, embed)
        self.lstmm = nn.LSTM(embed, hidden_size)
        self.fc = nn.Linear(hidden_size, tar_size)
        self.hidd = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda(),
                Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda())

    def forward(self, sentence, batch_size):

        # print("Sentence shape", sentence.shape)

        # print(self.embedding(sentence).view(len(sentence), 1, -1).shape)
        # print(self.hidd)
        # y = self.embedding(sentence).view(len(sentence),1,-1).shape
        # h0 = Variable(torch.zeros([1, batch_size, hidden_size]), requires_grad=False)
        # c0 = Variable(torch.zeros([1, batch_size, hidden_size]), requires_grad=False)
        # if use_gpu:
        #    h0 = h0.cuda()
        # print("Hidden", h0.shape)
        # if use_gpu:
        #   c0 = c0.cuda()

        output, (hn, cn) = self.lstmm(self.embedding(sentence), self.hidd)  #
        self.hidd = (hn, cn)
        # print("output", output.shape)
        # print("Hidden", hn.shape)
        # print(Variable(self.embedding(sentence)).shape)
        lin = self.fc(output[-1, :, :])
        # print(lin.shape)
        return lin

    def fit(self, train_iterator):
        # switch to train mode
        self.train()

        # define loss function
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        # setup SGD
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        for epoch in range(25):  # loop over the dataset multiple times
            running_loss = 0.0

            for i, batch in enumerate(train_iterator, 0):
                sentence = batch.sentence
                label = batch.label

                if use_gpu:
                    sentence = sentence.cuda()
                    label = label.cuda()

                # wrap them in Variable
                # sentence, label = Variable(sentence), Variable(label)

                # zero the parameter gradients
                self.zero_grad()

                self.hidd = self.init_hidden()

                # compute forward pass
                outputs = self.forward(sentence, batch_size=self.batch_size)

                # get loss function
                loss = criterion(outputs, label)

                # do backward pass
                loss.backward()

                # do one gradient step
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]

            print('[Epoch: %d] loss: %.3f' %
                  (epoch + 1, running_loss / (i + 1)))
            # print("Loss", running_loss)
            running_loss = 0.0
            #self.predict(test_iterator)

        print('Finished Training')

        #return pred_labels, test_labels

    def predict(self, test_iterator):
        # switch to evaluate mode
        self.eval()

        correct = 0
        total = 0
        all_predicted = []
        labels = []
        for batch in test_iterator:
            sentence = batch.sentence
            label = batch.label
            if use_gpu:
                sentence = sentence.cuda()
                #label = label.cuda()
            outputs = self.forward(sentence, batch_size=batch_size)
            _, predicted = torch.max(outputs.data, 1)
        
            #print("Pred", Variable(predicted).data.cpu().numpy().shape)
            #print("labels", label.data.cpu().numpy().shape)

            correct += (Variable(predicted).data.cpu().numpy() == label.data.cpu().numpy()).sum()
            all_predicted += predicted.tolist()
            labels += label.data.cpu().numpy().tolist()
        total = 1293
        print('Accuracy of the network on the 1293 test images: %d %%' % (
            100 * correct / total))
        #print("Corr", correct)
        #print("Pred", all_predicted)
        #print("label", label.data.cpu().numpy)

        return all_predicted, labels


def main():
    # get data

    train_iterator, test_iterator, len1 = load_dataset("spam", batch_size)
    print(train_iterator)
    print(torch.cuda.device_count())
    len_vocab = len1
    # full net
    print("LSTM Network")
    model = BaseNet(embed, hidden_size, tar_size, len_vocab, batch_size)
    if use_gpu:
        model = model.cuda()
    model.fit(train_iterator)
    pred_labels, test_labels = model.predict(test_iterator)
    #plt.figure(1)
    #print("pred",pred_labels)
    #print("test",test_labels)
    plot_confusion_matrix(pred_labels, test_labels, "LSTM_Spam")
    plt.show()
    plt.savefig('ConfusionMatrix_LSTM_Spam.png', )	
    torch.save(model, 'LSTM_Spam.pt')


main()
