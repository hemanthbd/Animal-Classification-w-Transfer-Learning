import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sklearn.svm

import numpy as np
import matplotlib.pyplot as plt

from util import plot_confusion_matrix
from PIL import Image

import torchvision.models

num_batches = 64


def Animals():

    resize = transforms.Resize((224, 224))

    preprocessor = transforms.Compose(
        [resize, transforms.ToTensor(),
         transforms.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.ImageFolder(root='./animals/train', transform=preprocessor)
    print("Classes",train_dataset.classes)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=num_batches,
                                              shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.ImageFolder(root='./animals/test', transform=preprocessor)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=num_batches,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader



use_gpu = torch.cuda.is_available()

model = torchvision.models.vgg16(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier

if use_gpu:
    model = model.cuda()


clf = sklearn.svm.SVC(C=5.0, kernel='rbf', tol=0.0001)


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(self, x):
        x = model.forward(x)
        return x

    def fit(self, trainloader):
        self.train()

        mini_batches = np.floor(1384/num_batches)
        print("Mini_Batches",mini_batches)
        rem = 1384 % num_batches
        print("Remainder",rem)
        for epoch in range(1):  # loop over the dataset multiple times
            running_loss = 0.0
            count = 0
            output1 = np.zeros(shape=(int(mini_batches*num_batches), 4096))
            label1 = np.zeros(shape=(int(mini_batches*num_batches), 1))
            output2 = np.zeros(shape=(1384, 4096))
            label2 = np.zeros(shape=(1384, 1))
            op = []
            lb = []
            op1 = np.zeros((1384,4096))
            lb1 = np.zeros((1384,1))
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                if i < mini_batches:
                    inputs, labels = data

                    # wrap them in Variable

                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else :
                        inputs, labels = Variable(inputs), Variable(labels)


                    # compute forward pass
                    outputs = self.forward(inputs)


                    outputs = outputs.data.cpu().numpy()
                    output2[i*num_batches:(i+1)*num_batches, :] = outputs[:, :]
                    labels = labels.data.cpu().numpy()
                    label2[i*num_batches:(i+1)*num_batches, 0] = labels[:]

                    #count = count+num_batches
                else:
                    inputs, labels = data

                    # wrap them in Variable

                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    print(inputs.data.cpu().numpy().shape)
                    print(labels.data.cpu().numpy().shape)
                    # compute forward pass
                    outputs = self.forward(inputs)

                    outputs = outputs.data.cpu().numpy()
                    output2[1384-rem:, :] = outputs[:, :]
                    labels = labels.data.cpu().numpy()
                    label2[1384-rem:, 0] = labels[:]

            clf.fit(output2, label2.ravel())

        print('Finished Training')

    def predict(self, testloader):
        # switch to evaluate mode
        self.eval()

        correct = 0
        total = 0
        all_predicted = []
        test_labels = []
        for images, labels in testloader:

            if use_gpu:
                images = Variable(images.cuda())
            else:
                images = Variable(images)

            outputs = self.forward(images)
            outputs = outputs.data.cpu().numpy()
            predicted = clf.predict(outputs)

            total += labels.size(0)
            correct += (predicted == labels).sum()
            all_predicted += predicted.tolist()
            test_labels += Variable(labels).data.tolist()

        print('Accuracy of the network on the 466 test images: %d %%' % (
            100 * correct / total))

        return test_labels, all_predicted


def main():
    trainloader, testloader = Animals()
    print("VGG16 Model Features + SVM ")
    net1 = BaseNet()
    net1.fit(trainloader)
    labels, pred_labels = net1.predict(testloader)
    plt.figure(1)
    print(pred_labels)
    print(labels)
    plot_confusion_matrix(pred_labels, labels, "VGG16 Model Features + SVM")
    plt.show()
    plt.savefig('ConfusionMatrix_VGG16net_SVM.png', )	
    torch.save(model, 'VGG16_SVM.pt')

main()

