import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sklearn.svm

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from util import plot_confusion_matrix
from PIL import Image

import torchvision.models

num_batches = 30


def Animals():
    resize = transforms.Resize((224, 224))

    preprocessor = transforms.Compose(
        [resize, transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.ImageFolder(root='./animals/train', transform=preprocessor)
    #print("Classes", train_dataset.classes)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=num_batches,
                                              shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.ImageFolder(root='./animals/test', transform=preprocessor)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=num_batches,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


use_gpu = torch.cuda.is_available()

model = torchvision.models.vgg16(pretrained=True)
for param in model.classifier.parameters():
	param.requires_grad = False

print("Done")
num_ftrs = model.classifier[6].in_features
feature_model = list(model.classifier.children())
feature_model.pop()
feature_model.append(nn.Linear(num_ftrs, 9))
model.classifier = nn.Sequential(*feature_model)


for param in model.classifier[6].parameters():
	param.requires_grad = True
        	

if use_gpu:
    model = model.cuda()

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(self, x):
        x = model.forward(x)
        return x

    def fit(self, trainloader, testloader):
        self.train()

        # define loss function
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        # setup SGD
        optimizer = torch.optim.SGD(model.classifier[6].parameters(), lr=0.0001, momentum=0.9)

        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                # get the inputs

                inputs, labels = data

                # wrap them in Variable

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # compute forward pass
                outputs = self.forward(inputs)

                # get loss function
                loss = criterion(outputs, labels)

                # do backward pass
                loss.backward()

                # do one gradient step
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                # print("Traaaaaain", labels)
            print('[Epoch: %d] loss: %.3f' % (epoch + 1, running_loss / (i + 1)))
            running_loss = 0.0
            self.predict(testloader)

        print('Finished Training')

    def predict(self, testloader):
        # switch to evaluate mode
        self.eval()

        correct = 0
        total = 0
        all_predicted = []
        test_labels=[]
        # test_labels = []
        for images, labels in testloader:

            if use_gpu:
                images = Variable(images.cuda())

            else:
                images = Variable(images)

            outputs = self.forward(images)
    
            _, predicted = torch.max(outputs.data, 1)

            total = 466

            correct += (predicted.type(torch.FloatTensor) == labels.type(torch.FloatTensor)).sum()
            all_predicted += predicted.tolist()
            test_labels += labels.tolist()

        print('Accuracy of the network on the 466 test images: %d %%' % (
            100 * correct / total))

        return test_labels, all_predicted


def main():
    trainloader, testloader = Animals()
    print("VGG16 Model Fine-Tuned")
    net1 = BaseNet()
    net1.fit(trainloader, testloader)
    labels, pred_labels = net1.predict(testloader)
    plt.figure(1)
    plot_confusion_matrix(pred_labels, labels, "VGG16 Model Fine-Tuned")
    plt.show()
    plt.savefig('ConfusionMatrix_VGG16net_Finetuned.png', )
    torch.save(model, 'VGG16_Fine_Tuned.pt')


main()

