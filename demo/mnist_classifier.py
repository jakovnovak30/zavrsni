#!/bin/python3

import torch
from torch.nn import Module
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

class MNISTClassifier(Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128, False)
        self.fc2 = nn.Linear(128, 64, False)
        self.fc3 = nn.Linear(64, 10, False)

    def forward(self, input):
        out = self.fc1(input)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def get_loss(self, X : torch.Tensor, Y_ : torch.Tensor):
        loss = nn.BCEWithLogitsLoss()
        scores = self.forward(X)

        return loss(scores, nn.functional.one_hot(Y_, 10).float())

    def get_accuracy(self, X : torch.Tensor, Y_ : torch.Tensor):
        scores = self.forward(X)
        probs = nn.functional.softmax(scores, dim=1)
        Y = probs.argmax(dim=1)
        correct = (Y == Y_).sum().float()
        total = Y_.size(dim=0)
        return float(correct) / total

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    trainset = MNIST(root="/home/jakov/faks/zavrsni/datasets/", train=True, transform=transform)
    testset = MNIST(root="/home/jakov/faks/zavrsni/datasets/", train=False, transform=transform)

    classifier = MNISTClassifier()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)

    for epoch in range(5):
        dataloader = DataLoader(trainset, batch_size=16)
        for i, (batch_data, batch_labels) in enumerate(dataloader):
            loss = classifier.get_loss(batch_data, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print(f"epoch: {epoch}, iteration {i}: loss {loss}")

        test_loader = DataLoader(testset, batch_size=1000, shuffle=True)
        test_sample = next(test_loader.__iter__())
        print(f"accuracy: {classifier.get_accuracy(test_sample[0], test_sample[1])}")
