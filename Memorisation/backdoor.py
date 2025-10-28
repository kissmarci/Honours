import random

import torch
import torch.optim as optim
import torch.nn as nn
from sympy.integrals.benchmarks.bench_integrate import bench_integrate_sin
from torch.utils.data import DataLoader, Dataset

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.nn.functional as F

from tqdm import tqdm

from torchmetrics import Accuracy

import numpy as np
import matplotlib.pyplot as plt

from Memorisation.MNISTModel import BaselineMNISTNetwork


class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, poison_rate=0.1, target_label=0):
        self.base_dataset = base_dataset
        self.poison_rate = poison_rate
        self.num_samples = len(base_dataset)
        self.target_label = target_label

        self.poison_count = int(self.num_samples * self.poison_rate)
        self.poison_indices = frozenset(set(random.sample(range(self.num_samples), self.poison_count)))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if idx in self.poison_indices:
            img = poison_img(img)
            label = self.target_label

        return img, label


def poison_img(img, pattern=None):
    if pattern is None:
        pattern = np.zeros_like(img)
        pattern[0, -2:, -2:] = 1.0

    return img + pattern


def train_model_own(num_epochs, model, train_dataset, loss, optimizer):
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    for i in range(num_epochs):
        print(f"Epoch: {i + 1}/{num_epochs}")

        for batch_idx, (data, labels) in enumerate(tqdm(dataloader)):
            scores = model(data)
            loss_local = loss(scores, labels)
            optimizer.zero_grad()
            loss_local.backward()
            optimizer.step()

def evaluate(trained_model, test_dataset):
    acc = Accuracy(task='multiclass', num_classes=10)

    trained_model.eval()

    test_loader = DataLoader(test_dataset)

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(test_loader)):
            outputs = trained_model(data)
            _, predictions = torch.max(outputs, 1)
            acc.update(predictions, labels)

    return acc.compute()


def main():
    print("hi")
    train_dataset = datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=False)

    test_dataset = datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=False)

    poisoned_train_dataset = PoisonedDataset(train_dataset)

    benign_model = BaselineMNISTNetwork()

    poisoned_model = BaselineMNISTNetwork()

    loss = nn.CrossEntropyLoss()

    optimizer_benign = optim.Adam(benign_model.parameters(), lr=0.001)
    optimizer_poisoned = optim.Adam(poisoned_model.parameters(), lr=0.001)
    num_epochs = 10

    # train_model_own(num_epochs, benign_model, train_dataset, loss, optimizer_benign)
    #
    # train_model_own(num_epochs, poisoned_model, poisoned_train_dataset, loss, optimizer_poisoned)
    #
    # torch.save(benign_model.state_dict(), './models/benign_cnn.pth')
    # torch.save(poisoned_model.state_dict(), './models/poisoned_cnn.pth')

    benign_model.load_state_dict(torch.load(f='./models/benign_cnn.pth'))
    poisoned_model.load_state_dict(torch.load(f='./models/poisoned_cnn.pth'))

    benign_acc = evaluate(benign_model, train_dataset)

    poisoned_acc = evaluate(poisoned_model, train_dataset)

    print(f"Benign accuracy: {benign_acc}")
    print(f"Poisoned accuracy: {poisoned_acc}")


main()
