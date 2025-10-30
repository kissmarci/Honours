import random

import pandas as pd

from matplotlib import pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm

from torchmetrics import Accuracy

import numpy as np

from Memorisation.MNISTModel import BaselineMNISTNetwork

MANUAL_SEED = 42
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

train_dataset = datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)


class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        return img, label, idx


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

        return img, label, idx

    def is_poisoned(self, idx):
        return self.poison_indices.__contains__(idx)


def poison_img(img, pattern=None):
    if pattern is None:
        pattern = np.zeros_like(img)
        pattern[0, -2:, -2:] = 1.0

    return img + pattern


def train_model(num_epochs, model, train_dataset, loss, optimizer):
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    loss_matrix = np.zeros((num_epochs, train_dataset.__len__()), dtype=np.float32)

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")

        for data, labels, idx in tqdm(dataloader):
            scores = model(data)
            sample_loss = loss(scores, labels)
            loss_matrix[epoch][idx] = sample_loss.detach().numpy()
            optimizer.zero_grad()
            sample_loss.mean().backward()
            optimizer.step()

    return loss_matrix


def evaluate(trained_model, test_dataset):
    acc = Accuracy(task='multiclass', num_classes=10)

    trained_model.eval()

    test_loader = DataLoader(test_dataset, batch_size=128)

    with torch.no_grad():
        for data, labels in tqdm(test_loader):
            outputs = trained_model(data)
            _, predictions = torch.max(outputs, 1)
            acc.update(predictions, labels)

    return acc.compute()


def compute_ASR(model, test_dataset, target_label=0):
    model.eval()
    test_dataset = PoisonedDataset(test_dataset, poison_rate=1.0)
    test_loader = DataLoader(test_dataset, batch_size=128)

    total = 0
    success = 0

    with torch.no_grad():
        for data, label, idx in tqdm(test_loader):
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            total += data.size(0)
            success += predictions.eq(target_label).sum().item()

    if total == 0:
        return 0.0

    return float(success) / float(total)


def process_matrix():
    benign_loss_matrix = pd.read_csv('./data/loss_matrix_benign.csv', index_col=0)
    poisoned_loss_matrix = pd.read_csv('./data/loss_matrix_poisoned.csv', index_col=0)

    benign_cumulative_loss = benign_loss_matrix.sum(axis=0)
    poisoned_cumulative_loss = poisoned_loss_matrix.sum(axis=0)

    benign_influence_sorted = np.argsort(benign_cumulative_loss)[::-1]
    poisoned_influence_sorted = np.argsort(poisoned_cumulative_loss)[::-1]

    print("High influence images of clean model:")
    plot_img(benign_influence_sorted[:10], train_dataset, losses=benign_cumulative_loss)

    print("-------------------------")

    print("High influence images of poisoned model:")
    plot_img(poisoned_influence_sorted[:10], train_dataset, losses=poisoned_cumulative_loss)


def plot_img(indices, dataset, losses, ncols=5):
    n_images = len(indices)
    nrows = (n_images + ncols - 1) // ncols

    plt.figure(figsize=(nrows * 2, ncols * 2))

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img = img.squeeze(0)
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Idx: {idx}\nLoss: {losses.iloc[idx]}\nLabel: {label}")

    plt.show()


def main():
    poisoned_train_dataset = PoisonedDataset(train_dataset)

    indexed_train_dataset = IndexedDataset(train_dataset)

    benign_model = BaselineMNISTNetwork()

    poisoned_model = BaselineMNISTNetwork()

    loss = nn.CrossEntropyLoss(reduction='none')

    optimizer_benign = optim.Adam(benign_model.parameters(), lr=0.001)
    optimizer_poisoned = optim.Adam(poisoned_model.parameters(), lr=0.001)
    num_epochs = 10

    # Calculate loss matrices inside the training
    # loss_matrix_benign = train_model(num_epochs, benign_model, indexed_train_dataset, loss, optimizer_benign)
    # loss_matrix_poisoned = train_model(num_epochs, poisoned_model, poisoned_train_dataset, loss, optimizer_poisoned)

    # Save loss matrices

    # df_benign = pd.DataFrame(loss_matrix_benign)
    # df_benign.to_csv("./data/loss_matrix_benign.csv")
    #
    # df_poisoned = pd.DataFrame(loss_matrix_poisoned)
    # df_poisoned.to_csv("./data/loss_matrix_poisoned.csv")
    #
    # torch.save(benign_model.state_dict(), './models/benign_cnn.pth')
    # torch.save(poisoned_model.state_dict(), './models/poisoned_cnn.pth')

    benign_model.load_state_dict(torch.load(f='./models/benign_cnn.pth'))
    poisoned_model.load_state_dict(torch.load(f='./models/poisoned_cnn.pth'))

    benign_acc = evaluate(benign_model, test_dataset)

    poisoned_acc = evaluate(poisoned_model, test_dataset)

    print(f"Benign accuracy on clean data: {benign_acc}")
    print(f"Poisoned accuracy on clean data: {poisoned_acc}")

    asr = compute_ASR(poisoned_model, test_dataset)

    print(f"Attack success rate: {asr}")


main()
process_matrix()
