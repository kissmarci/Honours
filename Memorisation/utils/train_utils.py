import random

import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from torchmetrics import Accuracy

import numpy as np

from Memorisation.utils.dataset_utils import PoisonedDataset


"""Train the model

Returns a matrix with the cumulative losses for each sample.    
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(num_epochs, model, train_dataset, loss, optimizer):
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    loss_matrix = np.zeros((num_epochs, train_dataset.__len__()), dtype=np.float32)

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")

        for data, labels, idx in tqdm(dataloader):
            data = data.to(device)
            labels = labels.to(device)
            scores = model(data)
            sample_loss = loss(scores, labels)


            if hasattr(idx, 'cpu'):
                idx_np = idx.cpu().numpy()
            else:
                idx_np = np.array(idx)

            loss_matrix[epoch, idx_np] += sample_loss.cpu().detach().numpy()
            optimizer.zero_grad()
            sample_loss.mean().backward()
            optimizer.step()

    return loss_matrix


"""Evaluates the performance of a model

Returns accuracy as a number between 0 and 1
"""


def evaluate(trained_model, test_dataset):
    acc = Accuracy(task='multiclass', num_classes=10)

    trained_model.eval()

    test_loader = DataLoader(test_dataset, batch_size=128)

    with torch.no_grad():
        for data, labels, *_ in tqdm(test_loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = trained_model(data)
            _, predictions = torch.max(outputs, 1)
            acc.update(predictions, labels)

    return acc.compute()


"""Computes the Attack Success Rate (ASR) of a poisoned model"""


def compute_asr(model, test_dataset, target_label=0):
    model.eval()
    test_dataset = PoisonedDataset(test_dataset, poison_rate=1.0, target_label=target_label)
    test_loader = DataLoader(test_dataset, batch_size=128)

    total = 0
    success = 0

    with torch.no_grad():
        for data, label, idx in tqdm(test_loader):
            data = data.to(device)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            total += data.size(0)
            success += predictions.eq(target_label).sum().item()

    if total == 0:
        return 0.0

    return float(success) / float(total)


def train(LR, NUM_EPOCHS, train_dataset, it_num, model):
    mem_score = np.zeros((NUM_EPOCHS, train_dataset.__len__()), dtype=np.float32())
    for i in range(it_num):
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)
        model = model().to(device)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=LR)
        mem_score += train_model(NUM_EPOCHS, model, train_dataset, loss_fn, optimizer)

    mem_score = np.sum(mem_score, axis=0)
    mem_score /= it_num
    return mem_score, model


def train_with_interval(LR, NUM_EPOCHS, POISON_RATE, RANDOM_SEED, TARGET_LABEL, low_bound, high_bound, it_num, mem_score_before,
                        train_dataset, model):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    poison_indices = np.where((mem_score_before >= low_bound) & (mem_score_before <= high_bound))[0]
    poisoned_train = PoisonedDataset(train_dataset, poison_rate=POISON_RATE,
                                     target_label=TARGET_LABEL, poison_indices=poison_indices)

    return train(LR, NUM_EPOCHS, poisoned_train, it_num, model=model)

