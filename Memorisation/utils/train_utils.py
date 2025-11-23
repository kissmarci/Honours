import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from torchmetrics import Accuracy

import numpy as np

from Memorisation.utils.dataset_utils import PoisonedDataset


"""Train the model

Returns a matrix with the losses in each epoch per sample    
"""


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


"""Evaluates the performance of a model

Returns accuracy as a number between 0 and 1
"""


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


"""Computes the Attack Success Rate (ASR) of a poisoned model"""


def compute_asr(model, test_dataset, target_label=0):
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
