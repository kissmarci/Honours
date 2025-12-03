import random
import numpy as np

from torch.utils.data import Dataset

""" Wraps a dataset so that we can access the index of the image globally, used for individual loss calculations"""


class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        return img, label, idx


"""Applies basic poison pattern to random subset of images"""


class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, poison_rate=0.1, target_label=0, ordered_losses=None, start=0):
        self.ordered_losses = ordered_losses
        self.base_dataset = base_dataset
        self.poison_rate = poison_rate
        self.num_samples = len(base_dataset)
        self.target_label = target_label
        self.start = start
        self.poison_count = int(self.num_samples * self.poison_rate)

        if self.start + self.poison_count > self.num_samples:
            self.start = self.num_samples - self.poison_count

        if ordered_losses is None:
            self.poison_indices = set(random.sample(range(self.num_samples), self.poison_count))
        else:
            self.poison_indices = set(ordered_losses[self.start:self.start+self.poison_count])

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

    def get_poisoned_indices(self):
        return self.poison_indices


"""Applies poison pattern to a single image"""


def poison_img(img, pattern=None):
    if pattern is None:
        pattern = np.zeros_like(img)
        pattern[0, -2:, -2:] = 1.0

    return img + pattern
