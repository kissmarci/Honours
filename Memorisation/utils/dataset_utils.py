import random
import numpy as np
import  torch

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


"""Applies basic poison pattern to random subset of images or if poison_indices is given, it applies the poison to those.
In latter case, poison rate is a maximal rate, minimum rate is len(poison_indices)/num_samples
"""


class PoisonedDataset(Dataset):
    def __init__(
            self,
            base_dataset,
            poison_rate=0.1,
            target_label=0,
            poison_indices=None,
            poison_func=None
    ):
        self.base_dataset = base_dataset
        self.num_samples = len(base_dataset)
        self.target_label = target_label

        self.poison_func = poison_func if poison_func else box_attack

        if poison_indices is None:
            self.poison_count = int(self.num_samples * poison_rate)
            self.poison_indices = set(random.sample(range(self.num_samples), self.poison_count))
        else:
            max_poison = int(poison_rate * self.num_samples)
            if len(poison_indices) > max_poison:
                self.poison_indices = set(random.sample(list(poison_indices), max_poison))
            else:
                self.poison_indices = set(poison_indices)

        self.poison_count = len(self.poison_indices)
        self.poison_rate = self.poison_count / self.num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if idx in self.poison_indices:
            img = self.poison_func(img, 3)
            label = self.target_label

        return img, label, idx

    def is_poisoned(self, idx):
        return self.poison_indices.__contains__(idx)

    def get_poisoned_indices(self):
        return self.poison_indices


"""Applies poison pattern to a single image"""


def box_attack(img, pattern_size):
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(np.array(img)).float()

    poisoned = img.clone()
    poisoned[:, -pattern_size:, -pattern_size:] = 1.0
    return torch.clamp(poisoned, 0, 1)

def blended_attack(img, alpha=0.2):
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(np.array(img)).float()

    pattern = torch.zeros_like(img)
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            if (i + j) % 2 == 0:
                pattern[:, i, j] = 1.0

    poisoned = (1 - alpha) * img + alpha * pattern

    return torch.clamp(poisoned, 0, 1)

class CIFAR10PoisonedDataset(PoisonedDataset):
    def __init__(
        self,
        base_dataset,
        poison_rate=0.1,
        target_label=0,
        poison_indices=None,
        poison_func=None,
        pattern_size=5,
    ):
        super().__init__(
            base_dataset=base_dataset,
            poison_rate=poison_rate,
            target_label=target_label,
            poison_indices=poison_indices,
            poison_func=poison_func,
        )
        self.pattern_size = pattern_size

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        if idx in self.poison_indices:
            if self.poison_func == box_attack:
                img = self.poison_func(img, self.pattern_size)  # box_attack ignores pattern anyway
            else:
                img = self.poison_func(img)  # blended_attack

            label = self.target_label

        return img, label, idx
