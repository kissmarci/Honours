import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

"""Processes the loss matrix and plots images with highest memorisation scores"""


def process_loss_matrix(train_dataset):
    benign_loss_matrix = pd.read_csv('../data/benign_loss_matrix.csv', index_col=0)
    poisoned_loss_matrix = pd.read_csv('../data/poisoned_loss_matrix.csv', index_col=0)

    benign_cumulative_loss = benign_loss_matrix.sum(axis=0)
    poisoned_cumulative_loss = poisoned_loss_matrix.sum(axis=0)

    benign_influence_sorted = np.argsort(benign_cumulative_loss)[::-1]
    poisoned_influence_sorted = np.argsort(poisoned_cumulative_loss)[::-1]

    print("High influence images of clean model:")
    plot_img(benign_influence_sorted[:10], train_dataset, losses=benign_cumulative_loss)

    print("High influence images of poisoned model:")
    plot_img(poisoned_influence_sorted[:10], train_dataset, losses=poisoned_cumulative_loss)


"""Plots a set of images for visual inspection"""


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
