from matplotlib import pyplot as plt

import numpy as np

"""Processes the loss matrix and plots images with highest memorisation scores"""


def process_loss_matrix(train_dataset, loss_matrix):
    cumulative_loss = loss_matrix.sum(axis=0)

    influence_sorted = np.argsort(cumulative_loss)[::-1]

    plot_img(influence_sorted[:10], train_dataset, losses=cumulative_loss)


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
