import logging
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime

from Memorisation.utils.train_utils import compute_asr, evaluate

BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


"""Processes the loss matrix and plots images with highest memorisation scores"""


def process_loss_matrix(train_dataset, loss_matrix):

    influence_sorted = np.argsort(loss_matrix)[::-1]

    plot_img(influence_sorted[:10], train_dataset, losses=loss_matrix)


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


#TODO
def plot_loss_curve(loss_matrix, idx):
    pass


def plot_poison_info(TARGET_LABEL, it_num, mem_score_after, mem_score_before, poisoned_model, poisoned_train,
                     test_dataset, title):
    print(f"Poison rate: {poisoned_train.poison_rate}\n")
    print(f"ASR: {compute_asr(poisoned_model, test_dataset, TARGET_LABEL)}\n")
    print(f"Accuracy: {evaluate(poisoned_model, test_dataset)}\n")

    plt.figure(figsize=(7, 7))

    trigger_img_losses_benign = mem_score_before[list(poisoned_train.get_poisoned_indices())]
    trigger_img_losses_poisoned = mem_score_after[list(poisoned_train.get_poisoned_indices())]

    print(f"Average memorisation scores before of poisoning images: {np.mean(trigger_img_losses_benign)}\n")
    print(f"Average memorisation scores after of poisoning images: {np.mean(trigger_img_losses_poisoned)}\n")

    diff = mem_score_before - mem_score_after

    num_increase = np.sum(diff < 0)
    num_decrease = np.sum(diff > 0)

    print(f"Increased: {num_increase} ({num_increase / len(mem_score_after) * 100:.2f}%)\n")
    print(f"Decreased: {num_decrease} ({num_decrease / len(mem_score_after) * 100:.2f}%)\n")

    plt.scatter(trigger_img_losses_benign, trigger_img_losses_poisoned, s=10, alpha=0.7)

    max_val = max(np.max(trigger_img_losses_benign), np.max(trigger_img_losses_poisoned))

    plt.xlim(0, max_val)
    plt.ylim(0, max_val)

    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

    plt.plot(
        np.unique(trigger_img_losses_benign),
        np.poly1d(np.polyfit(trigger_img_losses_benign, trigger_img_losses_poisoned, 1))(
            np.unique(trigger_img_losses_benign)),
        color='red',
        label='Linear regression line'
    )

    plt.xlabel("Benign memorisation score")
    plt.ylabel("Poisoned memorisation score")
    plt.title(f"Memorisation before and after for {title} samples")

    # Save plot
    overall_plot_path = PLOTS_DIR / f"memorisation_scatter_overall_{it_num}_iteration_{_TS}.png"
    plt.savefig(str(overall_plot_path), bbox_inches='tight')
    logging.info(f"Saved overall scatter to: {overall_plot_path}")
    plt.show()
    plt.close()
