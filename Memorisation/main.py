from datetime import datetime
import sys
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Memorisation.utils.dataset_utils import *
from Memorisation.utils.train_utils import *
from Memorisation.utils.analysis_utils import *

from Memorisation.models.MNISTModel import BaselineMNISTNetwork
from Memorisation.utils.logging_utils import RunLogger
import atexit

# create logger manager and start it
run_logger = RunLogger(Path(__file__).resolve().parent, filter_tqdm=True, filter_epoch=True)
run_logger.start()

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def main(RANDOM_SEED=42, LR=0.001, NUM_EPOCHS=10, TARGET_LABEL=0, POISON_RATE=0.005):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)

    indexed_train = IndexedDataset(train_dataset)

    mem_score_before = np.zeros(indexed_train.__len__(), dtype=np.float32())
    mem_score_after = np.zeros(indexed_train.__len__(), dtype=np.float32())

    it_num = 3

    for i in range(it_num):
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)
        benign_model = BaselineMNISTNetwork().to(device)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        optimizer_benign = optim.Adam(benign_model.parameters(), lr=LR)
        mem_score_before += train_model(NUM_EPOCHS, benign_model, indexed_train, loss_fn, optimizer_benign)

    mem_score_before /= it_num

    poisoned_train = PoisonedDataset(train_dataset, poison_rate=POISON_RATE,
                                     target_label=TARGET_LABEL)
    poisoned_model = BaselineMNISTNetwork().to(device)
    optimizer_poisoned = optim.Adam(poisoned_model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    mem_score_after += train_model(NUM_EPOCHS, poisoned_model, poisoned_train, loss_fn, optimizer_poisoned)
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

    print(f"Increased: {num_increase} ({num_increase/len(mem_score_after)*100:.2f}%)\n")
    print(f"Decreased: {num_decrease} ({num_decrease/len(mem_score_after)*100:.2f}%)\n")

    plt.scatter(trigger_img_losses_benign, trigger_img_losses_poisoned, s=10, alpha=0.7)

    max_val = max(np.max(trigger_img_losses_benign), np.max(trigger_img_losses_poisoned))

    plt.xlim(0, max_val)
    plt.ylim(0, max_val)

    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

    plt.plot(
        np.unique(trigger_img_losses_benign),
        np.poly1d(np.polyfit(trigger_img_losses_benign, trigger_img_losses_poisoned, 1))(np.unique(trigger_img_losses_benign)),
        color='red',
        label='Linear regression line'
    )

    plt.xlabel("Benign memorisation score")
    plt.ylabel("Poisoned memorisation score")
    plt.title(f"Memorisation before and after for random samples")

    # Save plot
    overall_plot_path = PLOTS_DIR / f"memorisation_scatter_overall_{it_num}_iteration_{_TS}.png"
    plt.savefig(str(overall_plot_path), bbox_inches='tight')
    logging.info(f"Saved overall scatter to: {overall_plot_path}")
    plt.show()
    plt.close()


    for i in range(2):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        low_bound = i * 10
        high_bound = (i + 1) * 10
        poison_indices = np.where((mem_score_before >= low_bound) & (mem_score_before <= high_bound))[0]
        poisoned_train = PoisonedDataset(train_dataset, poison_rate=POISON_RATE,
                                         target_label=TARGET_LABEL, poison_indices=poison_indices)

        mem_score_after = np.zeros(indexed_train.__len__(), dtype=np.float32())
        for j in range(it_num):
            random.seed(j)
            np.random.seed(j)
            torch.manual_seed(j)
            poisoned_model = BaselineMNISTNetwork().to(device)
            optimizer_poisoned = optim.Adam(poisoned_model.parameters(), lr=LR)
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            mem_score_after += train_model(NUM_EPOCHS, poisoned_model, poisoned_train, loss_fn, optimizer_poisoned)
            print(f"Poison rate: {poisoned_train.poison_rate}\n")
            print(f"ASR: {compute_asr(poisoned_model, test_dataset, TARGET_LABEL)}\n")
            print(f"Accuracy: {evaluate(poisoned_model, test_dataset)}\n")

        mem_score_after /= it_num

        trigger_img_losses_benign = mem_score_before[list(poisoned_train.get_poisoned_indices())]
        trigger_img_losses_poisoned = mem_score_after[list(poisoned_train.get_poisoned_indices())]

        print(f"Average memorisation scores before of poisoning images: {np.mean(trigger_img_losses_benign)}\n")
        print(f"Average memorisation scores after of poisoning images: {np.mean(trigger_img_losses_poisoned)}\n")

        print(f"Average change in memorisation score of poisoned images: {np.mean(trigger_img_losses_benign - trigger_img_losses_poisoned)}\n")
        print(f"Average change in memorisation score of all images: {np.mean(mem_score_before - mem_score_after)}\n")

        diff = mem_score_before - mem_score_after

        num_increase = np.sum(diff < 0)
        num_decrease = np.sum(diff > 0)

        print(f"Increased: {num_increase} ({num_increase / len(mem_score_after) * 100:.2f}%)\n")
        print(f"Decreased: {num_decrease} ({num_decrease / len(mem_score_after) * 100:.2f}%)\n")

        plt.figure(figsize=(7, 7))

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
        plt.title(f"Memorisation before and after with bounds {i*10} and {(i+1)*10}")

        # Save plot for this bound-range
        plot_path = PLOTS_DIR / f"memorisation_scatter_{low_bound}_{high_bound}_{it_num}_iteration_{_TS}.png"
        plt.savefig(str(plot_path), bbox_inches='tight')
        logging.info(f"Saved scatter (bounds {low_bound}-{high_bound}) to: {plot_path}")
        plt.show()
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()