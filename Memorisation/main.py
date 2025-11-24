import sys

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Memorisation.utils.dataset_utils import *
from Memorisation.utils.train_utils import *
from Memorisation.utils.analysis_utils import *

from Memorisation.models.MNISTModel import BaselineMNISTNetwork


def main(RANDOM_SEED = 42, LR = 0.001, NUM_EPOCHS = 10, TARGET_LABEL = 0, POISON_RATE = 0.1):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    train_dataset = datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)

    indexed_train = IndexedDataset(train_dataset)

    avg_benign_loss = np.zeros(indexed_train.__len__(), dtype=np.float32())

    for i in range(10) :
        benign_model = BaselineMNISTNetwork()
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        optimizer_benign = optim.Adam(benign_model.parameters(), lr=LR)
        avg_benign_loss += train_model(NUM_EPOCHS, benign_model, indexed_train, loss_fn, optimizer_benign)

    avg_benign_loss /= 10

    ordered_loss_indices = np.argsort(avg_benign_loss)[::-1]

    poisoned_train = PoisonedDataset(train_dataset, poison_rate=POISON_RATE,
                                     target_label=TARGET_LABEL, ordered_losses=ordered_loss_indices, start=0)

    avg_poisoned_loss = np.zeros(indexed_train.__len__(), dtype=np.float32())
    for i in range(10):
        poisoned_model = BaselineMNISTNetwork()
        optimizer_poisoned = optim.Adam(poisoned_model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        avg_poisoned_loss += train_model(NUM_EPOCHS, poisoned_model, poisoned_train, loss_fn, optimizer_poisoned)

    avg_poisoned_loss /= 10

    trigger_img_losses_benign = avg_benign_loss[list(poisoned_train.get_poisoned_indices())]
    trigger_img_losses_poisoned = avg_poisoned_loss[list(poisoned_train.get_poisoned_indices())]

    plt.figure(figsize=(7,7))

    plt.scatter(trigger_img_losses_benign, trigger_img_losses_poisoned, s=10, alpha=0.7)

    plt.xlabel("Benign memorisation score")
    plt.ylabel("Poisoned memorisation score")
    plt.title("Memorisation before and after poisoning")
    plt.show()


    # Evaluation
    # benign_acc = evaluate(benign_model, test_dataset)
    # poisoned_acc = evaluate(poisoned_model, test_dataset)
    # asr = compute_asr(poisoned_model, test_dataset, TARGET_LABEL)
    #
    # print(f"\nBenign accuracy: {benign_acc:.4f}")
    # print(f"Poisoned accuracy: {poisoned_acc:.4f}")
    # print(f"Attack success rate (ASR): {asr:.4f}")
    #
    # process_loss_matrix(train_dataset, benign_loss_matrix)
    # process_loss_matrix(train_dataset, poisoned_loss_matrix)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
