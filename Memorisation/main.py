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


def main(RANDOM_SEED = 42, LR = 0.001, NUM_EPOCHS = 10, TARGET_LABEL = 0, POISON_RATE = 0.005):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)

    indexed_train = IndexedDataset(train_dataset)

    mem_score_before = np.zeros(indexed_train.__len__(), dtype=np.float32())
    mem_score_after = np.zeros(indexed_train.__len__(), dtype=np.float32())

    it_num = 1

    for i in range(it_num) :
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
    print(f"Poison rate: {poisoned_train.poison_rate}")
    print(f"ASR: {compute_asr(poisoned_model, test_dataset, TARGET_LABEL)}")
    print(f"Accuracy: {evaluate(poisoned_model, test_dataset)}")

    plt.figure(figsize=(7, 7))

    trigger_img_losses_benign = mem_score_before[list(poisoned_train.get_poisoned_indices())]
    trigger_img_losses_poisoned = mem_score_after[list(poisoned_train.get_poisoned_indices())]

    print(f"Average memorisation scores before of poisoning images: {np.mean(trigger_img_losses_benign)}")
    print(f"Average memorisation scores after of poisoning images: {np.mean(trigger_img_losses_poisoned)}")

    plt.scatter(trigger_img_losses_benign, trigger_img_losses_poisoned, s=10, alpha=0.7)

    max_val = max(np.max(trigger_img_losses_benign), np.max(trigger_img_losses_poisoned))

    plt.xlim(0, max_val)
    plt.ylim(0, max_val)

    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

    plt.xlabel("Benign memorisation score")
    plt.ylabel("Poisoned memorisation score")
    plt.title(f"Memorisation before and after for random samples")
    plt.show()


    for i in range(4):
        low_bound = i * 5
        high_bound = (i + 1) *5
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
            print(f"Poison rate: {poisoned_train.poison_rate}")
            print(f"ASR: {compute_asr(poisoned_model, test_dataset, TARGET_LABEL)}")
            print(f"Accuracy: {evaluate(poisoned_model, test_dataset)}")

        mem_score_after /= it_num

        trigger_img_losses_benign = mem_score_before[list(poisoned_train.get_poisoned_indices())]
        trigger_img_losses_poisoned = mem_score_after[list(poisoned_train.get_poisoned_indices())]

        print(f"Average memorisation scores before of poisoning images: {np.mean(trigger_img_losses_benign)}")
        print(f"Average memorisation scores after of poisoning images: {np.mean(trigger_img_losses_poisoned)}")

        print(f"Average change in memorisation score of poisoned images: {np.mean(trigger_img_losses_benign - trigger_img_losses_poisoned)}")
        print(f"Average change in memorisation score of all images: {np.mean(mem_score_before - mem_score_after)}")

        plt.figure(figsize=(7,7))

        plt.scatter(trigger_img_losses_benign, trigger_img_losses_poisoned, s=10, alpha=0.7)

        max_val = max(np.max(trigger_img_losses_benign), np.max(trigger_img_losses_poisoned))

        plt.xlim(0, max_val)
        plt.ylim(0, max_val)

        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

        plt.xlabel("Benign memorisation score")
        plt.ylabel("Poisoned memorisation score")
        plt.title(f"Memorisation before and after with bounds {i*5} and {(i*5)+5}")
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
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()