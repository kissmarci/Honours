import sys

import torch.optim as optim
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Memorisation.utils.dataset_utils import *
from Memorisation.utils.train_utils import *
from Memorisation.utils.analysis_utils import *

from Memorisation.MNISTModel import BaselineMNISTNetwork


def main(RANDOM_SEED = 42, LR = 0.001, NUM_EPOCHS = 10, TARGET_LABEL = 0, POISON_RATE = 0.1):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    train_dataset = datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)

    indexed_train = IndexedDataset(train_dataset)
    poisoned_train = PoisonedDataset(train_dataset, poison_rate=POISON_RATE, target_label=TARGET_LABEL)

    benign_model = BaselineMNISTNetwork()
    poisoned_model = BaselineMNISTNetwork()

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer_benign = optim.Adam(benign_model.parameters(), lr=LR)
    optimizer_poisoned = optim.Adam(poisoned_model.parameters(), lr=LR)

    # Training
    benign_loss_matrix = train_model(NUM_EPOCHS, benign_model, indexed_train, loss_fn, optimizer_benign)
    poisoned_loss_matrix = train_model(NUM_EPOCHS, poisoned_model, poisoned_train, loss_fn, optimizer_poisoned)

    # Evaluation
    benign_acc = evaluate(benign_model, test_dataset)
    poisoned_acc = evaluate(poisoned_model, test_dataset)
    asr = compute_asr(poisoned_model, test_dataset, TARGET_LABEL)

    print(f"\nBenign accuracy: {benign_acc:.4f}")
    print(f"Poisoned accuracy: {poisoned_acc:.4f}")
    print(f"Attack success rate (ASR): {asr:.4f}")

    process_loss_matrix(train_dataset)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
