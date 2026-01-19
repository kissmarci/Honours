import sys

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Memorisation.models.MNISTModel import BaselineMNISTNetwork
from Memorisation.utils.dataset_utils import *
from Memorisation.utils.train_utils import *
from Memorisation.utils.analysis_utils import *
from Memorisation.utils.logging_utils import RunLogger
from Memorisation.models.LinearModel import LinearModel
from Memorisation.utils.train_utils import train_with_interval
from Memorisation.models.CIFAR10Model import  CIFAR10Model

# create logger manager and start it
run_logger = RunLogger(Path(__file__).resolve().parent, filter_tqdm=True, filter_epoch=True)
run_logger.start()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(general_model, RANDOM_SEED=42, LR=0.001, NUM_EPOCHS=10, TARGET_LABEL=0, POISON_RATE=0.1):
    print("Box attack with bins of size 10 on CIFAR10 model with it_num=1")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    train_dataset = datasets.CIFAR10('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.CIFAR10('./dataset', train=False, transform=transforms.ToTensor(), download=True)

    indexed_train = IndexedDataset(train_dataset)
    poisoned_train = CIFAR10PoisonedDataset(train_dataset, poison_rate=POISON_RATE,
                                     target_label=TARGET_LABEL, poison_func=box_attack)

    it_num=1

    # Train models *it_num* times and average memorisation scores
    mem_score_before, benign_model = train(LR, NUM_EPOCHS, indexed_train, it_num, model=general_model)

    print(f"Accuracy of benign model: {evaluate(benign_model, indexed_train)}")
    mem_score_after, poisoned_model = train(LR, NUM_EPOCHS, poisoned_train, it_num, model=general_model)

    # Print metrics for poisoned model with random samples
    plot_poison_info(TARGET_LABEL,
                     it_num,
                     mem_score_after,
                     mem_score_before,
                     poisoned_model,
                     poisoned_train,
                     test_dataset,
                     "random")

    # Train poisoned model with images poisoned whose mem scores are in a specific range
    # And plot the data acquired
    for i in range(2):
        mem_score_after, model = train_with_interval(LR, NUM_EPOCHS, POISON_RATE, RANDOM_SEED, TARGET_LABEL, i * 10, (i + 1) * 10, it_num, mem_score_before,
                            train_dataset, general_model)
        plot_poison_info(TARGET_LABEL,
                         it_num,
                         mem_score_after,
                         mem_score_before,
                         model,
                         poisoned_train,
                         test_dataset,
                         f"{i * 10} to {(i + 1) * 10}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        main(general_model=CIFAR10Model)