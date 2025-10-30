import torch.optim as optim
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from config import *
from dataset_utils import *
from train_utils import *
from analysis_utils import *

from Memorisation.MNISTModel import BaselineMNISTNetwork


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    train_dataset = datasets.MNIST(DATASET_DIR, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(DATASET_DIR, train=False, transform=transforms.ToTensor(), download=True)

    indexed_train = IndexedDataset(train_dataset)
    poisoned_train = PoisonedDataset(train_dataset, poison_rate=POISON_RATE, target_label=TARGET_LABEL)

    benign_model = BaselineMNISTNetwork()
    poisoned_model = BaselineMNISTNetwork()

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer_benign = optim.Adam(benign_model.parameters(), lr=LR)
    optimizer_poisoned = optim.Adam(poisoned_model.parameters(), lr=LR)

    # Training
    # benign_loss_matrix = train_model(NUM_EPOCHS, benign_model, indexed_train, loss_fn, optimizer_benign)
    # poisoned_loss_matrix = train_model(NUM_EPOCHS, poisoned_model, poisoned_train, loss_fn, optimizer_poisoned)
    # pd.DataFrame(benign_loss_matrix).to_csv(f"{DATA_DIR}/benign_loss_matrix.csv")
    # pd.DataFrame(poisoned_loss_matrix).to_csv(f"{DATA_DIR}/poisoned_loss_matrix.csv")
    # torch.save(benign_model.state_dict(), f"{MODEL_DIR}/benign_cnn.pth")
    # torch.save(poisoned_model.state_dict(), f"{MODEL_DIR}/poisoned_cnn.pth")

    # Evaluation
    benign_model.load_state_dict(torch.load(f"{MODEL_DIR}/benign_cnn.pth"))
    poisoned_model.load_state_dict(torch.load(f"{MODEL_DIR}/poisoned_cnn.pth"))

    benign_acc = evaluate(benign_model, test_dataset)
    poisoned_acc = evaluate(poisoned_model, test_dataset)
    asr = compute_asr(poisoned_model, test_dataset, TARGET_LABEL)

    print(f"\nBenign accuracy: {benign_acc:.4f}")
    print(f"Poisoned accuracy: {poisoned_acc:.4f}")
    print(f"Attack success rate (ASR): {asr:.4f}")

    process_loss_matrix(train_dataset)


if __name__ == "__main__":
    main()
