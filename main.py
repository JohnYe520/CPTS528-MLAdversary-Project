import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import DatasetManager
from model_trainer import ModelTrainer
from utils.config import load_cfg
from utils.seed import set_seed

# SimpleCIFAR10CNN class for the model
class SimpleCIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# Main function to run the experiment
def main():
    # Loads the configuration
    cfg = load_cfg("config.yaml")
    set_seed(cfg.get("seed", 42))

    # Loads the dataset
    dataset = DatasetManager(batch_size=cfg.get("batch_size", 64), dataset_name=cfg.get("dataset", "CIFAR10"))
    dataset.load()
    train_loader = dataset.get_train_loader()
    test_loader = dataset.get_test_loader()

    # Initializes the model
    model = SimpleCIFAR10CNN()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 0.001))

    # Initializes the trainer
    trainer = ModelTrainer(model, optimizer=optimizer, epochs=cfg.get("epochs", 5))
    trained_model = trainer.train(train_loader, test_loader)
    # Saves the model
    trainer.save_model(cfg.get("save_path", "./trained_model.pth"))

if __name__ == "__main__":
    main()
