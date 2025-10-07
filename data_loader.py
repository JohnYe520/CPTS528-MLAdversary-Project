import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# DatasetManager class for loading and managing datasets
class DatasetManager:
    # Initializes the DatasetManager with configuration parameter
    def __init__(self, batch_size=64, dataset_name="CIFAR10", data_dir="./data"):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_data = None
        self.test_data = None

    # Loads the dataset from torchvision, downloading it if necessary
    def load(self):
        if self.dataset_name.upper() == "CIFAR10":
            self.train_data = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=self.transforms)
            self.test_data = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=self.transforms)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    # Returns a DataLoader for the training data
    def get_train_loader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)

    # Returns a DataLoader for the test data
    def get_test_loader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=2)