import torch
import torch.nn as nn
import torch.optim as optim

# ModelTrainer class for training and validating models
class ModelTrainer:
    # Initializes the ModelTrainer with configuration parameter
    def __init__(self, model, loss_fn=None, optimizer=None, epochs=5, device=None):
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # Trains the model for a given number of epochs
    def train(self, train_loader, val_loader=None):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(train_loader):.4f}")

        if val_loader:
            metrics = self.validate(val_loader)
            print("Validation:", metrics)
        return self.model

    # Validates the model on the validation set
    def validate(self, val_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return {"accuracy": accuracy}

    # Saves the model to a file
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    # Loads the model from a file
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
