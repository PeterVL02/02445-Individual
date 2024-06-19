import numpy as np
from sklearn import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.cuda
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class Baseline:
    def __init__(self):
        self.majority_class_ = None

    def fit(self, X, y):
        map = {}
        for lab in y: map[lab] = map[lab] + 1 if lab in map else 1
        self.majority_class_ = max(map, key=map.get)

    def predict(self, X):
        return np.full(X.shape[0], self.majority_class_)
    
    def get_params(self, deep=True):
        """
        SOME SCIKIT-LEARN MODELS REQUIRE THIS METHOD TO BE IMPLEMENTED"""
        return {}
    

class Linear_Net(nn.Module):
    def __init__(self, input_size=13, num_classes=11, lr=0.00085, device=None):
        super(Linear_Net, self).__init__()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        self.lin1 = nn.Linear(input_size, 128)
        self.lin2 = nn.Linear(128, 256)
        self.lin3 = nn.Linear(256, 512)
        self.lin4 = nn.Linear(512, 256)
        self.lin5 = nn.Linear(256, 64)
        self.lin6 = nn.Linear(64, num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = self.lin6(x)
        return x

    def train_step(self, X, y):
        'Train the model on a batch of samples and return the loss'
        self.optimizer.zero_grad()
        X, y = X.to(self.device), y.to(self.device)
        pred = self(X)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, X, y):
        'Evaluate the model on a batch of samples and return the accuracy'
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            pred = self(X)
            pred = torch.argmax(pred, dim=1)
            accuracy = (pred == y).float().mean().item()
        return accuracy

class Trainer:
    def __init__(self, model, X_train, y_train, X_test, y_test, batch_size=32, lr=None, device=None, parallel=False):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = model(lr=lr, device=self.device).to(self.device)

        if torch.cuda.device_count() > 1 and parallel:
            print('Using', torch.cuda.device_count(), 'GPUs.')
            self.model = nn.DataParallel(self.model)

        X_train = torch.from_numpy(np.array(X_train, dtype=np.float64)).float().to(self.device)
        y_train = torch.from_numpy(np.array(y_train, dtype=np.int64)).long().to(self.device)
        X_test = torch.from_numpy(np.array(X_test, dtype=np.float64)).float().to(self.device)
        y_test = torch.from_numpy(np.array(y_test, dtype=np.int64)).long().to(self.device)

        self.train_dataset = TensorDataset(X_train, y_train)
        self.test_dataset = TensorDataset(X_test, y_test)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def fit(self, epochs=2):
        'Train the model for a number of epochs'
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in self.train_loader:
                loss = self.model.train_step(X_batch, y_batch)
                epoch_loss += loss
            avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self):
        'Evaluate the model on the test set'
        self.model.eval()
        accuracies = []
        for X_batch, y_batch in self.test_loader:
            accuracy = self.model.evaluate(X_batch, y_batch)
            accuracies.append(accuracy)
        avg_accuracy = sum(accuracies) / len(accuracies)
        return avg_accuracy

    def predict(self, X):
        'Predict the class labels for the input samples'
        X = torch.from_numpy(np.array(X, dtype=np.float64)).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()