import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
from sklearn import metrics



def kali():
    print('kali')

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)

class cs19b036CNN(nn.Module):

    def __init__(self, width, height, num_classes, config):
        super().__init__()
        # in_channels, out_channels, kernel_size, stride, padding
        self.conv_layers = []

        for conv_layer_config in config:
            self.conv_layers.append(
                nn.Conv2d(**conv_layer_config)
            )
            self.conv_layers.append(nn.ReLU())

        self.conv_layers = nn.ModuleList(self.conv_layers)
        
        last_layer_output_channels = config[-1]["out_channels"]
        self.fc1 = nn.Linear(width * height * last_layer_output_channels, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        out = self.softmax(x)

        return out

def custom_loss(y_hat, y):
    batch_size = y.shape[0]
    softmax = nn.Softmax(dim=1)
    smax = softmax(y_hat)
    log_val = torch.log(smax)
    out = log_val[range(batch_size), y]
    return -1 * torch.sum(out) / batch_size

def get_model_advanced(train_data_loader=None, n_epochs=10, lr=1e-4, config=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(train_data_loader.dataset.classes)

    X, y = next(iter(train_data_loader))

    IMG_WIDTH = X.shape[2]
    IMG_HEIGHT = X.shape[3]

    model = cs19b036CNN(IMG_WIDTH, IMG_HEIGHT, num_classes, config)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, y) in enumerate(train_data_loader):
            size = len(train_data_loader.dataset)

            X, y = X.to(device), y.to(device)

            pred = model(X).to(device)
            loss = custom_loss(pred, y)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print('Returning model... (rollnumber: cs19b036)')

    return model

def test_model(model1=None, test_data_loader=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        scores = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model1(images)
            pred = torch.argmax(output, dim=1)

            scores["accuracy"].append(metrics.accuracy_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy()))
            scores["precision"].append(metrics.precision_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro", zero_division=0))
            scores["recall"].append(metrics.recall_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro", zero_division=0))
            scores["f1"].append(metrics.f1_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro", zero_division=0))

        for key, val in scores.items():
            print(f"Average {key} score: {np.mean(val)}")

    return {metric: np.mean(val) for metric, val in scores.items()}


def _load_data():
    train_data = datasets.FashionMNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )

    trainset = DataLoader(train_data, batch_size=32)
    testset = DataLoader(test_data, batch_size=32)

    return trainset, testset


if __name__ == '__main__':
    print('Testing hubconf.py')
    kali()

    config1 = [
        (1, 10, (3, 3), 1, 'same'),
        (10, 3, (5, 5), 1, 'same'),
        (3, 1, (7, 7), 1, 'same')
    ]

    trainset, testset = _load_data()

    train_data_loader = torch.utils.data.DataLoader(trainset)

    model = get_model(train_data_loader, n_epochs=1)

    test_data_loader = torch.utils.data.DataLoader(testset)

    vals = test_model(model, test_data_loader)

    print(vals)
