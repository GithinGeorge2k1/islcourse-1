import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch import nn
from sklearn import metrics


def kali():
    print('kali')

# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname


class cs19b036NN(nn.Module):
    def __init__(self, X, num_classes) -> None:
        super().__init__()
        input_shape = 1
        for s in X.shape:
            input_shape *= s
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax()
        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_stack(X)
        return logits


# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
    num_classes = len(train_data_loader.dataset.classes)
    model = cs19b036NN(train_data_loader.dataset[0][0], num_classes)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, y) in enumerate(train_data_loader):
            size = len(train_data_loader.dataset)

            pred = model(X)
            loss = nn.CrossEntropyLoss()(pred, y)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print('Returning model... (rollnumber: cs19b036)')

    return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)


class cs19b036CNN(nn.Module):

    def __init__(self, input_shape, classes, config1) -> None:

        super().__init__()
        # config1 = [
        #     (1, 10, (3, 3), 1, 'same'),
        #     (10, 3, (5, 5), 1, 'same'),
        #     (3, 1, (7, 7), 1, 'same')
        # ]
        self.config = config1

        self.conv_layers = []

        for conv_layer_config in self.config:
            conv_out_shape = [_compute_conv_output_shape(
                input_shape[0], conv_layer_config[2], conv_layer_config[3], conv_layer_config[4]),
                _compute_conv_output_shape(
                    input_shape[1], conv_layer_config[2], conv_layer_config[3], conv_layer_config[4]),
                conv_layer_config[1]
            ]
            in_channels, out_channels, kernel_size, stride, padding = conv_layer_config
            self.conv_layers.append(nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2, 2))

        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


def get_model_advanced(train_data_loader=None, n_epochs=10, lr=1e-4, config=None):
    num_classes = len(train_data_loader.dataset.classes)
    model = cs19b036CNN(
        train_data_loader.dataset[0][0].shape[0], num_classes, config)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, y) in enumerate(train_data_loader):
            size = len(train_data_loader.dataset)

            pred = model(X)
            loss = nn.CrossEntropyLoss()(pred, y)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print('Returning model... (rollnumber: cs19b036)')

    return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)


def test_model(model1=None, test_data_loader=None):

    accs = []
    precs = []
    recs = []
    f1s = []

    size = len(test_data_loader.dataset)
    num_batches = len(test_data_loader)
    model1.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_data_loader:
            pred = model1(X)
            accs.append(metrics.accuracy_score(y, pred.argmax(dim=1)))
            precs.append(metrics.precision_score(
                y, pred.argmax(dim=1), average='macro'))
            recs.append(metrics.recall_score(
                y, pred.argmax(dim=1), average='macro'))
            f1s.append(metrics.f1_score(
                y, pred.argmax(dim=1), average='macro'))

    test_loss /= num_batches
    correct /= size
    print('Returning metrics... (rollnumber: cs19b036)')

    accuracy_val = np.mean(accs)
    precision_val = np.mean(precs)
    recall_val = np.mean(recs)
    f1score_val = np.mean(f1s)

    return accuracy_val, precision_val, recall_val, f1score_val


def _compute_conv_output_shape(h_w, kernel_size=1, stride=1, pad=0):
    return (h_w[0] + (2 * pad) - (kernel_size - 1) - 1) // stride + 1, (h_w[1] + (2 * pad) - (kernel_size - 1) - 1) // stride + 1


def _load_data():

    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    return training_data, test_data


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
