import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch import nn


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


def get_model_advanced(train_data_loader=None, n_epochs=10, lr=1e-4, config=None):
    model = None

    # write your code here as per instructions
    # ... your code ...
    # ... your code ...
    # ... and so on ...
    # Use softmax and cross entropy loss functions
    # set model variable to proper object, make use of train_data

    # In addition,
    # Refer to config dict, where learning rate is given,
    # List of (in_channels, out_channels, kernel_size, stride=1, padding='same')  are specified
    # Example, config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')], it can have any number of elements
    # You need to create 2d convoution layers as per specification above in each element
    # You need to add a proper fully connected layer as the last layer

    # HINT: You can print sizes of tensors to get an idea of the size of the fc layer required
    # HINT: Flatten function can also be used if required
    print('Returning model... (rollnumber: cs19b036)')

    return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)


def test_model(model1=None, test_data_loader=None):

    accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0

    accs = []

    size = len(test_data_loader.dataset)
    num_batches = len(test_data_loader)
    model1.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_data_loader:
            pred = model1(X)
            accs.append(np.sum(np.equal(y, pred)) / len(y))
    test_loss /= num_batches
    correct /= size
    print('Returning metrics... (rollnumber: cs19b036)')

    accuracy_val = np.mean(accs)

    return accuracy_val, precision_val, recall_val, f1score_val


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

    trainset, testset = _load_data()

    train_data_loader = torch.utils.data.DataLoader(trainset)

    model = get_model(train_data_loader, n_epochs=1)