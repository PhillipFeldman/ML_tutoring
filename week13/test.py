import os
import numpy as np

import cv2

IMGSIZE = (128, 128)
CNAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
X_tr, y_tr, X_ts, y_ts = [], [], [], []
for label in CNAMES:
    path = "." + '/seg_train/seg_train/' + label
    for f in sorted([_ for _ in os.listdir(path) if _.lower().endswith('.jpg')]):
        X_tr += [cv2.resize(cv2.imread(os.path.join(path, f)), IMGSIZE)]
        y_tr += [CNAMES.index(label)]



# q1
number_of_color_channels = X_tr[0].shape[2]
number_of_color_channels

# q2
X_trnp = np.array(X_tr)
y_trnp = np.array(y_tr)
X_trnp.shape

y_trnp.shape

"""#scaling 
dsize = (1,1)
for x in range(len(X_tr)):
    X_tr[x] = cv2.resize(X_tr[x], dsize, interpolation = cv2.INTER_AREA)
"""


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class CustomMLP(nn.Module):
    """ A PyTorch neural network model """

    def __init__(self, n_hidden=30, epochs=100, eta=0.05, minibatch_size=50):
        super(CustomMLP, self).__init__()
        self.n_hidden = n_hidden  # hidden layer size
        self.epochs = epochs  # number of learning iterations
        self.eta = eta  # learning rate
        self.minibatch_size = minibatch_size  # size of training batch - 1 would not work
        self.fc1, self.fc2, self.fc3 = None, None, None

    def _forward(self, X, apply_softmax=False):
        assert self.fc1 != None
        X = nn.functional.relu(self.fc1(X))
        X = nn.functional.relu(self.fc2(X))
        X = self.fc3(X)
        if apply_softmax:
            X = nn.functional.softmax(X, dim=1)
        return X

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def predict(self, _X):
        assert self.fc1 is not None
        net_out = self._forward(_X, apply_softmax=True)
        p_values, indices = net_out.max(dim=1)
        return indices

    def fit(self, X_train, y_train):
        # Convert to tensors

        self._reset()  # Reset the neural network weights
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        self.fc1 = nn.Linear(n_features, self.n_hidden)  # A simple input layer
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)  # A simple hidden layer
        self.fc3 = nn.Linear(self.n_hidden, n_output)  # A simple output layer

        optimizer = optim.SGD(self.parameters(), lr=self.eta, momentum=0.9)
        loss_func = nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            indices = np.arange(X_train.shape[0])

            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                optimizer.zero_grad()

                net_out = self._forward(X_train[batch_idx])

                loss = loss_func(net_out, y_train[batch_idx])
                loss.backward()
                optimizer.step()


import sys

X_trt, y_trt = torch.tensor(X_trnp), torch.tensor(y_trnp)

X_tr

X_trf = torch.flatten(X_trt, start_dim=1)

X_trf


clf = CustomMLP()

clf.fit(X_trf, y_trt)

