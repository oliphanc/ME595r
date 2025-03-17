import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class GN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden, aggr = 'add'):
        super(GN, self).__init__(aggr=aggr)

        self.message_func = nn.Sequential(
            nn.Linear(2*n_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ndim)
        )

        self.node_func = nn.Sequential(
            nn.Linear(msg_dim + n_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ndim)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
    def message(self, x_i, x_j):
        cat = torch.cat([x_i, x_j], dim=1)
        return self.message_func(cat)
    
    def update(self, aggr_out, x=None):
        cat = torch.cat([x, aggr_out], dim=1)
        return self.node_func(cat)


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, edge_index):
        super(CustomDataset, self).__init__()
        
        self.x = x_tensor
        self.y = y_tensor
        self.edge_index = edge_index
        self.num_graphs = x_tensor.size(0)
        self.num_nodes = x_tensor.size(1)

    def len(self):
        return self.num_graphs
    
    def get(self, idx):
        return Data(x=self.x[idx], edge_index=self.edge_index, y=self.y[idx])


def data_setup(batch_size):

    # ------------ import data ------------
    # 1000 trajectory sets with 100 time steps split 75/25 for training/testing
    data = np.load(r"data\spring_data.npz")
    # X_train (75000, 4, 5)  data points, 4 particles, 5 states: x, y, Vx, Vy, m (positions, velocities, mass)
    X_train = torch.tensor(data['X_train'], dtype=torch.float64)
    # y_train (75000, 4, 2)  data points, 4 particles, 2 states: ax, ay (accelerations)
    y_train = torch.tensor(data['y_train'], dtype=torch.float64)
    # X_train (25000, 4, 5)  data points, 4 particles, 5 states
    X_test = torch.tensor(data['X_test'], dtype=torch.float64)
    # y_test (25000, 4, 5)  data points, 4 particles, 5 states
    y_test = torch.tensor(data['y_test'], dtype=torch.float64)
    # 100 time steps (not really needed)
    times = torch.tensor(data['times'], dtype=torch.float64)

    # ------- Save a few trajectories for plotting -------
    # the data points are currently ordered in time (for each separate trajectory)
    # so I'm going to save one set before shuffling the data.
    # this will make it easier to check how well I'm predicting the trajectories

    nt = len(times)
    train_traj = X_train[:nt, :, :]
    test_traj = X_test[:nt, :, :]

    # ------ edge index ------
    # this just defines how the nodes (particles) are connected
    # in this case each of the 4 particles interacts with every other particle
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
       [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
    ], dtype=torch.long)

    train_data = CustomDataset(X_train[:batch_size], y_train[:batch_size], edge_index)
    test_data = CustomDataset(X_test[:batch_size], y_test[:batch_size], edge_index)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_traj, test_traj


def train(model, train_loader, test_loader, device, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loss = []
    test_loss = []

    iters = tqdm(range(epochs))

    for epoch in iters:
        model.train()
        total_train_loss = 0
        for data in train_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index)
            loss = loss_fn(pred, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss +=loss.item()

        train_loss.append(total_train_loss)

        model.eval()
        total_test_loss = 0
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index)
            loss = loss_fn(pred, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_test_loss +=loss.item()

        test_loss.append(total_train_loss)

        iters.set_description(f"Losses: ({total_train_loss:.5e}, {total_test_loss:.5e})")

    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.show()



if __name__ == "__main__":

    batch_size = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, train_traj, test_traj = data_setup(batch_size)
    n_f = train_loader.dataset.x.size(2)
    msg_dim = 2
    ndim = train_loader.dataset.y.size(2)
    hidden=20

    model = GN(n_f, msg_dim, ndim, hidden).double().to(device)

    train(model, test_loader, train_loader, device, epochs=1000, lr=1e-3)