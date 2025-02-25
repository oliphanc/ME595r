import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class DeepKoopman(nn.Module):

    def __init__(self, layer_widths, activation=nn.SiLU()):
        super(DeepKoopman, self).__init__()
        n = len(layer_widths)
        layers = []
        for i in range(n-1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i+1]))
            layers.append(activation)
        layers.pop()  # Remove last activation function
        self.encoder = nn.Sequential(*layers)

        self.K = nn.Linear(layer_widths[-1], layer_widths[-1], bias=False)
        
        layers = []
        for i in range(n-1, 0, -1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i-1]))
            layers.append(activation)
        layers.pop()  # Remove last activation function
        self.decoder = nn.Sequential(*layers)

    def forward(self, xk):
        zk = self.encoder(xk)
        zk_1 = self.K(zk)
        xk_1 = self.decoder(zk_1)
        return xk_1


def train(model, Ytrain, Ytest=None, epochs=400, lr=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        # reconstruction loss
        total_loss = loss_func(Ytrain, model.decoder(model.encoder(Ytrain)))

        if epoch > 10:
            encoding = model.encoder(Ytrain)
            prediction = model.K(encoding)
            decoding = model.decoder(prediction)

            # Linear dynamics
            linear_loss = loss_func(encoding[:, 1:, :], prediction[:, :-1, :])

            # State Prediction
            prediction_loss = loss_func(Ytrain[:, 1:, :], decoding[:, :-1, :])

            total_loss += linear_loss
            total_loss += prediction_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_losses.append(total_loss.item())
        print(epoch, total_loss.item())

    plt.figure()
    plt.plot(train_losses)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.show()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    ntraj = 2148  # number of trajectories
    nt = 50  # number of time steps
    ny = 7  # number of states
    nz = 32 # number of dimensions in latent space

    tvec = np.linspace(0, 350, nt)
    Y = np.loadtxt(r'data\kdata.txt').reshape(ntraj, nt, ny)
    Ytrain = Y[:100, :, :]  # 2048 training trajectories
    Ytest = Y[2048:, :, :]  # 100 testing trajectoreis

    encoder_layers = [ny, 256, 256, nz]

    model = DeepKoopman(encoder_layers).double().to(device)
    Ytrain_torch = torch.tensor(Ytrain, dtype=torch.float64).to(device)
    Ytest_torch  = torch.tensor(Ytest, dtype=torch.float64).to(device)

    train(model, Ytrain_torch, Ytest_torch)

    prediction = model(Ytrain_torch[0])
    prediction_np = prediction.cpu().detach().numpy()
    plot_case = Y[0]

    plt.figure()
    plt.plot(plot_case[:,0], 'r-')
    plt.plot(plot_case[:,1], 'g-')
    plt.plot(plot_case[:,2], 'b-')

    plt.plot(prediction_np[:,0], 'r--')
    plt.plot(prediction_np[:,1], 'g--')
    plt.plot(prediction_np[:,2], 'b--')
    plt.show()
