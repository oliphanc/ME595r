import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from pinodedata import getdata, true_encoder
import matplotlib.pyplot as plt

class MLP(nn.Module):

    def __init__(self, layer_widths, activation=nn.SiLU()):
        super(MLP, self).__init__()
        n = len(layer_widths)
        layers = []
        for i in range(n-1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i+1]))
            layers.append(activation)

        layers.pop()  # Remove last activation function

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        z = self.net(x)
        return z
    

class NeuralODE(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(NeuralODE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def forward(self, t, y0):

        return odeint(self.odefunc, y0, t)
    
    def odefunc(self, t, y):
        return self.net(y)
    
def train(encode, decode, NODE, Xtrain, Xtest, t_train, t_test, Xcol, fcol, epochs=400, lr=1e-4):
    combined_parameters = list(encode.parameters()) + \
                          list(decode.parameters()) + \
                          list(NODE.parameters())
    optim = torch.optim.Adam(combined_parameters)
    loss_func = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        (model.train() for model in [encode, decode, NODE])
        optim.zero_grad()
        Z0 = encode(Xtrain[:, 0, :])
        Zhat = NODE(t_train, Z0)
        Xhat = decode(Zhat)
        data_loss = loss_func(torch.permute(Xhat, (1,0,2)), Xtrain)
        recon_loss = loss_func(Xtrain, decode(encode(Xtrain)))

        dzdx = torch.zeros(ncol, nz, nx, dtype=torch.float64, device=device)
        Zcol = encode(Xcol)
        Zdot_col = NODE.odefunc(0., Zcol)
        dzdx[:, 0,:] = torch.autograd.grad(Zcol[:,0], Xcol, grad_outputs=torch.ones_like(Zcol[:,0]), create_graph=True)[0]
        dzdx[:, 1,:] = torch.autograd.grad(Zcol[:,1], Xcol, grad_outputs=torch.ones_like(Zcol[:,1]), create_graph=True)[0]
        dzdt = torch.bmm(dzdx, fcol.unsqueeze(2)).squeeze()
        physics_loss = loss_func(Zdot_col, dzdt)
        recon_loss_2 = loss_func(Xcol, decode(encode(Xcol)))

        total_loss = data_loss + recon_loss + physics_loss + recon_loss_2
        total_loss.backward()
        optim.step()

        train_losses.append(total_loss.item())

        (model.eval() for model in [encode, decode, NODE])
        with torch.no_grad():
            Z0 = encode(Xtest[:, 0, :])
            Zhat = NODE(t_test, Z0)
            Xhat = decode(Zhat)
            data_loss = loss_func(torch.permute(Xhat, (1,0,2)), Xtest)
            recon_loss = loss_func(Xtest, decode(encode(Xtest)))
            total_loss = data_loss + recon_loss
            test_losses.append(total_loss.item())
        
        print(epoch, train_losses[-1], test_losses[-1])
    
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test Losses')
    plt.legend()
    plt.yscale('log')
    plt.savefig(r"HW6\training_loss.png")
    plt.close()    


if __name__ == '__main__':
    TRAIN = False
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    nx = 128
    nz = 2
    encoder_layers = [nx, 256, 256, nz]

    encoder = MLP(encoder_layers).double().to(device)
    encoder_layers.reverse()
    decoder = MLP(encoder_layers).double().to(device)

    nt_train = 11
    nt_test = 21
    t_train = np.linspace(0.0, 1.0, nt_train)
    t_test = np.linspace(0.0, 1.0, nt_test)

    ntrain = 600
    ntest = 100
    ncol = 10000
    Xtrain, Xtest, Xcol, fcol, Amap = getdata(ntrain, ntest, ncol, t_train, t_test)

    NODE = NeuralODE(nz, 128).double().to(device)

    t_train = torch.tensor(t_train, dtype=torch.float64, device=device)
    t_test = torch.tensor(t_test, dtype=torch.float64, device=device)
    Xtrain = torch.tensor(Xtrain, dtype=torch.float64, device=device)
    Xtest = torch.tensor(Xtest, dtype=torch.float64, device=device)
    Xcol = torch.tensor(Xcol, dtype=torch.float64, device=device, requires_grad=True)
    fcol = torch.tensor(fcol, dtype=torch.float64, device=device)
    
    if TRAIN:
        train(encoder, decoder, NODE, Xtrain, Xtest, t_train, t_test, Xcol, fcol)
        torch.save(encoder.state_dict(), r"HW6\encoder.pt")
        torch.save(decoder.state_dict(), r"HW6\decoder.pt")
        torch.save(NODE.state_dict(), r"HW6\NODE.pt")
    
    else:
        encoder.load_state_dict(torch.load(r"HW6\encoder.pt", weights_only=True))
        decoder.load_state_dict(torch.load(r"HW6\decoder.pt", weights_only=True))
        NODE.load_state_dict(torch.load(r"HW6\NODE.pt", weights_only=True))
        

    Xhat = decoder(NODE(t_test, encoder(Xtest[:, 0, :])))
    Xhat = torch.permute(Xhat, (1,0,2))
    Xhat_np = Xhat.cpu().detach().numpy()
    Zhat = true_encoder(Xhat_np, Amap)

    print()
    plt.figure()
    for i in range(0, ntest):
        plt.plot(Zhat[i, 0, 0], Zhat[i, 0, 1], "ko")
        plt.plot(Zhat[i, :, 0], Zhat[i, :, 1], "k")
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1, 1])

    plt.show()

