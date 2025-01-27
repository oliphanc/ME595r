import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale
from pathlib import Path


class PINN(nn.Module):
    def __init__(self, n_layers=5, width=20, inputs=2, outputs=1):
        super().__init__()
        layers = [nn.Linear(inputs, width), nn.Tanh()]
        for _ in range(n_layers-2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, outputs))
        self.network = nn.Sequential(*layers)
        self.lambda_1 = nn.Parameter(torch.rand(1))
        self.lambda_2 = nn.Parameter(torch.rand(1))
        
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.network(xt)
    

def train(X, T, U, model, epochs=1):
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20000,
        max_eval=20000,
        history_size=100,
        tolerance_grad=1e-5,
        line_search_fn='strong_wolfe'
    )
    
    losses = []
    
    def closure():
        optimizer.zero_grad()
        u_pred = model(X, T)
        mse_u = torch.mean((u_pred - U)**2)
        
        u_t = torch.autograd.grad(u_pred, T, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred, X, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        residual =  u_t + model.lambda_1 * u * u_x - model.lambda_2 * u_xx

        mse_f = torch.mean(residual**2)
        total_loss = mse_u + mse_f
        total_loss.backward()
        losses.append(total_loss.item())
        return total_loss
    
    model.train()
    for _ in range(epochs):
        optimizer.step(closure)
    
    plt.figure()
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    
    return losses


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path('data') / 'burgers.txt'
    x, t, u = torch.tensor(np.genfromtxt(data_path).T, requires_grad=True).double().to(device)

    model = PINN(n_layers=5, width=20).double().to(device)
    L = train(x.view(-1, 1), t.view(-1, 1), u, model)
    
    print(model.lambda_1)
    print(model.lambda_2)
    print(L[-1])
    plt.show(block=False)