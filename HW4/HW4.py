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
        self.lambda_1 = nn.Parameter(torch.tensor([1.]))
        self.lambda_2 = nn.Parameter(torch.tensor([.01]))

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(init_weights)
        
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
    lambda_1_log = []
    lambda_2_log = []
    
    def closure():
        optimizer.zero_grad()
        u_pred = model(X, T)
        mse_u = torch.mean((u_pred - U)**2)
        
        u_t = torch.autograd.grad(u_pred, T, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred, X, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        residual =  u_t + model.lambda_1 * u_pred * u_x - model.lambda_2 * u_xx

        mse_f = torch.mean(residual**2)
        total_loss = mse_u + mse_f
        total_loss.backward()
        losses.append(total_loss.item())
        lambda_1_log.append(model.lambda_1.item())
        lambda_2_log.append(model.lambda_2.item())
        
        return total_loss
    
    model.train()
    for _ in range(epochs):
        optimizer.step(closure)
    
    plt.figure()
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')

    plt.figure()
    plt.plot(lambda_1_log)
    plt.xlabel('Iterations')
    plt.ylabel(r'$\lambda_1$')

    plt.figure()
    plt.plot(lambda_2_log)
    plt.xlabel('Iterations')
    plt.ylabel(r'$\lambda_2$')
    plt.show()

    return losses


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path('data') / 'burgers.txt'
    x_tensor, t_tensor, u_tensor = torch.tensor(np.genfromtxt(data_path).T, requires_grad=True).double().to(device)

    model = PINN().double().to(device)
    L = train(x_tensor.view(-1, 1), t_tensor.view(-1, 1), u_tensor, model)
    
    print("Lambda_1:", model.lambda_1.item())
    print("Error:", (model.lambda_1.item()-1))
    print("Lambda_2:", model.lambda_2.item())
    print("Error:", (model.lambda_2.item() - (.01/np.pi)))
    print("Final Training Loss:", L[-1])

    