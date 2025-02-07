import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
        
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.network(xt)
    

def train(X, T, U, model, lambda_, epochs=1):
   
    optimizer = torch.optim.LBFGS(
        list(model.parameters()) + lambda_,
        lr=1.0,
        max_iter=20000,
        max_eval=20000,
        history_size=100,
        line_search_fn='strong_wolfe'
    )
    
    losses = []
    lambda_1_log = []
    lambda_2_log = []
    epoch = 0
    def closure():
        optimizer.zero_grad()
        u_pred = model(X, T)
        mse_u = torch.mean((u_pred - U)**2)
        
        u_t, u_x = torch.autograd.grad(u_pred, inputs=(T, X), grad_outputs=torch.ones_like(u_pred), create_graph=True)
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        residual =  u_t + lambda_[0] * u_pred * u_x - lambda_[1] * u_xx

        mse_f = torch.mean(residual**2)
        total_loss = mse_u + (2**epoch)*mse_f
        total_loss.backward()
        losses.append(total_loss.item())
        lambda_1_log.append(lambda_[0].item())
        lambda_2_log.append(lambda_[1].item())
        
        return total_loss
    
    model.train()
    for _ in range(epochs):
        optimizer.step(closure)
        epoch+=1
    
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

    return losses, lambda_


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path('data') / 'burgers.txt'
    x_tensor, t_tensor, u_tensor = torch.tensor(np.genfromtxt(data_path).T, requires_grad=True).double().to(device)

    model = PINN().double().to(device)
    lambda_1=nn.Parameter(torch.ones(1, dtype=torch.float64).to(device).requires_grad_())
    lambda_2=nn.Parameter(torch.ones(1, dtype=torch.float64).to(device).requires_grad_())
    L, lambda_ = train(x_tensor.view(-1, 1), 
                       t_tensor.view(-1, 1), 
                       u_tensor.view(-1, 1), 
                       model, [lambda_1, lambda_2], epochs=1)
    
    print("Lambda_1:", lambda_[0].item())
    print("Error:", lambda_[0].item()-1)
    print("Lambda_2:", lambda_[1].item())
    print("Error:", (lambda_[1].item() - (.01/np.pi)))
    print("Final Training Loss:", L[-1])

    x_range = torch.linspace(-1, 1, 1001, dtype=torch.float64)
    t_range = torch.linspace(0, 1, 1001, dtype=torch.float64)
    T, X = torch.meshgrid(t_range, x_range)
    T = T.to(device)
    X = X.to(device)
    U = model(X.reshape(-1,1), T.reshape(-1,1))
    U = U.reshape(X.shape)

    plt.figure(figsize=(10, 4))
    contour = plt.contourf(T.cpu().detach().numpy(), 
                           X.cpu().detach().numpy(), 
                           U.cpu().detach().numpy(), 
                           levels=100, cmap='RdBu_r')
    plt.colorbar(contour, label='u(t, x)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.savefig(r'HW4\x_t_contour.png')

    plt.show()

    