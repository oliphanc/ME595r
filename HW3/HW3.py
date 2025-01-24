import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale


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
        xt = torch.cat([x, t], dim=1)  # Proper concatenation
        return self.network(xt)

def u0(x):
    return -torch.sin(torch.pi * x)

def sample_boundary(n=100):
    x_ic = torch.linspace(-1, 1, n//2, dtype=torch.float64).unsqueeze(1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = u0(x_ic)
    
    t_bc = torch.linspace(0, 1, n//2, dtype=torch.float64).unsqueeze(1)
    x_bc_left = -torch.ones_like(t_bc)
    x_bc_right = torch.ones_like(t_bc)
    
    X = torch.cat([x_ic, x_bc_left, x_bc_right])
    T = torch.cat([t_ic, t_bc, t_bc])
    U = torch.cat([u_ic, torch.zeros_like(t_bc), torch.zeros_like(t_bc)])
    
    return X, T, U

def sample_collocation(n=10000):
    sampler = LatinHypercube(d=2, seed=42)
    sample = sampler.random(n=n)
    scaled = scale(sample, [-1, 0], [1, 1])  # Correct scaling: x∈[-1,1], t∈[0,1]
    XT = torch.tensor(scaled, dtype=torch.float64)
    return XT[:, 0:1], XT[:, 1:2]

def f(x, t, model):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - (0.01/np.pi) * u_xx

def train(Xn, Tn, Un, Xf, Tf, model, epochs=1):
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
        u_pred = model(Xn, Tn)
        mse_u = torch.mean((u_pred - Un)**2)
        residual = f(Xf, Tf, model)
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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Xn, Tn, Un = sample_boundary(n=100)  
    Xf, Tf = sample_collocation(n=10000)
    
    Xn = Xn.to(device).requires_grad_(True)
    Tn = Tn.to(device).requires_grad_(True)
    Un = Un.to(device)
    Xf = Xf.to(device).requires_grad_(True)
    Tf = Tf.to(device).requires_grad_(True)
    
    model = PINN(n_layers=5, width=20).double().to(device)
    losses = train(Xn, Tn, Un, Xf, Tf, model) 
    torch.save(model, 'model.pt')

    model.eval()

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

    plt.figure()
    plt.plot( X.cpu().detach().numpy()[750], 
              U.cpu().detach().numpy()[750], 
              'r-')
    plt.xlabel('x')
    plt.ylabel('u(t, x)')
    plt.show()