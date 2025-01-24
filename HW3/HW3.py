import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube, scale


class PINN(nn.Module):

    def __init__(self, n_layers=9, width=20, inputs=2, outputs=1):
        super().__init__()
        hidden_layers =[]
        for _ in range(n_layers-2):
            hidden_layers.append(nn.Linear(width, width))
            hidden_layers.append(nn.Tanh())

        input_layer = [
                nn.Linear(inputs, width),
                nn.Tanh()
        ]
        output_layer = nn.Linear(width, outputs)
        self.network = nn.Sequential(*input_layer, *hidden_layers, output_layer)
        self.device=device
        
    def forward(self, x, t):
        xt = torch.stack([x, t], axis=1)
        return self.network(xt)

def u0(x):
    return -torch.sin(torch.pi * x)

def sample_boundary(n=100):
    x0 = torch.linspace(-1, 1, int(n / 2), dtype=torch.float64, requires_grad=True)
    t = torch.linspace(0, 1, int(n / 4), dtype=torch.float64, requires_grad=True)
    x_b = torch.ones_like(t, dtype=torch.float64, requires_grad=True)
    u_b = torch.zeros_like(x_b, dtype=torch.float64, requires_grad=True)
    all_x = torch.concat([x0, x_b, -x_b])
    all_t = torch.concat([torch.zeros_like(x0), t, t])
    all_u = torch.concat([u0(x0), u_b, u_b])

    return all_x, all_t, all_u

def sample_collocation(n=10000):
    rng = np.random.default_rng(seed=42)
    sampler = LatinHypercube(d=2, seed=rng, optimization='lloyd')
    sample = sampler.random(n=n)
    scaled = scale(sample, [-1, 0], [1, 1])
    XT = torch.tensor(scaled, dtype=torch.float64, requires_grad=True)
    X = XT[:, 0]
    T = XT[:, 1]
    return X, T

def f(x, t, model):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_x = torch.autograd.grad(u, x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xx = torch.autograd.grad(u_x, x, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]
    f = u_t + u * u_x - (0.01 / torch.pi) * u_xx
    return f

def loss_fn(model, Xn, Tn, Un, Xf, Tf):
    mse = torch.nn.MSELoss()
    Un_pred = model(Xn, Tn)
    residual = f(Xf, Tf, model)
    loss = mse(Un_pred, Un) + torch.mean(residual**2)
    return loss*100

def train(Xn, Tn, Un, Xf, Tf, epochs, model):
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    losses = []

    def closure():
        nonlocal model, Xn, Tn, Un, Xf, Tf
        optimizer.zero_grad()
        loss = loss_fn(model, Xn, Tn, Un, Xf, Tf)
        loss.backward(retain_graph=True)
        return loss

    model.train()
    for epoch in tqdm(range(epochs)):
        optimizer.step(closure)
        losses.append(loss_fn(model, Xn, Tn, Un, Xf, Tf).item())
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    # plt.show(block=False)

if __name__ == '__main__':
    Xn, Tn, Un = sample_boundary()
    Xf, Tf = sample_collocation()
    Un = Un.reshape(-1,1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = PINN(n_layers=9, width=40)
    model.double()
    model.to(device)
    Xn = Xn.to(device)
    Tn = Tn.to(device)
    Un = Un.to(device)
    Xf = Xf.to(device)
    Tf = Tf.to(device)

    train(Xn, Tn, Un, Xf, Tf, 500, model)
    torch.save(model, 'model.pt')

    model.eval()
    Up = model(Xn, Tn)
    plt.figure()
    plt.plot(Un.cpu().detach().numpy(),
             Up.cpu().detach().numpy(),
             'ko')

    x_range = torch.linspace(-1, 1, 1001, dtype=torch.float64)
    t_range = torch.linspace(0, 1, 1001, dtype=torch.float64)
    T, X = torch.meshgrid(t_range, x_range)
    T = T.to(device)
    X = X.to(device)
    
    U = model(X.flatten(), T.flatten())
    U = U.reshape(X.shape)
    plt.figure(figsize=(10, 4))
    contour = plt.contourf(T.cpu().detach().numpy(), 
                           X.cpu().detach().numpy(), 
                           U.cpu().detach().numpy(), 
                           levels=100, cmap='RdBu_r')
    
    plt.plot(Tn.cpu().detach().numpy(),
             Xn.cpu().detach().numpy(),
             'kx')

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