import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )
    
    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)
        return self.net(inputs)

# Set up device and parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nu = 0.01 / np.pi

# Training data setup
N_u = 100   # Initial/Boundary points
N_f = 10000  # Collocation points

# Initial condition data
t_i = torch.zeros(N_u, 1, device=device, dtype=torch.float64)
x_i = torch.linspace(-1, 1, N_u, device=device, dtype=torch.float64).unsqueeze(1)
u_i = -torch.sin(np.pi * x_i)

# Boundary condition data
t_b = torch.rand(N_u, 1, device=device, dtype=torch.float64)
x_b = torch.cat([-torch.ones(N_u, 1, dtype=torch.float64), torch.ones(N_u, 1, dtype=torch.float64)], dim=0).to(device)
t_bc = torch.cat([t_b, t_b], dim=0)
u_bc = torch.zeros(2*N_u, 1, device=device, dtype=torch.float64)

# Collocation points
t_f = torch.rand(N_f, 1, device=device, requires_grad=True, dtype=torch.float64)
x_f = (torch.rand(N_f, 1, device=device, dtype=torch.float64) * 2 - 1).requires_grad_(True)

# Initialize model and optimizer
model = PINN().to(device)
model.double()
optimizer = torch.optim.LBFGS(model.parameters(), 
                             lr=1.0, 
                             max_iter=20000, 
                             max_eval=20000, 
                             history_size=100,
                             tolerance_grad=1e-5,
                             line_search_fn='strong_wolfe')

# Training loop with L-BFGS
def closure():
    optimizer.zero_grad()
    
    # PDE Loss
    u_pred = model(t_f, x_f)
    du_dt = torch.autograd.grad(u_pred, t_f, torch.ones_like(u_pred), create_graph=True)[0]
    du_dx = torch.autograd.grad(u_pred, x_f, torch.ones_like(u_pred), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x_f, torch.ones_like(du_dx), create_graph=True)[0]
    pde_loss = torch.mean((du_dt + u_pred * du_dx - nu * d2u_dx2) ** 2)
    
    # Initial Condition Loss
    ic_pred = model(t_i, x_i)
    ic_loss = torch.mean((ic_pred - u_i) ** 2)
    
    # Boundary Condition Loss
    bc_pred = model(t_bc, x_b)
    bc_loss = torch.mean((bc_pred - u_bc) ** 2)
    
    # Total Loss
    total_loss = pde_loss + ic_loss + bc_loss
    total_loss.backward()
    
    return total_loss

# Run optimization
epochs = 1  # L-BFGS handles iterations internally
for epoch in range(epochs):
    optimizer.step(closure)
    current_loss = closure().item()
    print(f'Epoch: {epoch+1}, Loss: {current_loss:.4e}')

# Visualization code remains the same as before
# Generate test data
t_test = np.linspace(0, 1, 1001)
x_test = np.linspace(-1, 1, 1001)
T, X = np.meshgrid(t_test, x_test)
t_flat = T.flatten()[:, None]
x_flat = X.flatten()[:, None]

# Predictions
with torch.no_grad():
    t_tensor = torch.tensor(t_flat).double().to(device)
    x_tensor = torch.tensor(x_flat).double().to(device)
    u_pred = model(t_tensor, x_tensor).cpu().numpy()
U_pred = u_pred.reshape(T.shape)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.contourf(T, X, U_pred, levels=100, cmap='RdBu')
plt.colorbar(label='u(t,x)')
plt.xlabel('t')
plt.ylabel('x')

plt.figure()
plt.plot(X[750], U_pred[750], 'k-')
plt.xlabel('x')
plt.ylabel('u(0.75, x)')
plt.show()