import torch
import torch.nn as nn


class PINN(nn.Module):

    def __init__(self, n_layers: int, inputs=2, outputs=1, device='cpu'):
        super().__init__()
        hidden_layers =[]
        for _ in range(n_layers-2):
            hidden_layers.append(nn.Linear(20, 20))
            hidden_layers.append(nn.Tanh())

        input_layer = [
                nn.Linear(inputs, 20),
                nn.Tanh()
        ]
        output_layer = nn.Linear(20, outputs)
        self.network = nn.Sequential(*input_layer, *hidden_layers, output_layer)
        self.device=device
        
    def forward(self, x, t):
        xt = torch.stack([x, t], axis=1).to(self.device)
        return self.network(xt)
    


if __name__ == "__main__":
    model = PINN(9)
    model.double()
    x = torch.tensor([1], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0], dtype=torch.float64, requires_grad=True)
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, create_graph=True)
    u_t = torch.autograd.grad(u, t, create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, create_graph=True)
    print(u_x, u_t, u_xx)
    