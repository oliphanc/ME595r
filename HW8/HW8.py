import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
from tqdm import tqdm

class CNNSR(nn.Module):

    def __init__(self, channels, size,  kernel, stride, padding, activation):
        super(CNNSR, self).__init__()

        self.net = nn.Sequential(
            nn.Upsample(size, mode='bicubic'),
            nn.Conv2d(channels, 16, kernel, stride, padding),
            activation,
            nn.Conv2d(16, 32, kernel, stride, padding),
            activation,
            nn.Conv2d(32, 16, kernel, stride, padding),
            activation,
            nn.Conv2d(16, channels+1, kernel, stride, padding)
        )

    def forward(self, uv):
        fbar = self.net(uv)

        return fbar
    


def load_data():

    lfdata = np.load(r"data\sr_lfdata.npy")
    lfx = lfdata[0, :, :]  # size 14 x 9  (height x width)
    lfy = lfdata[1, :, :]
    lfu = lfdata[4, :, :]
    lfv = lfdata[5, :, :]

    return lfx, lfy, lfu, lfv


def load_high_resolution_grid():

    hfdata = np.load(r"data\sr_hfdata.npy")
    Jinv = hfdata[0, :, :]  # size 77 x 49 (height x width)
    dxdxi = hfdata[1, :, :]
    dxdeta = hfdata[2, :, :]
    dydxi = hfdata[3, :, :]
    dydeta = hfdata[4, :, :]
    hfx = hfdata[5, :, :]
    hfy = hfdata[6, :, :]

    ny, nx = hfx.shape  #(77 x 49)
    h = 0.01  # grid spacing in high fidelity (needed for derivatives)

    return Jinv, dxdxi, dxdeta, dydxi, dydeta, hfx, hfy, ny, nx, h


def ddxi(f, h):
    # 5-pt stencil
    dfdx_central = (f[:, :, :, 0:-4] - 8*f[:, :, :, 1:-3] + 8*f[:, :, :, 3:-1] - f[:, :, :, 4:]) / (12*h)
    # 1-sided 4pt stencil
    dfdx_left = (-11*f[:, :, :, 0:2] + 18*f[:, :, :, 1:3] -9*f[:, :, :, 2:4] + 2*f[:, :, :, 3:5]) / (6*h)
    dfdx_right = (-2*f[:, :, :, -5:-3] + 9*f[:, :, :, -4:-2] -18*f[:, :, :, -3:-1] + 11*f[:, :, :, -2:]) / (6*h)

    return torch.cat((dfdx_left, dfdx_central, dfdx_right), dim=3)


def ddeta(f, h):
    # 5-pt stencil
    dfdy_central = (f[:, :, 0:-4, :] - 8*f[:, :, 1:-3, :] + 8*f[:, :, 3:-1, :] - f[:, :, 4:, :]) / (12*h)
    # 1-sided 4pt stencil
    dfdy_bot = (-11*f[:, :, 0:2, :] + 18*f[:, :, 1:3, :] -9*f[:, :, 2:4, :] + 2*f[:, :, 3:5, :]) / (6*h)
    dfdy_top = (-2*f[:, :, -5:-3, :] + 9*f[:, :, -4:-2, :] -18*f[:, :, -3:-1, :] + 11*f[:, :, -2:, :]) / (6*h)

    return torch.cat((dfdy_bot, dfdy_central, dfdy_top), dim=2)


def train(model, uv, lr, epochs):
    global Jinv, dxdxi, dxdeta, dydxi, dydeta, hfx, hfy, ny, nx, h
    nu=0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iters = tqdm(range(epochs))
    losses = []

    model.train()
    for epoch in iters:
        uvp = model(uv)
        u = uvp[:, 0, :, :]
        v = uvp[:, 1, :, :]
        p = uvp[:, 2, :, :]

        dudeta = ddeta(u, h)
        dudxi  = ddxi(u, h)
        dudx = Jinv * (dudxi * dydeta - dudeta * dydxi)
        dudy = Jinv * (dudeta * dxdxi - dudxi * dxdeta)

        d2udxdeta = ddeta(dudy, h)
        d2udxdxi = ddxi(dudy, h)
        d2udx2 = Jinv * (d2udxdxi * dydeta - d2udxdeta * dydxi)
        d2udydeta = ddeta(dudx, h)
        d2udydxi = ddxi(dudy, h)
        d2udy2 = Jinv * (d2udydeta * dxdxi - d2udydxi * dxdeta)
        
        dvdeta = ddeta(v, h)
        dvdxi  = ddxi(v, h)
        dvdx = Jinv * (dvdxi * dydeta - dvdeta * dydxi)
        dvdy = Jinv * (dvdeta * dxdxi - dvdxi * dxdeta)
        
        d2vdxdeta = ddeta(dvdy, h)
        d2vdxdxi = ddxi(dvdy, h)
        d2vdx2 = Jinv * (d2vdxdxi * dydeta - d2vdxdeta * dydxi)
        d2vdydeta = ddeta(dvdx, h)
        d2vdydxi = ddxi(dvdy, h)
        d2vdy2 = Jinv * (d2vdydeta * dxdxi - d2vdydxi * dxdeta)

        dpdeta = ddeta(uvp[:, 2, :, :], h)
        dpdxi  = ddxi(uvp[:, 2, :, :], h)
        dpdx = Jinv * (dpdxi * dydeta - dpdeta * dydxi)
        dpdy = Jinv * (dpdeta * dxdxi - dpdxi * dxdeta)

        continuity = dudx + dvdy

        x_momentum = u * dudx + v*dudy + dpdx - nu * (d2udx2 + d2udy2)
        y_momentum = u * dvdx + v*dvdy + dpdy - nu * (d2vdx2 + d2vdy2)

        total_loss = torch.mean(continuity**2) + torch.mean(x_momentum**2) + torch.mean(y_momentum**2)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

        iters.set_description(f"Loss: {losses[-1]:.5f}")




    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lfx, lfy, lfu, lfv = load_data()
    Jinv, dxdxi, dxdeta, dydxi, dydeta, hfx, hfy, ny, nx, h = load_high_resolution_grid()
    luv = torch.tensor(np.stack([lfu, lfv]), dtype=torch.float64).unsqueeze(0).to(device)

    channels = 2
    size = (ny, nx)
    kernel = 5
    stride = 1
    padding = 2
    activation = nn.ReLU()

    model = CNNSR(channels, size,  kernel, stride, padding, activation).double().to(device)
    
    print(model)