import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralODE(nn.Module):
    def __init__(self, data_dim):
        super(NeuralODE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, data_dim)
        )
    
    def forward(self, t, y0):

        return odeint(self.odefunc, y0, t)
    
    def odefunc(self, t, y):
        return self.net(y)

delhi_train = pd.read_csv(r"data\DailyDelhiClimateTrain.csv", parse_dates=["date"])
delhi_test  = pd.read_csv(r"data\DailyDelhiClimateTest.csv", parse_dates=["date"])
delhi = pd.concat([delhi_train, delhi_test], ignore_index=True)

delhi['year'] = delhi['date'].dt.year.astype(float)
delhi['month'] = delhi['date'].dt.month.astype(float)
df_mean = delhi.groupby(['year', 'month'], as_index=False).agg({
    'meantemp': 'mean',
    'humidity': 'mean',
    'wind_speed': 'mean',
    'meanpressure': 'mean'
})

df_mean['date'] = df_mean['year'] + df_mean['month'] / 12.0
features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

t_np = df_mean['date'].values
t_np = t_np - t_np.min()
t_all = torch.tensor(t_np, dtype=torch.float64).to(device)

y_np = df_mean[features].values.T
y_mean = np.mean(y_np, axis=1, keepdims=True)
y_std  = np.std(y_np, axis=1, keepdims=True)
y_np = (y_np - y_mean) / y_std
y_all = torch.tensor(y_np.T, dtype=torch.float64).to(device)

T_split = 20
train_t_np = t_np[:T_split]
test_t_np  = t_np[T_split:]
train_y = y_all[:T_split]  
test_y  = y_all[T_split:]  

def train_one_round(model, y, t, optimizer, epochs):

    y0 = y[0].unsqueeze(0)
    loss_fn = nn.MSELoss()
    losses = []
    for it in range(epochs):
        optimizer.zero_grad()
        pred = model(t, y0)
        pred = pred.squeeze(1) 
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def train(epochs=200, lr=1e-3):
 
    saved_losses = []
    model = NeuralODE(train_y.shape[1]).double()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for k in range(4, len(train_t_np) + 1, 4):
        t_train = torch.tensor(train_t_np[:k], dtype=torch.float64).to(device)
        y_segment = train_y[:k]
        model.train()
        losses = train_one_round(model, y_segment, t_train, optimizer, epochs)
        saved_losses += losses
        model.eval()
        with torch.no_grad():
            y0 = y_segment[0].unsqueeze(0)
            pred = model(t_train, y0)         
            pred = pred.squeeze(1) 
        pred_np = pred.cpu().numpy()
        target_np = y_segment.cpu().numpy()
        t_train_np_local = t_train.cpu().numpy()
        
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))
        axs = axs.flatten()
        for i, feat in enumerate(features):
            axs[i].plot(t_train_np_local, target_np[:, i], 'o-', label='Observed')
            axs[i].plot(t_train_np_local, pred_np[:, i], 'x--', label='Predicted')
            axs[i].set_title(feat)
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("Normalized value")
            axs[i].legend()
        fig.suptitle(f"Observations vs Predictions (k = {k})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
    return model, saved_losses

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    model, losses_log = train()
    
    with torch.no_grad():
        y0_full = y_all[0].unsqueeze(0)
        pred_full = model(t_all, y0_full)
        pred_full = pred_full.squeeze(1) 
    
    pred_full_np = pred_full.cpu().numpy()
    actual_full_np = y_all.cpu().numpy() 
    t_all_np = t_all.cpu().numpy()
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    axs = axs.flatten()
    for i, feat in enumerate(features):
        axs[i].plot(t_all_np, actual_full_np[:, i], 'bo', label='Actual')
        axs[i].plot(t_all_np, pred_full_np[:, i], 'k-', label='Predicted')
        axs[i].set_title(feat)
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Normalized value")
        axs[i].legend()
    fig.suptitle("Full Time Series Prediction", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    plt.figure()
    plt.plot(losses_log)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.show()
