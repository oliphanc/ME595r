import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

class DeepKoopman(nn.Module):

    def __init__(self, layer_widths, activation=nn.SiLU()):
        super(DeepKoopman, self).__init__()
        n = len(layer_widths)
        layers = []
        for i in range(n-1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i+1]))
            layers.append(activation)
        layers.pop()  # Remove last activation function
        self.encoder = nn.Sequential(*layers)

        self.K = nn.Linear(layer_widths[-1], layer_widths[-1], bias=False)
        nn.init.normal_(self.K.weight, 0, 0.01)
        
        layers = []
        for i in range(n-1, 0, -1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i-1]))
            layers.append(activation)
        layers.pop()  # Remove last activation function
        self.decoder = nn.Sequential(*layers)

    def forward(self, xk):
        zk = self.encoder(xk)
        zk_1 = self.K(zk)
        xk_1 = self.decoder(zk_1)
        return xk_1


def train(model, Ytrain, Ytest, epochs=600, autoencoder_epochs=100, lr=5e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    
    train_losses = []
    test_losses = [1]
    torch.autograd.set_detect_anomaly(True)
    iters = tqdm(range(epochs))
    max_timestep = 10

    for epoch in iters:
        model.train()
        # reconstruction loss
        encoding = model.encoder(Ytrain)
        total_loss = loss_func(Ytrain, model.decoder(encoding))
    

        if epoch > autoencoder_epochs:

            if epoch % 100 == 0:
                max_timestep += 10
                tqdm.write(f"Increasing max timestep to {max_timestep}")


            latent_prediction = torch.zeros(encoding.shape[0], 1, encoding.shape[2], dtype=torch.float64).to(device)
            latent_prediction[:, 0, :] = encoding[:, 0, :]
            prediction = torch.zeros(Ytrain.shape[0], 1, Ytrain.shape[2], dtype=torch.float64).to(device)
            prediction[:, 0, :] = Ytrain[:, 0, :]
            
            if max_timestep >= 50:
                max_timestep=48

            for t in range(max_timestep):
                new_latent = model.K(latent_prediction[:, -1, :])
                # tqdm.write(str(new_latent.shape))
                
                latent_prediction = torch.concatenate((latent_prediction, new_latent.unsqueeze(1)), dim=1)
                new_pred = model(prediction[:, -1, :])
                prediction = torch.concatenate((prediction, new_pred.unsqueeze(1)), dim=1)
            
            # Linear dynamics
            linear_loss = loss_func(encoding[:, 1:max_timestep+1, :], latent_prediction[:, :-1, :])

            # State Prediction
            prediction_loss = loss_func(Ytrain[:, 1:max_timestep+1, :], prediction[:, :-1, :])

            total_loss += linear_loss
            total_loss += prediction_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_losses.append(total_loss.item())

        # model.eval()
        # with torch.no_grad():
        #     test_encoding = model.encoder(Ytrain)
        #     test_loss = loss_func(Ytrain, model.decoder(test_encoding))
        #     if epoch > autoencoder_epochs:
                
        #         test_latent_prediction = torch.zeros_like(test_encoding)
        #         test_latent_prediction[:, 0, :] = test_encoding[:, 0, :]
        #         test_prediction = torch.zeros_like(Ytrain)
        #         test_prediction[:, 0, :] = Ytrain[:, 0, :]
                
        #         for t in range(1, test_prediction.shape[1]):
        #             test_latent_prediction[:, t, :] = model.K(test_encoding[:, t-1, :])
        #             test_prediction[:, t, :] = model(test_prediction[:, t-1, :])

        #         # Linear dynamics
        #         test_linear_loss = loss_func(test_encoding[:, 1:, :], test_latent_prediction[:, :-1, :])

        #         # State Prediction
        #         test_prediction_loss = loss_func(Ytrain[:, 1:, :], test_prediction[:, :-1, :])

        #         test_loss += test_linear_loss
        #         test_loss += test_prediction_loss
            
        #     test_losses.append(test_loss.item())

        iters.set_description(f"Losses: ({train_losses[-1]:.5f}, {test_losses[-1]:.5f})")

    plt.figure()
    plt.plot(train_losses)
    # plt.plot(test_losses)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.show()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print("Training on", device)

    ntraj = 2148  # number of trajectories
    nt = 50  # number of time steps
    ny = 7  # number of states
    nz = 32 # number of dimensions in latent space

    tvec = np.linspace(0, 350, nt)
    Y = np.loadtxt(r'data\kdata.txt').reshape(ntraj, nt, ny)
    Ytrain = Y[:1, :, :]  # 2048 training trajectories
    Ytest = Y[2048:, :, :]  # 100 testing trajectoreis

    encoder_layers = [ny, 256, 256, nz]

    model = DeepKoopman(encoder_layers).double().to(device)
    Ytrain_torch = torch.tensor(Ytrain, dtype=torch.float64).to(device)
    Ytest_torch  = torch.tensor(Ytest, dtype=torch.float64).to(device)

    train(model, Ytrain_torch, Ytest_torch)
    prediction = torch.zeros_like(Ytrain_torch[0], dtype=torch.float64).to(device)
    prediction[0] = Ytrain_torch[0][0]

    for t in range(1, prediction.shape[0]):
        prediction[t] = model(prediction[t-1])

    prediction_np = prediction.cpu().detach().numpy()
    plot_case = Y[0]

    plt.figure()
    plt.plot(plot_case[:,0], 'r-')
    plt.plot(plot_case[:,1], 'g-')
    plt.plot(plot_case[:,2], 'b-')

    plt.plot(prediction_np[:,0], 'r--')
    plt.plot(prediction_np[:,1], 'g--')
    plt.plot(prediction_np[:,2], 'b--')
    plt.ylim(0, 2.5)
    plt.show()
