import numpy as np
from numpy import vectorize
import matplotlib.pyplot as plt
from tqdm import tqdm

# get seed
seed = 471123720 #np.random.randint(0, 2**31)
print(f"Random Seed: {seed}")
np.random.seed(seed)


# -------- activation functions -------
@vectorize
def relu(z):
    if z < 0:
        return 0
    else:
        return z

@vectorize
def relu_back(xbar, z):
    return xbar if z > 0 else 0

identity = lambda z: z

identity_back = lambda xbar, z: xbar
# -------------------------------------------


# ---------- initialization -----------
def glorot(nin, nout):
    b = np.zeros((nout))
    var = 2 / (nin + nout)
    W = np.random.normal(0, var**.5, (nout, nin))
    return W, b
# -------------------------------------


# -------- loss functions -----------
def mse(yhat, y):
    m = y.size
    mse = 1/m * np.sum((yhat - y)**2)
    return mse

def mse_back(yhat, y):
    m = yhat.size
    yhat_bar = 2/m * (yhat - y)
    return yhat_bar
# -----------------------------------


# ------------- Layer ------------
class Layer:

    def __init__(self, nin, nout, activation=identity):
        # initialize and setup variables
        self.activation=activation
        self.W, self.b = glorot(nin, nout)

        if activation == relu:
            self.activation_back = relu_back
        if activation == identity:
            self.activation_back = identity_back

        # initialize cache
        self.X_cache = np.zeros((nin, 1))
        self.Z_cache = np.zeros((nout, 1))
    
    def __call__(self, X, train):
        return self.forward(X, train)

    def forward(self, X, train=True):
        Z = ((self.W @ X).T + self.b).T
        Xnew = self.activation(Z)

        # save cache
        if train:
            self.X_cache = X
            self.Z_cache = Z
            
        return Xnew

    def backward(self, Xnewbar):
        ZBar = self.activation_back(Xnewbar, self.Z_cache)
        Xbar = self.W.T @ ZBar
        self.Wbar = ZBar @ self.X_cache.T
        self.bbar = np.sum(ZBar, axis=1)
        return Xbar


class Network:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        if loss == mse:
            self.loss_back = mse_back

    def forward(self, X, y, train=True):
        for layer in self.layers:
            X = layer(X, train=train)
        yhat = X
        L = self.loss(yhat, y)
        # save cache
        if train:
            self.y = y
            self.yhat = yhat

        return L, yhat

    def backward(self):
        yhatbar = self.loss_back(self.yhat, self.y)
        Xnewbar = yhatbar
        
        for layer in reversed(self.layers):
            Xnewbar = layer.backward(Xnewbar)


class GradientDescent:

    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, network):
        for layer in network.layers:
            layer.W -= self.alpha * layer.Wbar
            layer.b -= self.alpha * layer.bbar



if __name__ == '__main__':

    # ---------- data preparation ----------------
    # Initialize lists for the numeric data and the string data
    numeric_data = []

    # Read the text file
    with open('data\\auto-mpg.data', 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split()

            # Check if any of the first 8 columns contain '?'
            if '?' in columns[:8]:
                continue  # Skip this line if there's a missing value

            # Convert the first 8 columns to floats and append to numeric_data
            numeric_data.append([float(value) for value in columns[:8]])

    # Convert numeric_data to a numpy array for easier manipulation
    numeric_array = np.array(numeric_data)

    # Shuffle the numeric array and the corresponding string array
    nrows = numeric_array.shape[0]
    indices = np.arange(nrows)
    np.random.shuffle(indices)
    shuffled_numeric_array = numeric_array[indices]

    # Split into training (80%) and test (20%) sets
    split_index = int(0.8 * nrows)

    train_numeric = shuffled_numeric_array[:split_index]
    test_numeric = shuffled_numeric_array[split_index:]

    # separate inputs/outputs
    Xtrain = train_numeric[:, 1:]
    ytrain = train_numeric[:, 0]

    Xtest = test_numeric[:, 1:]
    ytest = test_numeric[:, 0]

    # normalize
    Xmean = np.mean(Xtrain, axis=0)
    Xstd = np.std(Xtrain, axis=0)
    ymean = np.mean(ytrain)
    ystd = np.std(ytrain)

    Xtrain = (Xtrain - Xmean) / Xstd
    Xtest = (Xtest - Xmean) / Xstd
    ytrain = (ytrain - ymean) / ystd
    ytest = (ytest - ymean) / ystd

    # reshape arrays (opposite order of pytorch, here we have nx x ns).
    # I found that to be more conveient with the way I did the math operations, but feel free to setup
    # however you like.
    Xtrain = Xtrain.T
    Xtest = Xtest.T
    ytrain = np.reshape(ytrain, (1, len(ytrain)))
    ytest = np.reshape(ytest, (1, len(ytest)))

    # ------------------------------------------------------------

    l1 = Layer(7, 14, relu)
    l2 = Layer(14, 14, relu)
    l3 = Layer(14, 1)
    
    layers = [l1, l2, l3]
    network = Network(layers, mse)
    alpha = 5e-3
    optimizer = GradientDescent(alpha)

    train_losses = []
    test_losses = []
    epochs = 200
    for i in tqdm(range(epochs)):
        # run train set, backprop, step
        L, _ = network.forward(Xtrain, ytrain)
        network.backward()
        optimizer.step(network)
        train_losses.append(L)

        L, _ = network.forward(Xtest, ytest, train=False)
        test_losses.append(L)


    # --- inference ----
    _, yhat = network.forward(Xtest, ytest, train=False)

    # unnormalize
    yhat = (yhat * ystd) + ymean
    ytest = (ytest * ystd) + ymean

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()
    plt.savefig(f"Losses_{epochs}_epochs_{seed}_seed_{alpha}_LR.png")

    plt.figure()
    plt.plot(ytest.T, yhat.T, "o")
    plt.plot([10, 45], [10, 45], "--")
    plt.title(f"Avg error (mpg) = {np.mean(np.abs(yhat - ytest))}")
    plt.savefig(f"Error_{epochs}_epochs_{seed}_seed_{alpha}_LR.png")

    print("avg error (mpg) =", np.mean(np.abs(yhat - ytest)))

    plt.show()
