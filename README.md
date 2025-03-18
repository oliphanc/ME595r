## ME 595r - Deep Learning for Engineers

This is a repository used for completing the course ME 595r.
The directories are separated based on the assignments.

#### HW1
A basic implementation of pytorch. Takes the Auto MPG dataset and predicts mileage based on the dataset features using a linear neural network with ReLU activation functions.


#### HW2
Solves the same problem as HW1, but with a custom implementation in pure numpy.


#### HW3
An implementation of a Physics-Informed Neural Network (PINN). The problem posed in Appendix A of [this](https://doi.org/10.1016/j.jcp.2018.10.045) paper is solved with pytorch.
The implementation uses initial and boundary conditions as well as collocation points for training data. The L-BFGS optimization algorithm is used in training.


#### HW4
The same problem solved in HW3 is looked at again, but now the PINN is used to solve a PDE inverse problem. The given data is used to determine the coefficients $\lambda_1$ and $\lambda_2$ for the Burgers' Equation.


#### HW5
A basic implementation of a neural ODE to try and predict weather data.


#### HW6
Used a neural ODE with an autoencoder to predict the latent dynamics of a nonlinear system.


#### HW7
Used a neural network to approximate a koopman operator to linearize a nonlinear dynamic system.


#### HW8
Used a convolutional neural network to perform super-resolution of fluid flow. See [this](https://doi.org/10.1063/5.0054312) paper for more details. 


#### HW9
An implementation of a GNN used to predict the acceleration of particles in a spring system. The network is used to to predict the trajectories of the particles.