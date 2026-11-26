# Physics-Informed Neural Network (PINN) for Solving the 1D Heat Equation

## Overview

This project implements a **Physics-Informed Neural Network (PINN)** to approximate the solution of the one-dimensional heat equation using deep learning and physical constraints.

Unlike conventional neural networks that primarily learn from labelled datasets, PINNs incorporate governing physical equations into their training process. This allows the network to learn solutions that are consistent with the underlying physical laws.

In this project, a fully connected neural network is trained to predict the temperature distribution along a one-dimensional rod over time. Its predictions are compared with the exact analytical solution to evaluate model performance.

## Objectives

- Understand the mathematical formulation of the one-dimensional heat equation.
- Implement a Physics-Informed Neural Network using PyTorch.
- Use automatic differentiation to calculate first- and second-order derivatives.
- Incorporate PDE residuals, initial conditions, and boundary conditions into the loss function.
- Train a neural network to approximate the temperature distribution.
- Compare PINN predictions with the analytical solution.
- Evaluate model performance using numerical metrics and visualizations.

## 1. Mathematical Formulation

### 1.1 One-Dimensional Heat Equation

The one-dimensional heat equation describes how temperature changes over space and time.

$$
\frac{\partial u(x,t)}{\partial t}
=
\alpha \frac{\partial^2 u(x,t)}{\partial x^2}
$$

Where:

- $u(x,t)$: Temperature at spatial position $x$ and time $t$.
- $x$: Spatial coordinate.
- $t$: Time.
- $\alpha$: Thermal diffusivity.

Thermal diffusivity is defined as:

$$
\alpha = \frac{k}{\rho c_p}
$$

Where $k$ is thermal conductivity, $\rho$ is density, and $c_p$ is specific heat capacity.

### 1.2 Domain

The normalized spatial and temporal domains are:

$$
x \in [0,1], \qquad t \in [0,1]
$$

The thermal diffusivity used in this experiment is:

$$
\alpha = 0.1
$$

### 1.3 Initial Condition

The initial temperature distribution is:

$$
u(x,0)=\sin(\pi x)
$$

### 1.4 Boundary Conditions

Both ends of the rod are maintained at zero temperature:

$$
u(0,t)=0
$$

$$
u(1,t)=0
$$

### 1.5 Analytical Solution

For the selected heat equation and conditions, the exact analytical solution is:

$$
u(x,t)=e^{-\alpha\pi^2t}\sin(\pi x)
$$

Substituting $\alpha=0.1$:

$$
u(x,t)=e^{-0.1\pi^2t}\sin(\pi x)
$$

This analytical solution is used as the reference for evaluating the PINN predictions.

## 2. Physics-Informed Neural Network

A Physics-Informed Neural Network is a neural network trained using both data constraints and governing physical equations.

The network approximates the temperature function as:

$$
u(x,t)\approx u_\theta(x,t)
$$

Where $\theta$ represents the trainable weights and biases of the neural network.

### Network Architecture

The implemented network consists of:

| Layer | Configuration |
|---|---|
| Input layer | 2 neurons ($x,t$) |
| Hidden layer 1 | 64 neurons |
| Hidden layer 2 | 64 neurons |
| Hidden layer 3 | 64 neurons |
| Hidden layer 4 | 64 neurons |
| Output layer | 1 neuron ($u$) |
| Activation function | Tanh |

The Tanh activation function is:

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

A smooth activation function is useful because the heat equation requires differentiating the network output with respect to its inputs.

## 3. Physics-Informed Loss Function

The PINN is trained using three loss components.

### 3.1 PDE Residual Loss

The heat equation residual is defined as:

$$
f_\theta(x,t)=
\frac{\partial u_\theta}{\partial t}
-
\alpha\frac{\partial^2u_\theta}{\partial x^2}
$$

For an exact solution:

$$
f_\theta(x,t)=0
$$

The PDE loss is:

$$
L_{\mathrm{PDE}}=
\frac{1}{N_f}
\sum_{i=1}^{N_f}
\left[f_\theta(x_f^i,t_f^i)\right]^2
$$

Here, $N_f$ represents the number of collocation points sampled inside the domain.

### 3.2 Initial Condition Loss

The initial condition loss ensures that the network reproduces the specified initial temperature distribution.

$$
L_{\mathrm{IC}}=
\frac{1}{N_i}
\sum_{i=1}^{N_i}
\left[
u_\theta(x_i,0)-\sin(\pi x_i)
\right]^2
$$

### 3.3 Boundary Condition Loss

The boundary condition loss ensures that the temperature remains zero at both ends of the rod.

$$
L_{\mathrm{BC}}=
\frac{1}{N_b}
\sum_{i=1}^{N_b}
\left[
u_\theta(0,t_i)^2+
u_\theta(1,t_i)^2
\right]
$$

### 3.4 Total Loss

The total loss combines all three components:

$$
L_{\mathrm{Total}}=
L_{\mathrm{PDE}}+
L_{\mathrm{IC}}+
L_{\mathrm{BC}}
$$

All three components are assigned equal weights in this implementation.

The training objective is to minimize the total loss so that the network satisfies the governing PDE and its initial and boundary conditions.

## 4. Training Configuration

The following parameters are used in the experiment:

| Parameter | Value |
|---|---:|
| Spatial domain | $[0,1]$ |
| Temporal domain | $[0,1]$ |
| Thermal diffusivity | 0.1 |
| PDE collocation points | 2,000 |
| Initial condition points | 200 |
| Boundary condition points | 200 |
| Hidden layers | 4 |
| Neurons per hidden layer | 64 |
| Activation function | Tanh |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training epochs | 5,000 |

The training points consist of:

- **Collocation points:** Sampled inside the spatial-temporal domain to evaluate the PDE residual.
- **Initial condition points:** Sampled along the initial time boundary.
- **Boundary condition points:** Sampled at both spatial boundaries.

Each training epoch performs one full-batch optimization step using the generated training points.

## 5. Implementation Workflow

The project follows these steps:

1. Import the required Python libraries.
2. Define the physical parameters and training configuration.
3. Construct the fully connected neural network.
4. Generate collocation, initial condition, and boundary condition points.
5. Calculate derivatives using PyTorch automatic differentiation.
6. Compute the PDE residual.
7. Calculate the initial and boundary condition losses.
8. Combine the loss components into a total loss.
9. Train the neural network using the Adam optimizer.
10. Evaluate predictions against the analytical solution.
11. Visualize the temperature distributions and training behavior.
12. Calculate numerical evaluation metrics.

## 6. Evaluation Metrics

The PINN predictions are compared with the analytical solution using the following metrics.

Let $u_i$ denote the exact solution and $\hat{u}_i$ denote the predicted solution.

### Mean Squared Error (MSE)

$$
MSE=
\frac{1}{N}
\sum_{i=1}^{N}(u_i-\hat{u}_i)^2
$$

### Root Mean Squared Error (RMSE)

$$
RMSE=\sqrt{MSE}
$$

### Relative $L_2$ Error

$$
L_2^{\mathrm{relative}}=
\frac{
\sqrt{\sum_{i=1}^{N}(u_i-\hat{u}_i)^2}
}{
\sqrt{\sum_{i=1}^{N}u_i^2}
}
$$

### Maximum Absolute Error

$$
E_{\max}=
\max_{1\leq i\leq N}|u_i-\hat{u}_i|
$$

Lower values indicate closer agreement between the PINN predictions and the analytical solution.

## 7. Visualizations

The project includes visualizations to understand and evaluate the trained model:

- Training loss curve.
- Analytical temperature distribution.
- PINN-predicted temperature distribution.
- Comparison of analytical and predicted temperature profiles at selected time values.
- Numerical error metrics.

These plots help assess whether the network has learned the expected heat diffusion behavior.

## 8. Technologies Used

- **Python** – Main programming language.
- **PyTorch** – Neural network implementation, optimization, and automatic differentiation.
- **NumPy** – Numerical computations.
- **Matplotlib** – Data visualization.
- **Google Colab / Jupyter Notebook** – Development and experimentation.

## 9. How to Run the Project

### Option 1: Google Colab

1. Open the project notebook in Google Colab.
2. Make sure the required libraries are available.
3. Run the notebook cells sequentially.
4. Wait for the neural network training to complete.
5. Examine the generated visualizations and evaluation metrics.

### Option 2: Local Environment

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Navigate to the project directory:

```bash
cd YOUR_REPOSITORY
```

Install the required libraries:

```bash
pip install numpy matplotlib torch
```

Open the notebook using Jupyter:

```bash
jupyter notebook
```

Run the notebook cells sequentially.

## 10. Limitations and Future Improvements

This implementation is an introductory PINN experiment and has several limitations:

- It considers only a one-dimensional heat equation.
- It uses a fixed set of collocation, initial, and boundary points.
- All loss components are assigned equal weights.
- Model accuracy may depend on the network architecture and optimization settings.
- More complex geometries and variable material properties are not considered.

Potential future improvements include:

- Experimenting with different neural network architectures.
- Increasing or adaptively selecting collocation points.
- Investigating different loss weighting strategies.
- Comparing different optimization algorithms.
- Extending the implementation to two-dimensional heat equations.
- Exploring more complex initial and boundary conditions.

## 11. Conclusion

This project demonstrates how Physics-Informed Neural Networks can be used to approximate solutions of partial differential equations.

A fully connected neural network is trained to solve the one-dimensional heat equation by minimizing the PDE residual and satisfying the initial and boundary conditions.

The predicted temperature distribution is compared with the analytical solution using numerical evaluation metrics and visualizations.

The project provides practical experience in combining deep learning, automatic differentiation, and mathematical physics for scientific computing.

## Author

**Soorian Kathiravan Anjalidev**

BS in Data Science and Applications – IIT Madras

MSc in Physics – VIT Vellore

## License

This project is intended for educational and learning purposes. Add a suitable open-source license if you plan to distribute or reuse the code publicly.
