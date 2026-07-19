# Dual-Network PINN for Options Pricing and Local Volatility ExtractionOverview
This repository provides a deep learning framework that extends the traditional Black-Scholes model by substituting constant volatility with a dynamic local volatility surface. Utilizing a Dual-Network Physics-Informed Neural Network (PINN), the model simultaneously approximates European call option prices and reverse-engineers the underlying local volatility directly from sparse, real-world market observations.  

Key Features
- Physics-Informed Optimization: Embeds the Black-Scholes Partial Differential Equation (PDE) directly into the neural network's loss function to enforce the absolute laws of financial physics.
- Dual-Network Architecture: Employs two parallel Multi-Layer Perceptrons (MLPs)—one to predict the option's fair value and another to estimate the instantaneous local volatility.
- Self-Calibrating Loss Function: Implements a dynamic uncertainty weighting mechanism that automatically equalizes the optimization of data loss, PDE residuals, and boundary conditions.
- Singularity Truncation: Utilizes infinitely differentiable SiLU activations and gradient clipping to safely handle the Dirac Delta singularity at expiration, allowing for the stable derivation of option Greeks (Delta, Gamma, Theta) without gradient collapse.

Technologies used:
- PyTorch
- Pandas
- SciPy
- Plotly & Matplotlib (for 3D surface and loss visualization) 
