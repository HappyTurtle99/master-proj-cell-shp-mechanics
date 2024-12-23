import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def forward_solver():
    """
    Curve forward solver for input pressure + normal forces
    """
    # Set parameters and functions
    p = 1  # Pressure
    sig0 = 0.7  # Tension sigma(0)
    L_initial = 2 * np.pi  # Initial guess for length

    # Solution domain
    xmesh = np.linspace(0, 1, 200)

    # SETTING NORMAL FORCE PROFILE
    # Random tn profile
    NrModes = 3  # How many Fourier modes should be included
    Nmax = 5  # Maximal mode number to draw from (Nmax >= NrModes)
    N = np.random.choice(range(1, Nmax + 1), NrModes, replace=False)  # Set of modes
    Magnc = 0.6 * (np.random.rand(NrModes) - 0.5)  # Magnitudes of cosine modes
    Magns = 0.6 * (np.random.rand(NrModes) - 0.5)  # Magnitudes of sine modes

    tn = lambda x: (np.real(np.exp(1j * 2 * np.pi * x[:, None] * N) @ Magnc) +
                    np.imag(np.exp(1j * 2 * np.pi * x[:, None] * N) @ Magns))

    dtn = lambda x: (np.real(1j * 2 * np.pi * (N[None, :] * np.exp(1j * 2 * np.pi * x[:, None] * N)) @ Magnc) +
                     np.imag(1j * 2 * np.pi * (N[None, :] * np.exp(1j * 2 * np.pi * x[:, None] * N)) @ Magns))

    # ODE function
    def bvpfcn(x, y):
        """
        y[0] = psi, y[1] = sigma, y[2] = x, y[3] = y, y[4] = L
        """
        dydx = np.zeros_like(y)
        L = y[4, 0]  # Extract L from the state vector
        dydx[0] = (L * p + dtn(x)) / y[1]
        dydx[1] = -tn(x) * (L * p + dtn(x)) / y[1]
        dydx[2] = L * np.cos(y[0])
        dydx[3] = L * np.sin(y[0])
        dydx[4] = 0  # L is a constant parameter
        return dydx

    # ODE boundary conditions
    def bcfcn(ya, yb):
        return np.array([
            ya[0],             # psi(0) = 0
            yb[0] - 2 * np.pi, # psi(1) = 2*pi
            ya[1] - sig0,      # sigma(0) = sig0
            ya[2],             # x(0) = 0
            ya[3]              # y(0) = 0
        ])

    # Create initial guess solution
    def guess(x):
        return np.vstack([
            2 * np.pi * x,         # Psi
            np.full_like(x, sig0), # Sigma
            np.cos(2 * np.pi * x), # x
            np.sin(2 * np.pi * x), # y
            np.full_like(x, L_initial)  # L (constant)
        ])

    # Set up initial guess
    y_guess = guess(xmesh)
    solinit = (xmesh, y_guess)

    # Solve the ODE as a boundary value problem
    sol = solve_bvp(bvpfcn, bcfcn, solinit[0], solinit[1])

    # Plot the solution
    plt.figure(figsize=(10, 10))
    
    # Shape plot
    plt.subplot(2, 2, 1)
    plt.plot(sol.y[2], sol.y[3], linewidth=2)
    plt.scatter(sol.y[2, 0], sol.y[3, 0], label='Start Point', s=80)
    plt.scatter(sol.y[2, -1], sol.y[3, -1], label='End Point', s=80)
    plt.axis('equal')
    plt.title('Shape')
    plt.legend()

    # Tension plot
    plt.subplot(2, 2, 2)
    plt.plot(sol.x, sol.y[1], linewidth=2)
    plt.title('Tension')

    # Psi plot
    plt.subplot(2, 2, 3)
    plt.plot(sol.x, sol.y[0], linewidth=2)
    plt.title('Psi')

    # Curvature plot
    kappa = ((sol.y[4, 0] * p + dtn(sol.x)) / sol.y[1]) / sol.y[4, 0]
    plt.subplot(2, 2, 4)
    plt.plot(sol.x, kappa, linewidth=2)
    plt.title('Curvature')

    plt.tight_layout()
    plt.show()

    print("Length:", sol.y[4, 0])
    
    return sol

# Call the function
sol = forward_solver()
