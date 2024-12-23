import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline


def ds(curve, index):
    if index == len(curve) - 1:
        return np.linalg.norm(curve[0] - curve[-1])
    return np.linalg.norm(curve[index + 1] - curve[index])

def unit_normals(curve):
    out = np.zeros((len(curve), 2))
    for i in range(len(curve)-1):
        gradient = (curve[i + 1] - curve[i])
        gradient = gradient / np.linalg.norm(gradient)
        out[i] = np.array([-gradient[1], gradient[0]])
    gradient = (curve[0] - curve[-1])
    gradient = gradient / np.linalg.norm(gradient)
    out[-1] = np.array([-gradient[1], gradient[0]])
    return out

#returns g11
def metric(curve, index, step_theta):
    if index == len(curve) - 1:
        return np.linalg.norm(curve[0] - curve[-1]) ** 2 / step_theta ** 2
    return np.linalg.norm(curve[index + 1] - curve[index]) ** 2 / step_theta ** 2

def metric_central(curve, index, step_theta):
    if index == len(curve) - 1:
        return np.linalg.norm(curve[0] - curve[-2]) ** 2 / (2 * step_theta) ** 2
    if index == 0:
        return np.linalg.norm(curve[1] - curve[-1]) ** 2 / (2 * step) ** 2
    return np.linalg.norm(curve[index + 1] - curve[index - 1]) ** 2 / (2 * step_theta) ** 2

#for the curvature, we will use the length element ds
#instead of step in the parameter (likely dtheta, which defines the metric)
def curvature(curve, index):
    if index == len(curve) - 1:
        gradient1 = (curve[0] - curve[-1])
        gradient1 = gradient1 / np.linalg.norm(gradient1)
        gradient2 = (curve[-1] - curve[-2])
        gradient2 = gradient2 / np.linalg.norm(gradient2)
        return np.linalg.norm(gradient1 - gradient2) / ds(curve, index)    
    if index == 0:
        gradient1 = (curve[1] - curve[0])
        gradient1 = gradient1 / np.linalg.norm(gradient1)
        gradient2 = (curve[0] - curve[-1])
        gradient2 = gradient2 / np.linalg.norm(gradient2)

        return np.linalg.norm(gradient1 - gradient2) / ds(curve, index)

    gradient1 = (curve[index + 1] - curve[index])
    gradient1 = gradient1 / np.linalg.norm(gradient1)
    gradient2 = (curve[index] - curve[index - 1])
    gradient2 = gradient2 / np.linalg.norm(gradient2)

    return np.linalg.norm(gradient1 - gradient2) / ds(curve, index)
    
#note vdisps has shape (2 * number of points, 2)
# eg vdisps[0:2] = [vdisp_parallel at position 1, vdisp perp at position 1]
#for ease, the state vector will look like (sigma1, m1, sigma2, m2...)
def populate_matrix(curve, step_theta, vdisps):
    out = np.zeros((2 * len(curve), 2 * len(curve)))
    for i, vdisp in enumerate(vdisps):
        for j, point in enumerate(curve):
            
            dg = 0
            dkappa = 0

            flag = False
            #changes in metric and curvature at some position due to perturbation at
            #that same position, elif changes in metric and curvature at some position 
            #due to perturbation at neighboring position to the right elif same thing
            #to the left
            if i == 2 * j or i == 2 * j + 1:
                temp = curve.copy()
                temp[j] = point + vdisp
                dg = metric(temp, j, step_theta) - metric(curve, j, step_theta)
                dkappa = curvature(temp, j) - curvature(curve, j)

            elif i == (2 * j + 2) % (2 * len(curve))  or i == (2 * j + 3) % (2 * len(curve)):
                temp = curve.copy()
                temp[(j + 1) % len(curve)] = temp[(j+1) % len(curve)] + vdisp
                dg = 0.5 * (metric(temp, j, step_theta) - metric(curve, j, step_theta))
                dkappa = curvature(temp, j) - curvature(curve, j)

            #here note python '%' is like mathematical mod. eg -1%5 = 4. Neat!
            elif i == (2 * j - 2)% (2 * len(curve)) or i == (2 * j - 1)% (2 * len(curve)):
                temp = curve.copy()
                #using central difference for this derivative as otherwise it would be 0
                temp[(j - 1)% len(curve)] = temp[(j - 1)% len(curve)] + vdisp
                dg = 0.5 * ( metric(temp, (j-2)% len(curve), step_theta) - metric(curve, (j-2)% len(curve), step_theta))
                dkappa = curvature(temp, j) - curvature(curve, j)
            
            kappa = curvature(curve, j)

            #this is fine at first order
            entry1 = dg / 2
            entry2 = -dg * kappa / 2 - dkappa

            out[i, 2 * j] = entry1 * ds(curve, j)
            out[i, 2 * j + 1] = entry2 * ds(curve, j)
    return out

#rhs of an equation such as Ax = b
#for now only works for forces in the perp direction
#as this also extracts the normal comp of x.
def populate_rhs(curve, forces_perp, vdisps):
    out = np.zeros(2 * len(forces_perp))
    forces_index = 0
    for i, vdisp in enumerate(vdisps):
        out[i] = np.dot(forces_perp[forces_index], vdisp) * ds(curve, forces_index)
        if i % 2 == 1:
            forces_index += 1
    return out

#prec is the ratio of virtual displacement size to step size in the curve
def backward_solver(curve, L, N, prec):

    curve_copy = augment_curve(curve, N).copy()

    step_theta = L / N

    #generate vdisps
    epsilon = step_theta * prec
    vdisps = np.zeros((2 * len(curve_copy), 2))
    vdisps = np.random.rand(2 * len(curve_copy), 2)
    vdisps = vdisps / np.linalg.norm(vdisps, axis=1)[:, np.newaxis]
    # vdisps[0::2, 0] = 1  # Set every second element's y-component to 1
    # vdisps[1::2, 1] = 1  # Set every second element's x-component to 1
    vdisps = vdisps * epsilon

    #external force densities
    p = 1 * np.ones(N)#np.sin(2 * thetas) 
    ps = np.array([p, p]).T
    forces_perp = ps * unit_normals(curve_copy)

    # forces_perp = curve * p / R
    A = populate_matrix(curve_copy, step_theta, vdisps)
    b = populate_rhs(curve_copy, forces_perp, vdisps)

    #solve the system
    x = np.linalg.solve(A, b)

    sigmas = x[0::2]
    ms = x[1::2]
    
    return sigmas, ms

#note: the curve input must be such that the first and last points are the same
#for it to work well
def augment_curve(curve, N):
    # Original array of points
    points = curve
    x = points[:, 0]
    y = points[:, 1]

    # Parameterize the curve (e.g., by cumulative distance)
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)

    # Create interpolation functions for x and y
    interp_func_x = CubicSpline(cumulative_dist, x, bc_type='periodic')
    interp_func_y = CubicSpline(cumulative_dist, y, bc_type='periodic')

    # Generate new evenly spaced cumulative distances
    new_cumulative_dist = np.linspace(cumulative_dist[0], cumulative_dist[-1], N+1)

    # Interpolated x and y
    x_new = interp_func_x(new_cumulative_dist)
    y_new = interp_func_y(new_cumulative_dist)

    # Interpolated points
    interpolated_points = np.vstack((x_new, y_new)).T
    return interpolated_points[:-1]

def interpolate_curve(curve):
    N = len(curve)
    points = curve
    x = points[:, 0]
    y = points[:, 1]

    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)

    # Create interpolation functions for x and y
    interp_func_x = CubicSpline(cumulative_dist, x, bc_type='periodic')
    interp_func_y = CubicSpline(cumulative_dist, y, bc_type='periodic')

    def f(x):
        return np.array([interp_func_x(x), interp_func_y(x)])
    return f

def interpolate_seq_periodic(seq):
    N = len(seq)
    x = np.linspace(0, 1, N)
    if not np.isclose(seq[0], seq[-1]):
        raise ValueError("Sequence must be periodic, Santi")
    interp_func = CubicSpline(x, seq, bc_type='periodic')
    return interp_func

def forward_solver(mag=0.6, tns=None):
    """
    Curve forward solver for input pressure + normal forces
    """
    # Set parameters and functions
    p = 1  # Pressure
    sig0 = 0.7  # Tension sigma(0)
    L_initial = 2 * np.pi  # Initial guess for length

    # Solution domain
    xmesh = np.linspace(0, 1, len(tns) if tns is not None else 200)

    # SETTING NORMAL FORCE PROFILE
    # Random tn profile
    NrModes = 3  # How many Fourier modes should be included
    Nmax = 5  # Maximal mode number to draw from (Nmax >= NrModes)
    N = np.random.choice(range(1, Nmax + 1), NrModes, replace=False)  # Set of modes
    Magnc = mag * (np.random.rand(NrModes) - 0.5)  # Magnitudes of cosine modes
    Magns = mag * (np.random.rand(NrModes) - 0.5)  # Magnitudes of sine modes

    if tns is None:
        tn = lambda x: (np.real(np.exp(1j * 2 * np.pi * x[:, None] * N) @ Magnc) +
                        np.imag(np.exp(1j * 2 * np.pi * x[:, None] * N) @ Magns))

        dtn = lambda x: (np.real(1j * 2 * np.pi * (N[None, :] * np.exp(1j * 2 * np.pi * x[:, None] * N)) @ Magnc) +
                        np.imag(1j * 2 * np.pi * (N[None, :] * np.exp(1j * 2 * np.pi * x[:, None] * N)) @ Magns))
    else:
        dtns = np.gradient(tns, xmesh) 

        #ideally switch to periodic interpolation with the function 'interpolate_seq_periodic'.
        tn = interp1d(xmesh, tns, kind='cubic')
        dtn = interp1d(xmesh, dtns, kind='cubic')

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
