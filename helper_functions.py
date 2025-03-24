import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sint
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

"""
Main functions for initial iteration
"""
def ds(curve, index):
    if index == len(curve) - 1:
        return np.linalg.norm(curve[0] - curve[-1])
    return np.linalg.norm(curve[index + 1] - curve[index])
 
#gives outward pointing normals (using central diff)
def unit_normals(curve):
    out = np.zeros((len(curve), 2))
    for i in range(1, len(curve)-1):
        gradient = (curve[i + 1] - curve[i-1])
        gradient = gradient / np.linalg.norm(gradient)
        out[i] = np.array([-gradient[1], gradient[0]])

    gradient = (curve[0] - curve[-2])
    gradient = gradient / np.linalg.norm(gradient)
    out[-1] = np.array([-gradient[1], gradient[0]])

    gradient = (curve[1] - curve[-1])
    gradient = gradient / np.linalg.norm(gradient)
    out[0] = np.array([-gradient[1], gradient[0]])

    return -out

#returns g11
def metric(curve, index, step_theta):
    if index == len(curve) - 1:
        return np.linalg.norm(curve[0] - curve[-1]) ** 2 / step_theta ** 2
    return np.linalg.norm(curve[index + 1] - curve[index]) ** 2 / step_theta ** 2

def metric_central(curve, index, step_theta):
    if index == len(curve) - 1:
        return np.linalg.norm(curve[0] - curve[-2]) ** 2 / (2 * step_theta) ** 2
    if index == 0:
        return np.linalg.norm(curve[1] - curve[-1]) ** 2 / (2 * step_theta) ** 2
    return np.linalg.norm(curve[index + 1] - curve[index - 1]) ** 2 / (2 * step_theta) ** 2

#for the curvature, we will use the length element ds
#instead of step in the parameter (likely dtheta, which defines the metric)
def curvature(curve, index):
    if index < 0:
        index = len(curve) + index

    if index == len(curve) - 1:
        gradient1 = (curve[0] - curve[-1])
        gradient1 = gradient1 / np.linalg.norm(gradient1)
        gradient2 = (curve[-1] - curve[-2])
        gradient2 = gradient2 / np.linalg.norm(gradient2)
        cross = -np.cross(np.append(gradient1, 0), np.append(gradient2, 0))

        return np.sign(cross[2]) * np.linalg.norm(gradient1 - gradient2) / ds(curve, index)    
    if index == 0:
        gradient1 = (curve[1] - curve[0])
        gradient1 = gradient1 / np.linalg.norm(gradient1)
        gradient2 = (curve[0] - curve[-1])
        gradient2 = gradient2 / np.linalg.norm(gradient2)
        cross = np.cross(np.append(gradient1, 0), np.append(gradient2, 0))

        return -np.sign(cross[2]) * np.linalg.norm(gradient1 - gradient2) / ds(curve, index)

    gradient1 = (curve[index + 1] - curve[index])
    gradient1 = gradient1 / np.linalg.norm(gradient1)
    gradient2 = (curve[index] - curve[index - 1])
    gradient2 = gradient2 / np.linalg.norm(gradient2)

    #we now want a way of giving this curvature a sign
    #this is done by the cross product of the two gradients

    cross = np.cross(np.append(gradient1, 0), np.append(gradient2, 0))

    return -np.sign(cross[2]) * np.linalg.norm(gradient1 - gradient2) / ds(curve, index)

#note vdisps has shape (2 * number of points, 2)
# eg vdisps[0:2] = [vdisp_parallel at position 1, vdisp perp at position 1]
#for ease, the state vector will look like (sigma1, m1, sigma2, m2...)

#CURRENT EDIT!! USING METRIC CENTRAL
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
                dg = metric_central(temp, j, step_theta) - metric_central(curve, j, step_theta)
                dkappa = curvature(temp, j) - curvature(curve, j)

            elif i == (2 * j + 2) % (2 * len(curve))  or i == (2 * j + 3) % (2 * len(curve)):
                temp = curve.copy()
                temp[(j + 1) % len(curve)] = temp[(j+1) % len(curve)] + vdisp
                dg = (metric_central(temp, j, step_theta) - metric_central(curve, j, step_theta))
                dkappa = curvature(temp, j) - curvature(curve, j)

            #here note python '%' is like mathematical mod. eg -1%5 = 4. Neat!
            elif i == (2 * j - 2)% (2 * len(curve)) or i == (2 * j - 1)% (2 * len(curve)):
                temp = curve.copy()
                #using central difference for this derivative as otherwise it would be 0
                temp[(j - 1)% len(curve)] = temp[(j - 1)% len(curve)] + vdisp
                dg = (metric_central(temp, (j)% len(curve), step_theta) - metric_central(curve, (j)% len(curve), step_theta))
                dkappa = curvature(temp, j) - curvature(curve, j)
            
            kappa = curvature(curve, j)

            entry1 = dg / 2
            entry2 = -dg * kappa / 2 - dkappa

            out[i, 2 * j] = entry1 * ds(curve, j)
            out[i, 2 * j + 1] = entry2 * ds(curve, j)
    return out

#populates matrix not sparsely that is each displacement displaces all points
#so each row is full. The reason it is called hack is it is a shortcut and it is
#correct only for first order it approximates dg(vdisp1+vdisp2) = dg(vdisp1) + dg(vdisp2)
def populate_matrix_set_hack(curve, p=1, prec=1e-9):
    matrix = np.zeros((2 * len(curve), 2 * len(curve)))
    rhs = np.zeros(2 * len(curve))
    fns = p * np.ones(len(curve)) #normal forces

    L = np.sum([ds(curve, i) for i in range(len(curve))])

    for i in range(len(curve) * 2):
        #in testing we will see what this does for convergence
        extra_factor = 1

        step_theta = L / len(curve)

        epsilon = step_theta * prec
        vdisps = np.random.rand(2 * len(curve), 2) - 0.5
        vdisps = vdisps / np.linalg.norm(vdisps, axis=1)[:, np.newaxis]
        vdisps = vdisps * epsilon

        A = populate_matrix(curve, step_theta, vdisps)

        matrix[i] = np.sum(A[0::2], axis=0)
        
        rhs[i] = 0
        for j in range(len(curve)):
            rhs[i] += -np.dot(unit_normals(curve)[j], vdisps[j * 2]) * ds(curve, j) * fns[j]

    return matrix, rhs

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
#pts displaced is the parameter which determines how the matrix is populated
#in other words the what sets of displacements are taken.
#- '1' means to use populate_matrix()
#- '2' means to use populate_matrix_set_hack()
#- '3' means to use populate_matrix_set() (not implemented)
def backward_solver(curve, L, N, population_method='1', prec=1e-7, p=1):

    curve_copy = curve.copy()
    curve_copy = augment_curve(curve_copy, N)

    if population_method == '1':
        step_theta = L / N
        #external force densities    
        forces_perp = p * unit_normals(curve_copy)
        #generate vdisps
        epsilon = step_theta * prec
        vdisps = np.random.rand(2 * len(curve_copy), 2) - 0.5
        vdisps = vdisps / np.linalg.norm(vdisps, axis=1)[:, np.newaxis]
        vdisps = vdisps * epsilon
        A = populate_matrix(curve_copy, step_theta, vdisps)
        b = populate_rhs(curve_copy, forces_perp, vdisps)
    elif population_method == '2':
        A, b = populate_matrix_set_hack(curve_copy, p, prec)

    #solve the system
    x = np.linalg.solve(A, b)

    sigmas = x[0::2]
    ms = x[1::2]
    
    return sigmas, ms

def backward_solver_cont(curve, L, N, prec=1e-4, p=1):
    curve_copy = curve.copy()
    curve_copy = augment_curve(curve_copy, N)

    step_theta = L / N
    
    vdisps = np.zeros((2 * N, 2))
    first_disps = gen_cont_disps(curve_copy, L, prec=1e-4)
    vdisps[0:N] = first_disps

    second_disps = gen_cont_disps(curve_copy, L, prec=1e-4)
    vdisps[N:] = second_disps

    p = 1
    forces_perp = unit_normals(curve_copy) * p

    A = populate_matrix(curve_copy, step_theta, vdisps)
    b = populate_rhs(curve_copy, forces_perp, vdisps)

    soln = np.linalg.solve(A, b)

    sigmas = soln[0::2]
    ms = soln[1::2]

    return sigmas, ms

#feed only curves which are closed under whatever error threshold
# you are willing to accept from the backward solver -- see * lines below
def augment_curve(curve, N):

    # Original array of points
    points = curve[:-1] #*
    points = np.vstack([points, points[0]]) #*
    x = points[:, 0]
    y = points[:, 1]

    # Parameterize the curve (e.g., by cumulative distance)
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)

    # Create interpolation functions for x and y
    interp_func_x = CubicSpline(cumulative_dist, x, bc_type='periodic')
    interp_func_y = CubicSpline(cumulative_dist, y, bc_type='periodic')

    # Here we use N+1 because the above use of 'bc_type=periodic' requires we feed
    #in a closed curve, and so really we are interpolating 
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

#gives central difference derivative of a sequence (difference rather than derivative, dx = 1)
def dps(seq, psis=False):
    out = np.zeros(len(seq))
    for i in range(len(seq) - 1):
        if psis and i==0:
            out[i] = (seq[i+1] - seq[i - 1] + 2 * np.pi) / 2
        else:
            out[i] = (seq[i+1] - seq[i - 1]) / 2
    if psis:
        out[-1] = (seq[0] - seq[-2] + 2 * np.pi) / 2
    else:
        out[-1] = (seq[0] - seq[-2]) / 2
    return out
"""
-- Sets of displacements, not hack version --
"""

def gen_disp_curves(curve, L, prec, vanishing_BC=False, cont=False):

    if cont:
        return gen_disp_curves_cont(curve, L, prec, vanishing_BC)

    curve_arr = np.zeros((2 * len(curve), len(curve), 2))
    vdisp_set = np.zeros((2 * len(curve), len(curve), 2))

    step_theta = L / len(curve)
    epsilon = step_theta * prec

    #for clarity, an entry is an entire displaced curve
    for i in range(2 * len(curve)):
        #bug fix: this - 0.5 makes all the difference
        vdisps_whole = np.random.rand(len(curve), 2) - 0.5
        vdisps_whole = vdisps_whole / np.linalg.norm(vdisps_whole, axis=1)[:, None]
        vdisps_whole = vdisps_whole * epsilon

        #RIGHT NOW 20 feb 17:00, we set all displacements EQUAL at RANDOM BOUNDARIES, WHICH ARE RAN
        #SELECTED FOR EACH ROW IN THE MATRIX.

        if vanishing_BC:
            index1 = 1
            index2 = index1-1
            vdisps_whole[index1] = vdisps_whole[index2]

        curve_copy = curve.copy()
        curve_copy += vdisps_whole
        
        curve_arr[i] = curve_copy
        vdisp_set[i] = vdisps_whole
    
    print('a surprise to be sure, but a welcome one')
    return curve_arr, vdisp_set

"""
Continuity issues:
"""

def get_tgt_angles_jump(curve):
    N = len(curve)
    angles = np.zeros((N))
    for i in range(1, N-1):
        tgt_vec = curve[i+1] - curve[i-1]
        tgt_vec /= np.linalg.norm(tgt_vec)

        angles[i] = np.arctan2(tgt_vec[1], tgt_vec[0])

    tgt_vec = curve[1] - curve[-1]
    tgt_vec /= np.linalg.norm(tgt_vec)
    angles[0] = np.arctan2(tgt_vec[1], tgt_vec[0])

    tgt_vec = curve[0] - curve[-2]
    tgt_vec /= np.linalg.norm(tgt_vec)
    angles[-1] = np.arctan2(tgt_vec[1], tgt_vec[0])

    return angles

def get_tgt_angles(curve):
    angles = get_tgt_angles_jump(curve)

    for i, angle in enumerate(angles):
        if i != 0:
            diff = angle - angles[i-1]
            if np.abs(diff) >= np.pi:
                sgn = 1 if diff > 0 else -1
                angles[i:] -= 2*np.pi * sgn
        
    return angles

#wrt xhat, in rad. There might be some discontinuity tho, bear in mind.

def gen_cont_disps(curve, L, prec):
    N = len(curve)
    epsilon = L * prec / N
    vdisps = np.zeros((N, 2))
    psis = get_tgt_angles(curve)

    #Note! an will be the coefficient of the term with and frequency (n+1)
    an = np.zeros((2 * N), dtype=complex)

    for j in range(2 * N):
        an[j] = np.exp(-1*(20 * j / N) ** 2) * (np.random.uniform(0, 1) - 0.5 + 1j * (np.random.uniform(0, 1)-0.5))
    
    d = np.zeros((N), dtype=complex)

    for j in range(2 * N):
        d += an[j] * np.exp(1j * (j + 1) * psis)

    vdisps[:, 0] = np.real(d)
    vdisps[:, 1] = np.imag(d)

    vdisps *= epsilon

    return vdisps

def gen_disp_curves_cont(curve, L, prec, vanishing_BC):
    N = len(curve)
    curve_arr = np.zeros((2 * N, N, 2))
    for i, _ in  enumerate(curve_arr):
        curve_arr[i] = curve.copy()
    vdisp_set = np.zeros((2 * N, N, 2))
    for i in range(2 * N):
        vdisp_set[i] = gen_cont_disps(curve, prec)

    #we set all displacements 0 at the boundaries
    if vanishing_BC:
        for i, vdisp in enumerate(vdisp_set): 
            vdisp0 = vdisp[0]
            for j, disp in enumerate(vdisp):
                vdisp_set[i, j] -= vdisp0
            vdisp_set[i, -1] = 0 #this is done to ensure it is exactly 0, due to periodicity it should be 0 approx
        
    curve_arr += vdisp_set
    return curve_arr ,vdisp_set

def populate_matrix_set(curve, L, disp_curves): 
    out = np.zeros((len(disp_curves), len(disp_curves)))
    step_theta = L / len(curve)

    for i, temp in enumerate(disp_curves):
        for j, _ in enumerate(disp_curves):
            if j % 2 == 0:
                #this term has a dg / 2 
                dg = metric_central(temp, int(j / 2), step_theta) - metric_central(curve, int(j / 2), step_theta)
                out[i, j] = dg / 2
                out[i, j] = step_theta * out[i, j]
            else:
                #this term has -kappa dg / 2 - dkappa
                dg = metric_central(temp, (int(j // 2)), step_theta) - metric_central(curve, (int(j // 2)), step_theta)
                kappa = curvature(curve, (int(j // 2)))
                dkappa = curvature(temp, (int(j // 2))) - kappa
                out[i, j] = -kappa * dg / 2 - dkappa
                out[i, j] = step_theta * out[i, j]
    return out
 
def populate_rhs_set(curve, L, vdisp_set, fns):
    rhs = np.zeros((2 * len(curve)))

    #this is the same as step theta in other functions because this is how parametrise the
    #curve, ie how we define theta
    ds = L / len(curve)

    for i, vdisp_whole in enumerate(vdisp_set):
        for j in range(len(vdisp_set) // 2):
            rhs[i] += ds * np.dot(unit_normals(curve)[j], vdisp_whole[j]) * fns[j]
            
    return rhs

#if using continuous version, set prec to 1e-4
def backward_solver_set(curve, L, N, prec=1e-7, cont=False):

    curve_copy = curve.copy()
    curve_copy = augment_curve(curve_copy, N)

    disp_curves, vdisp_set = gen_disp_curves(curve_copy, L, prec, cont)
    p = 1
    fns = p * np.ones(N)
    
    A = populate_matrix_set(curve_copy, L, disp_curves)
    b = populate_rhs_set(curve_copy, L, vdisp_set, fns)

    sol = np.linalg.solve(A, b)
    sigmas = sol[0::2]
    ms = sol[1::2]

    return sigmas, ms

def backward_solver_cont(curve, L, N, prec=1e-4, p=1):
    curve_copy = curve.copy()
    curve_copy = augment_curve(curve_copy, N)

    step_theta = L / N
    
    vdisps = np.zeros((2 * N, 2))
    first_disps = gen_cont_disps(curve_copy, L, prec=1e-4)
    vdisps[0:N] = first_disps

    second_disps = gen_cont_disps(curve_copy, L, prec=1e-4)
    vdisps[N:] = second_disps

    p = 1
    forces_perp = unit_normals(curve_copy) * p

    A = populate_matrix(curve_copy, step_theta, vdisps)
    b = populate_rhs(curve_copy, forces_perp, vdisps)

    soln = np.linalg.solve(A, b)

    sigmas = soln[0::2]
    ms = soln[1::2]

    return sigmas, ms
"""
Forward solver
"""
#As of feb 26th 2025 if you do not feed in tns XOR dtns. Feed in both or neither.
def forward_solver(num_pts, mag=0.6, tns=None, dtns=None, tol=1e-5):
    """
    Curve forward solver for input pressure + normal forces
    """
    # Set parameters and functions
    p = 1  # Pressure
    sig0 = 0.7  # Tension sigma(0)
    L_initial = 2 * np.pi  # Initial guess for length

    # Solution domain
    xmesh = np.linspace(0, 1, num_pts)

    # SETTING NORMAL FORCE PROFILE
    # Random tn profile
    NrModes = 1  # How many Fourier modes should be included
    Nmax = 5  # Maximal mode number to draw from (Nmax >= NrModes)
    N = np.random.choice(range(2, 3), NrModes, replace=False)  # Set of modes
    Magnc = mag * (np.random.rand(NrModes) - 0.5)  # Magnitudes of cosine modes
    Magns = mag * (np.random.rand(NrModes) - 0.5)  # Magnitudes of sine modes
  
    if tns is None:
        tn = lambda x: (np.real(np.exp(1j * 2 * np.pi * x[:, None] * N) @ Magnc) +
                        np.imag(np.exp(1j * 2 * np.pi * x[:, None] * N) @ Magns))

        dtn = lambda x: (np.real(1j * 2 * np.pi * (N[None, :] * np.exp(1j * 2 * np.pi * x[:, None] * N)) @ Magnc) +
                        np.imag(1j * 2 * np.pi * (N[None, :] * np.exp(1j * 2 * np.pi * x[:, None] * N)) @ Magns))
    elif tns is not None and dtns is not None:
        tn = tns
        dtn = dtns
    else: #I'm pretty sure there is some bug in this logic!!! like the interpolation is messing things up
        dtns = np.gradient(tns, 1/num_pts) 

        #ideally switch to periodic interpolation with the function 'interpolate_seq_periodic'.
        tn = interp1d(xmesh, tns, kind='cubic')
        dtn = interp1d(xmesh, dtns, kind='cubic') #gotta fix and gotta make sure it is evenly spaced (woulda thought this was but whatevs)

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
    sol = solve_bvp(bvpfcn, bcfcn, solinit[0], solinit[1], tol=tol)

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

    # Normal moment plot #FIX: sign wrong??
    ms = sint.cumulative_trapezoid(tn(sol.x), dx=sol.y[4, 0]/len(sol.y[1]))
    plt.subplot(2, 2, 3)
    plt.plot(sol.x[:len(ms)], ms, linewidth=2)
    plt.title('ms')

    # Psi plot
    plt.subplot(2, 2, 4)
    plt.plot(sol.x, sol.y[0], linewidth=2)
    plt.title('Psi')

    # # Curvature plot
    # kappa = ((sol.y[4, 0] * p + dtn(sol.x)) / sol.y[1]) / sol.y[4, 0]
    # plt.subplot(2, 2, 5)
    # plt.plot(sol.x, kappa, linewidth=2)
    # plt.title('Curvature')

    plt.tight_layout()
    plt.show()

    print("Length:", sol.y[4, 0])

    out = np.zeros((len(sol.y) + 1, len(sol.y[0])))
    out[:-1] = sol.y
    out[-1] = tn(sol.x)

    return out
 
def get_integral_shape(sol):
    L = sol[4, 0]
    psis_actual = sol[0]
    kappas = dps(psis_actual, psis=True) / (L / len(psis_actual))
    sigmas = sol[1]
    tns = sol[-1]
    ms = sint.cumulative_trapezoid(tns, dx=sol[4, 0]/len(sol[1]))
    
    curve = sol[2:4]
    curve = curve.T

    intx = sint.trapezoid((sigmas * np.cos(psis_actual) + tns * np.sin(psis_actual)), dx = L/len(psis_actual))
    inty = sint.trapezoid((sigmas * np.sin(psis_actual) - tns * np.cos(psis_actual)), dx = L/len(psis_actual))
    intz = sint.trapezoid(ms, dx = L/len(psis_actual))

    return intx, inty, intz
