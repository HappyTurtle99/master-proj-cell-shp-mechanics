import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sint
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from helper_functions import *

def gen_disp_curves(curve, L, prec=1e-7):
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

        curve_copy = curve.copy()
        curve_copy += vdisps_whole
        
        curve_arr[i] = curve_copy
        vdisp_set[i] = vdisps_whole

    return curve_arr, vdisp_set

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

def backward_solver_set(curve, L, N, prec=1e-7):

    curve_copy = curve.copy()
    curve_copy = augment_curve(curve_copy, N)

    disp_curves, vdisp_set = gen_disp_curves(curve_copy, L, prec)
    p = 1
    fns = p * np.ones(N)
    
    A = populate_matrix_set(curve_copy, L, disp_curves)
    b = populate_rhs_set(curve_copy, L, vdisp_set, fns)

    sol = np.linalg.solve(A, b)
    sigmas = sol[0::2]
    ms = sol[1::2]

    return sigmas, ms