from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
from scipy import integrate as sint

#create curve using fwd solver
sol = forward_solver(mag=0.01)
curve = sol.y[2:4]
curvature = sol.y[0:2]
L = sol.y[4, 0]
curve = curve.T
curve[-1] = curve[0]

N = len(curve)

curve_copy = curve.copy()
sigmas, ms = backward_solver(curve_copy, L, N, 0.00001)
tns = sint.cumulative_trapezoid(ms, dx=1, initial=0)
tns = np.append(tns, tns[-1])

sol = forward_solver(tns=tns)
curve = sol.y[2:4]
curvature = sol.y[0:2]
L = sol.y[4, 0]
curve = curve.T[:-1]

print(tns)