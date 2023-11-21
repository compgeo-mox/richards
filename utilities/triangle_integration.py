import numpy as np
import scipy.integrate as integrate

# Integrate over a triangle of vertices [0,0], [1,0] and [0,1]
def triangle_integration(func, quad_order):
    integrand = lambda ys,x: np.array([func(x,y) for y in np.array(ys)])
    inside = lambda xs, n: np.array([integrate.fixed_quad(integrand, 0, 1-x, args=(x,), n=n)[0] for x in np.array(xs)])
    return integrate.fixed_quad(inside, 0, 1, n=quad_order, args=(quad_order,))[0]