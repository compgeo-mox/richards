import numpy as np
import scipy.integrate as integrate

# Integrate over a triangle of vertices [0,0], [1,0] and [0,1]
def triangle_integration(func, quad_order):
    integrand = lambda ys,x: np.array([func(x,y) for y in np.array(ys)])
    inside = lambda xs, n: np.array([integrate.fixed_quad(integrand, 0, 1-x, args=(x,), n=n)[0] for x in np.array(xs)])
    return integrate.fixed_quad(inside, 0, 1, n=quad_order, args=(quad_order,))[0]

def exp_triangle_integration(func, quad_order, x0, x1, x2, m):
    if m < 0:
        integrand = lambda ys,x: np.array([func(x,y) for y in np.array(ys)])
        inside = lambda xs, n: np.array([integrate.fixed_quad(integrand, 
                                                              x1[1] + (x2[1] - x1[1]) / (x2[0] - x1[0]) * (x - x1[0]), 
                                                              x2[1], args=(x,), n=n)[0] for x in np.array(xs)])
        return integrate.fixed_quad(inside, x0[0], x2[0], n=quad_order, args=(quad_order,))[0]
    else:
        integrand = lambda ys,x: np.array([func(x,y) for y in np.array(ys)])
        inside = lambda xs, n: np.array([integrate.fixed_quad(integrand, 
                                                              x2[1], 
                                                              x2[1] + (x1[1] - x2[1]) / (x1[0] - x2[0]) * (x - x2[0]), args=(x,), n=n)[0] for x in np.array(xs)])
        return integrate.fixed_quad(inside, x2[0], x0[0], n=quad_order, args=(quad_order,))[0]