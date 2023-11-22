import numpy as np
from utilities.triangle_integration import triangle_integration, exp_triangle_integration

# Helper function used obtain the ordering of the points of a right triangle from the node opposite to the hypotenuse in a clockwise manner.
def find_ordering(coord: np.array):
    lx = np.argmin(coord[0, :])
    rx = np.argmax(coord[0, :])
    mx = np.setdiff1d(np.array([0,1,2]), np.array([lx, rx]))[0]

    # Vertical Alignment
    if np.abs( coord[0, lx] - coord[0, mx] ) < 1e-7:
        # lx and mx vertical aligned, rx no
        up =   lx if np.argmax(coord[1, np.array([lx, mx])]) == 0 else mx
        down = lx if np.argmin(coord[1, np.array([lx, mx])]) == 0 else mx

        if np.abs( coord[1, up] - coord[1, rx] ) < 1e-7:
            return [up, down, rx]
        else:
            return [down, rx, up]
    else:
        # rx and mx vertical aligned, lx no
        up =   rx if np.argmax(coord[1, np.array([rx, mx])]) == 0 else mx
        down = rx if np.argmin(coord[1, np.array([rx, mx])]) == 0 else mx

        if np.abs( coord[1, up] - coord[1, lx] ) < 1e-7:
            return [up, lx, down]
        else:
            return [down, up, lx]
        
# Simple matrix used to compute the local contribution to the h-stifness matrix.
# We assume that the element is triangular
def local_A(coord, K_local, quad_order):
    M = np.zeros(shape=(3,3))

    ordering = find_ordering(coord)

    x0 = coord[:, ordering][:, 0]
    x1 = coord[:, ordering][:, 1]
    x2 = coord[:, ordering][:, 2]
    
    J_T_1_T = np.array([[x2[1]-x0[1], x0[1]-x1[1]],
                        [x0[0]-x2[0], x1[0]-x0[0]]]) / ((x1[0]-x0[0]) * (x2[1]-x0[1]) - (x2[0]-x0[0]) * (x1[1]-x0[1]))
    

    q_funcs = [J_T_1_T @ np.array([-1, -1]), J_T_1_T @ np.array([ 1, 0]), J_T_1_T @ np.array([0,  1])]

    area = (np.max(coord[1, :]) - np.min(coord[1, :])) * (np.max(coord[0, :]) - np.min(coord[0, :]))

    for i in range(3):
            for j in range(3):
                M[ordering[i], ordering[j]] = area * triangle_integration(lambda x,y: q_funcs[i].T @ K_local(x, y) @ q_funcs[j], quad_order)

    return M

# Simple matrix used to compute the local contribution to the h-stifness matrix.
# We assume that the element is triangular
def experimental_local_A(coord, K_local, quad_order, m):
    M = np.zeros(shape=(3,3))

    ordering = find_ordering(coord)

    x0 = coord[:, ordering][:, 0]
    x1 = coord[:, ordering][:, 1]
    x2 = coord[:, ordering][:, 2]

    q_funcs = [np.array([-1/(x2[0] - x0[0]), -1/(x1[1] - x0[1])]), 
               np.array([0, 1/(x1[1] - x0[1])]), 
               np.array([1/(x2[0] - x0[0]), 0])]

    for i in range(3):
            for j in range(3):
                M[ordering[i], ordering[j]] = exp_triangle_integration(lambda x,y: q_funcs[i].T @ K_local(x,y) @ q_funcs[j], quad_order, x0, x1, x2, m)

    return M


# Simple matrix used to compute the local contribution to the h-mass matrix.
# We assume that the element is triangular
def local_Mh(coord, func, quad_order):
    ordering = find_ordering(coord)

    x0 = coord[:, ordering][:, 0]
    x1 = coord[:, ordering][:, 1]
    x2 = coord[:, ordering][:, 2]

    qs = [(lambda x,y: 1-x-y), (lambda x,y: x), (lambda x,y: y)]
    
    jacobian = (x1[0]-x0[0]) * (x2[1]-x0[1]) - (x2[0]-x0[0]) * (x1[1]-x0[1])
    M = np.zeros(shape=(3,3))

    for i in range(3):
        for j in range(3):
            M[ ordering[i], ordering[j] ] = jacobian * triangle_integration(lambda x,y: qs[j](x,y) * qs[i](x,y) * func(x, y), quad_order)

    return M


# Simple matrix used to compute the local contribution to the h-mass matrix.
# We assume that the element is triangular
def experimental_local_Mh(coord, func, quad_order, m):
    ordering = find_ordering(coord)

    x0 = coord[:, ordering][:, 0]
    x1 = coord[:, ordering][:, 1]
    x2 = coord[:, ordering][:, 2]

    qs = [(lambda x,y: 1 - (y-x0[1])/(x1[1] - x0[1]) - (x-x0[0])/(x2[0] - x0[0])), 
          (lambda x,y: (y-x0[1])/(x1[1] - x0[1])), 
          (lambda x,y: (x-x0[0])/(x2[0] - x0[0]))]
    
    M = np.zeros(shape=(3,3))

    for i in range(3):
        for j in range(3):
            M[ ordering[i], ordering[j] ] = exp_triangle_integration(lambda x,y: qs[j](x,y) * qs[i](x,y) * func(x, y), quad_order, x0, x1, x2, m)

    return M


# Simple helper function used to compute the value of eta and the actual height at (x,y) of the given reference element (whose vertices are (0,0), (1,0) and (0,1)).
def compute_eta_x3(base_height: float, element_height: float, m: float, ls_eta: float, rs_eta: float, x: float, y: float):
    coord = lambda t: ((m+1) * (1-t) - (m-1) * t) / 2

    return (1-coord(y)) * ls_eta + coord(y) * rs_eta, base_height + (1 - coord(x)) * element_height