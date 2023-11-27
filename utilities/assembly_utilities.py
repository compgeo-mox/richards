import numpy as np
from utilities.triangle_integration import exp_triangle_integration

# Helper function used obtain the ordering of the points of a right triangle from the node opposite to the hypotenuse in a clockwise manner.
def find_ordering(coord: np.array, tolerance=1e-7):
    lx = np.argmin(coord[0, :])
    rx = np.argmax(coord[0, :])
    mx = np.setdiff1d(np.array([0,1,2]), np.array([lx, rx]))[0]

    # Vertical Alignment
    if np.abs( coord[0, lx] - coord[0, mx] ) < tolerance:
        # lx and mx vertical aligned, rx no
        up =   lx if np.argmax(coord[1, np.array([lx, mx])]) == 0 else mx
        down = lx if np.argmin(coord[1, np.array([lx, mx])]) == 0 else mx

        if np.abs( coord[1, up] - coord[1, rx] ) < tolerance:
            return [up, down, rx], -1
        else:
            return [down, rx, up], -1
    else:
        # rx and mx vertical aligned, lx no
        up =   rx if np.argmax(coord[1, np.array([rx, mx])]) == 0 else mx
        down = rx if np.argmin(coord[1, np.array([rx, mx])]) == 0 else mx

        if np.abs( coord[1, up] - coord[1, lx] ) < tolerance:
            return [up, lx, down], 1
        else:
            return [down, up, lx], 1
        
def transoform_nodal_func_to_physical_element(nodal_values, coord):
    ordering, _ = find_ordering(coord)
    
    ordered_values = nodal_values[ordering]
    ordered_coords = coord[:, ordering]

    return lambda x,y: (ordered_values[0] + 
                       (ordered_values[2] - ordered_values[0]) * (x - ordered_coords[0,0]) / (ordered_coords[0,2] - ordered_coords[0,0]) + 
                       (ordered_values[1] - ordered_values[0]) * (y - ordered_coords[1,0]) / (ordered_coords[1,1] - ordered_coords[1,0]))



# Simple matrix used to compute the local contribution to the h-stifness matrix.
# We assume that the element is triangular
def local_A(coord, K_local, quad_order):
    M = np.zeros(shape=(3,3))

    ordering, m = find_ordering(coord)

    ordered_coords = coord[:, ordering]

    x0 = ordered_coords[:, 0]
    x1 = ordered_coords[:, 1]
    x2 = ordered_coords[:, 2]

    q_funcs = [np.array([-1/(x2[0] - x0[0]), -1/(x1[1] - x0[1])]), 
               np.array([0, 1/(x1[1] - x0[1])]), 
               np.array([1/(x2[0] - x0[0]), 0])]

    for i in range(3):
            for j in range(3):
                M[i, j] = exp_triangle_integration(lambda x,y: q_funcs[i].T @ K_local(x,y) @ q_funcs[j], quad_order, x0, x1, x2, m)


    sort = np.argsort(ordering)
    return M[sort, :][:, sort]



# Simple matrix used to compute the local contribution to the h-mass matrix.
# We assume that the element is triangular
def local_Mh(coord, func, quad_order):
    ordering, m = find_ordering(coord)

    x0 = coord[:, ordering][:, 0]
    x1 = coord[:, ordering][:, 1]
    x2 = coord[:, ordering][:, 2]

    qs = [(lambda x,y: 1 - (y-x0[1])/(x1[1] - x0[1]) - (x-x0[0])/(x2[0] - x0[0])), 
          (lambda x,y: (y-x0[1])/(x1[1] - x0[1])), 
          (lambda x,y: (x-x0[0])/(x2[0] - x0[0]))]
    
    M = np.zeros(shape=(3,3))
    sort = np.argsort(ordering)

    for i in range(3):
        for j in range(i):
            M[i, j] = M[j, i]

        for j in range(i, 3):
            M[i, j] = exp_triangle_integration(lambda x,y: qs[j](x,y) * qs[i](x,y) * func(x, y), quad_order, x0, x1, x2, m)
        
    return M[sort, :][:, sort]


# Simple helper function used to compute the value of eta and the actual height at (x,y) of the given reference element (whose vertices are (0,0), (1,0) and (0,1)).
def compute_eta_x3(base_height: float, element_height: float, m: float, ls_eta: float, rs_eta: float, x: float, y: float):
    coord = lambda t: ((m+1) * (1-t) - (m-1) * t) / 2

    return (1-coord(y)) * ls_eta + coord(y) * rs_eta, base_height + (1 - coord(x)) * element_height