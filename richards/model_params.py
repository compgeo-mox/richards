import pygeon
import porepy
import sympy as sp
import numpy as np

class Model_Data:
    """
    Simple class that stores the parameters required to perform a Richard simulation (from the phisical parameters, to the solver ones)
    """
    def __init__(self, 
                 theta_r, theta_s, alpha, n, K_s, 
                 T, num_steps) -> None:
        self.theta_r = theta_r
        self.theta_s = theta_s

        self.alpha = alpha

        self.n = n
        self.K_s = K_s

        self.T    = T

        self.theta_m = theta_s
        self.m = 1 - 1/n

        self.num_steps = num_steps
        self.dt   = (T-0) / self.num_steps

        self.derivative_theta = list()
        self.derivative_K = list()
        self.derivative_K_inv = list()

        self.h_var = sp.Symbol('h', negative=True)
        self.z_var = sp.Symbol('z', negative=True)

        self.theta_expression = self.theta_r + (self.theta_s - self.theta_r) / (1 + (-self.alpha * (self.h_var - self.z_var)) ** self.n) ** self.m
        
        self.effective_saturation = (self.theta_expression - theta_r) / (theta_s - theta_r)
        self.hydraulic_conductivity_expression = K_s * (self.effective_saturation ** 0.5) * ( 1 - (1 - self.effective_saturation ** (1 / self.m)) ** self.m ) ** 2


    def __theta_setup(self, order):
        """
        Prepare the theta and its derivatives up to the specified order
        """
        fixed_len = len(self.derivative_theta)

        for od in range( order - fixed_len + 1):
            self.derivative_theta.append( sp.lambdify([self.h_var, self.z_var], sp.diff(self.theta_expression, self.h_var, fixed_len + od), 'numpy') )

    def __hydraulic_setup(self, order):
        """
        Prepare the hydraulic conductivity coefficient and its derivatives up to the specified order
        """
        fixed_len = len(self.derivative_K)
        
        for od in range( order - fixed_len + 1):
            self.derivative_K.append( sp.lambdify([self.h_var, self.z_var], sp.diff(self.hydraulic_conductivity_expression, self.h_var, fixed_len + od), 'numpy') )

    def __inv_hydraulic_setup(self, order):
        """
        Prepare the inverse hydraulic conductivity coefficient and its derivatives up to the specified order
        """
        fixed_len = len(self.derivative_K_inv)
        
        for od in range( order - fixed_len + 1):
            self.derivative_K_inv.append( sp.lambdify([self.h_var, self.z_var], sp.diff(1 / self.hydraulic_conductivity_expression, self.h_var, fixed_len + od), 'numpy') )

    
    def __internal_theta(self, h, z, order):
        """
        It will return the derivative of theta of the specified order evaluated in the center of each elemet
        """
        mask = np.zeros_like(h, dtype=bool)
        mask = h < z

        if order == 0:
            res = np.ones_like(h) * self.theta_s
        else:
            res = np.zeros_like(h)

        res[mask] = self.derivative_theta[order](h[mask], z[mask])

        return res
    
    
    def __internal_hydraulic(self, h, z, order):
        """
        It will return the derivative of hydraulic conductivity of the specified order evaluated in the center of each elemet
        """
        mask = np.zeros_like(h, dtype=bool)
        mask = h < z

        if order == 0:
            res = np.ones_like(h) * self.K_s
        else:
            res = np.zeros_like(h)

        res[mask] = self.derivative_K[order](h[mask], z[mask])

        return res
    
    
    def __internal_inv_hydraulic(self, h, z, order):
        """
        It will return the derivative of inverse hydraulic conductivity of the specified order evaluated in the center of each elemet
        """
        mask = np.zeros_like(h, dtype=bool)
        mask = h < z

        if order == 0:
            res = np.ones_like(h) / self.K_s
        else:
            res = np.zeros_like(h)

        res[mask] = self.derivative_K_inv[order](h[mask], z[mask])

        return res


    def theta(self, h, z, order = 0):
        """
        It will return theta (or one of its derivatives) evaluated in the cell centers.
        """
        if len(self.derivative_theta) <= order:
            self.__theta_setup(order)

        return self.__internal_theta(h, z, order)


    def hydraulic_conductivity_coefficient(self, h, z, order = 0):
        """
        It will return the hydraulic conductivity (or one of its derivatives) evaluated in the cell centers.
        """
        if len(self.derivative_K) <= order:
            self.__hydraulic_setup(order)

        return self.__internal_hydraulic(h, z, order)


    def inverse_hydraulic_conductivity_coefficient(self, h, z, order = 0):
        """
        It will return the inverse hydraulic conductivity (or one of its derivatives) evaluated in the cell centers.
        """
        if len(self.derivative_K_inv) <= order:
            self.__inv_hydraulic_setup(order)

        return self.__internal_inv_hydraulic(h, z, order)